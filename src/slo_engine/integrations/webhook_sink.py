"""
Webhook sink for pushing approved SLO results to registered platform callbacks.

Notes
-----
In production, replace the in-memory ``_webhooks`` dict with a Redis hash for
persistence across restarts. The outbound payload follows a fixed schema that
any platform can consume without coupling to engine internals. HMAC-SHA256
signatures are added when a secret is registered for the webhook endpoint.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime

import httpx
from loguru import logger

logger = logger.bind(name=__name__)

_webhooks: dict[str, dict] = {}


def register_webhook(service_name: str, callback_url: str, secret: str = "") -> dict:
    """
    Register a callback URL for a service.

    Parameters
    ----------
    service_name : str
        Name of the service whose SLO events should be pushed.
    callback_url : str
        HTTP endpoint URL to POST approved SLO payloads to.
    secret : str, optional
        HMAC-SHA256 signing secret for payload verification. Defaults to ``""``.

    Returns
    -------
    dict
        Registration confirmation with ``registered``, ``service_name``,
        and ``callback_url`` fields.

    Notes
    -----
    Overwrites any existing registration for ``service_name`` silently.
    The platform should call this once during initial setup.
    """
    _webhooks[service_name] = {"url": callback_url, "secret": secret}
    logger.info("Webhook registered for '{}' -> {}", service_name, callback_url)
    return {"registered": True, "service_name": service_name, "callback_url": callback_url}


def get_registered_webhooks() -> list[dict]:
    """
    Return a list of all registered webhooks.

    Returns
    -------
    list of dict
        Each dict has ``service_name`` and ``callback_url`` keys.

    Notes
    -----
    Secrets are not included in the returned list to avoid accidental exposure
    in API responses.
    """
    return [{"service_name": k, "callback_url": v["url"]} for k, v in _webhooks.items()]


def _build_payload(service_name: str, slo_result: dict) -> dict:
    """
    Build the standardised outbound webhook payload.

    Parameters
    ----------
    service_name : str
        Name of the service the SLO applies to.
    slo_result : dict
        SLO result dict from the recommendation pipeline.

    Returns
    -------
    dict
        Standardised webhook payload with ``event``, ``service_name``, ``slo``,
        ``status``, ``review_id``, ``approved``, and ``timestamp`` fields.

    Notes
    -----
    The payload schema is stable across engine versions to avoid breaking
    platform integrations. Only fields from the fixed schema are included.
    """
    return {
        "event": "slo.approved",
        "service_name": service_name,
        "slo": {
            "availability": slo_result.get("availability"),
            "latency_p99_ms": slo_result.get("latency_p99_ms"),
            "error_rate": slo_result.get("error_rate"),
            "confidence_score": slo_result.get("confidence_score"),
        },
        "status": slo_result.get("status"),
        "review_id": slo_result.get("review_id"),
        "approved": slo_result.get("approved"),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _build_headers(payload_bytes: bytes, secret: str) -> dict:
    """
    Build HTTP headers for a webhook POST request.

    Parameters
    ----------
    payload_bytes : bytes
        JSON-encoded payload bytes used for HMAC signature computation.
    secret : str
        HMAC-SHA256 signing secret. An empty string skips signing.

    Returns
    -------
    dict
        Headers dict with ``Content-Type`` and optionally ``X-SLO-Signature``.

    Notes
    -----
    The signature format is ``sha256=<hex_digest>`` matching the GitHub webhook
    signature convention for easy platform integration.
    """
    headers = {"Content-Type": "application/json"}
    if secret:
        sig = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
        headers["X-SLO-Signature"] = f"sha256={sig}"
    return headers


async def push_slo_result(service_name: str, slo_result: dict) -> dict:
    """
    POST the approved SLO payload to the registered webhook URL asynchronously.

    Parameters
    ----------
    service_name : str
        Name of the service whose SLO result is being pushed.
    slo_result : dict
        Approved SLO result from the recommendation pipeline.

    Returns
    -------
    dict
        Delivery result with ``pushed`` bool and either ``status_code`` and
        ``url`` on success, or ``error`` and ``url`` on failure.
        Returns ``{"pushed": False, "reason": "no_webhook_registered"}``
        when no webhook is registered for the service.

    Notes
    -----
    Uses an ``httpx.AsyncClient`` with a 10-second timeout. Exceptions are
    caught and returned as error dicts to avoid crashing the calling pipeline.
    """
    entry = _webhooks.get(service_name)
    if not entry:
        logger.debug("No webhook registered for '{}', skipping push.", service_name)
        return {"pushed": False, "reason": "no_webhook_registered"}

    payload = _build_payload(service_name, slo_result)
    body = json.dumps(payload).encode()
    headers = _build_headers(body, entry.get("secret", ""))

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(entry["url"], content=body, headers=headers)
        logger.info(
            "Webhook push for '{}' -> {} (HTTP {})", service_name, entry["url"], resp.status_code
        )
        return {"pushed": True, "status_code": resp.status_code, "url": entry["url"]}
    except Exception as exc:
        logger.error("Webhook push failed for '{}': {}", service_name, exc)
        return {"pushed": False, "error": str(exc), "url": entry["url"]}


def push_slo_result_sync(service_name: str, slo_result: dict) -> dict:
    """
    POST the approved SLO payload to the registered webhook URL synchronously.

    Parameters
    ----------
    service_name : str
        Name of the service whose SLO result is being pushed.
    slo_result : dict
        Approved SLO result from the recommendation pipeline.

    Returns
    -------
    dict
        Delivery result with ``pushed`` bool and either ``status_code`` and
        ``url`` on success, or ``error`` and ``url`` on failure.
        Returns ``{"pushed": False, "reason": "no_webhook_registered"}``
        when no webhook is registered for the service.

    Notes
    -----
    Synchronous wrapper used by ``_gate_and_finalize`` which runs in a
    synchronous context within the ADK agent loop. Uses ``httpx.Client``
    with a 10-second timeout.
    """
    entry = _webhooks.get(service_name)
    if not entry:
        logger.debug("No webhook registered for '{}', skipping push.", service_name)
        return {"pushed": False, "reason": "no_webhook_registered"}

    payload = _build_payload(service_name, slo_result)
    body = json.dumps(payload).encode()
    headers = _build_headers(body, entry.get("secret", ""))

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(entry["url"], content=body, headers=headers)
        logger.info(
            "Webhook push for '{}' -> {} (HTTP {})", service_name, entry["url"], resp.status_code
        )
        return {"pushed": True, "status_code": resp.status_code, "url": entry["url"]}
    except Exception as exc:
        logger.error("Webhook push failed for '{}': {}", service_name, exc)
        return {"pushed": False, "error": str(exc), "url": entry["url"]}
