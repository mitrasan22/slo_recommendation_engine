"""
Integration layer routes for the SLO Recommendation Engine.

Notes
-----
Provides platform-agnostic catalog ingestion, webhook registration, SLO polling,
and A2A agent card discovery endpoints. Supports Backstage, Port, Cortex, and
generic service catalog formats.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from slo_engine.integrations.catalog_adapter import (
    detect_platform,
    from_backstage,
    from_cortex,
    from_generic,
    from_port,
)
from slo_engine.integrations.webhook_sink import (
    get_registered_webhooks,
    push_slo_result,
    register_webhook,
)

logger = logger.bind(name=__name__)

router = APIRouter()
agent_card_router = APIRouter()

_catalog_cache: dict[str, list[dict]] = {}

_slo_results: dict[str, dict] = {}


_AGENT_CARD = {
    "name": "SLO Recommendation Engine",
    "description": (
        "Analyzes service dependency graphs and generates MILP-optimized SLO "
        "recommendations with Bayesian confidence scoring."
    ),
    "url": "/api/v1",
    "version": "1.0.0",
    "provider": {"organization": "Platform Engineering"},
    "capabilities": {"streaming": False, "pushNotifications": True},
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": [
        {
            "id": "slo_recommend",
            "name": "Generate SLO Recommendation",
            "description": (
                "Full pipeline: dependency graph → Bayesian metrics → "
                "MILP-optimized SLOs with confidence scoring"
            ),
            "tags": ["slo", "reliability", "sre"],
        },
        {
            "id": "slo_review",
            "name": "Human Review",
            "description": (
                "Approve, reject, or modify a pending low-confidence SLO recommendation"
            ),
            "tags": ["review", "hitl"],
        },
        {
            "id": "catalog_ingest",
            "name": "Ingest Service Catalog",
            "description": (
                "Accept service dependency graph from Backstage, Port, Cortex, "
                "or any custom format"
            ),
            "tags": ["catalog", "integration"],
        },
    ],
    "authentication": {"schemes": ["Bearer", "None"]},
}


@agent_card_router.get("/.well-known/agent.json", tags=["discovery"], summary="A2A Agent Card")
async def get_agent_card() -> dict:
    """
    Return the A2A agent card for service discovery.

    Returns
    -------
    dict
        Agent card dictionary conforming to the A2A discovery specification,
        including name, capabilities, skills, and authentication schemes.

    Notes
    -----
    Served at ``/.well-known/agent.json`` without the ``/api/v1`` prefix to
    conform to the A2A discovery standard.
    """
    return _AGENT_CARD


class CatalogIngestRequest(BaseModel):
    """
    Request body for catalog ingestion.

    Attributes
    ----------
    platform : str or None
        Source platform identifier (``"backstage"``, ``"port"``, ``"cortex"``,
        ``"generic"``). If omitted, platform is auto-detected.
    entities : list of dict
        Raw entity list from the source platform.

    Notes
    -----
    Platform auto-detection uses heuristic field inspection via
    ``detect_platform``.
    """

    platform: str | None = None
    entities: list[dict]


_ADAPTERS = {
    "backstage": from_backstage,
    "port":      from_port,
    "cortex":    from_cortex,
    "generic":   from_generic,
}


@router.post("/catalog/ingest", summary="Ingest service catalog from any platform")
async def ingest_catalog(body: CatalogIngestRequest) -> dict:
    """
    Ingest a service catalog and normalise it into the internal format.

    Parameters
    ----------
    body : CatalogIngestRequest
        Request body containing the source platform and raw entity list.

    Returns
    -------
    dict
        Dictionary with ``platform``, ``services_ingested`` count, and
        ``service_graph`` list of normalised service dictionaries.

    Notes
    -----
    Normalised services are stored in the module-level ``_catalog_cache``
    keyed by service name so the agent pipeline can retrieve them without
    re-ingesting. If platform detection or adapter normalisation fails, an
    appropriate HTTP error is raised.
    """
    platform = body.platform
    if not platform:
        platform = detect_platform({"entities": body.entities})
        logger.info("Platform auto-detected as '{}'", platform)

    adapter = _ADAPTERS.get(platform)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown platform '{platform}'. Valid: backstage, port, cortex, generic.",
        )

    try:
        normalized = adapter(body.entities)
    except Exception as exc:
        logger.error("Catalog adapter '{}' failed: {}", platform, exc)
        raise HTTPException(status_code=422, detail=str(exc))

    for svc in normalized:
        _catalog_cache[svc["service"]] = svc

    logger.info("Catalog ingested via '{}': {} services", platform, len(normalized))
    return {
        "platform": platform,
        "services_ingested": len(normalized),
        "service_graph": normalized,
    }


def get_catalog_cache() -> dict[str, list[dict]]:
    """
    Return the module-level catalog cache.

    Returns
    -------
    dict
        Mapping of service name to normalised service dictionary as last
        populated by ``ingest_catalog``.

    Notes
    -----
    Intended for use by the agent pipeline to retrieve the most recently
    ingested catalog without repeating the ingestion HTTP call.
    """
    return _catalog_cache


class WebhookRegisterRequest(BaseModel):
    """
    Request body for webhook registration.

    Attributes
    ----------
    service_name : str
        Name of the service whose SLO results will be pushed to the webhook.
    callback_url : str
        HTTPS URL that will receive POST requests with SLO result payloads.
    secret : str
        Optional HMAC signing secret for payload verification. Defaults to
        an empty string (unsigned).

    Notes
    -----
    The secret is stored and used by ``push_slo_result`` to compute an
    HMAC-SHA256 signature in the ``X-SLO-Signature`` request header.
    """

    service_name: str
    callback_url: str
    secret: str = ""


@router.post("/webhooks/register", summary="Register webhook callback for a service")
async def register_webhook_endpoint(body: WebhookRegisterRequest) -> dict:
    """
    Register a webhook callback URL for a service.

    Parameters
    ----------
    body : WebhookRegisterRequest
        Request body with service name, callback URL, and optional secret.

    Returns
    -------
    dict
        Registration confirmation dictionary from the webhook sink.

    Notes
    -----
    Delegates to ``slo_engine.integrations.webhook_sink.register_webhook``.
    """
    return register_webhook(body.service_name, body.callback_url, body.secret)


@router.get("/webhooks", summary="List all registered webhooks")
async def list_webhooks() -> list[dict]:
    """
    Return all registered webhook entries.

    Returns
    -------
    list of dict
        List of registered webhook dictionaries, each with ``service_name``,
        ``callback_url``, and ``registered_at`` fields.

    Notes
    -----
    Delegates to ``slo_engine.integrations.webhook_sink.get_registered_webhooks``.
    """
    return get_registered_webhooks()


@router.get("/slo/{service_name}", summary="Poll SLO status for a service")
async def get_slo_status(service_name: str) -> dict:
    """
    Poll the SLO result for a specific service.

    Parameters
    ----------
    service_name : str
        Name of the service to retrieve the SLO result for.

    Returns
    -------
    dict
        SLO result dictionary previously stored by the agent pipeline.

    Notes
    -----
    Results are stored in the module-level ``_slo_results`` dictionary by
    ``store_slo_result``. Raises HTTP 404 if no result has been stored for
    the requested service.
    """
    result = _slo_results.get(service_name)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No SLO result found for service '{service_name}'. "
                   "Run the pipeline first or check the service name.",
        )
    return result


def store_slo_result(service_name: str, result: dict) -> None:
    """
    Store an SLO result so it can be retrieved by the polling endpoint.

    Parameters
    ----------
    service_name : str
        Name of the service the result belongs to.
    result : dict
        SLO result dictionary produced by the agent pipeline after the
        HITL gate.

    Returns
    -------
    None

    Notes
    -----
    Called by the agent pipeline after the final gate step. Results are
    held in the module-level ``_slo_results`` dictionary for the lifetime
    of the server process.
    """
    _slo_results[service_name] = result
