"""
Configurable metrics adapter with source priority fallback.

Notes
-----
Source priority: Prometheus -> Datadog -> mock.
The active source is read from ``config.metrics.source``. Both real source
adapters fall back to the mock implementation when unavailable or misconfigured.
"""
from __future__ import annotations

import random
from datetime import UTC

import httpx
from loguru import logger

logger = logger.bind(name=__name__)

_MOCK_BASE: dict[str, dict] = {
    "api-gateway":      {"avail": 0.9991, "p99": 250.0, "req": 1000, "err_rate": 0.001},
    "checkout-service": {"avail": 0.9950, "p99": 800.0, "req": 500,  "err_rate": 0.010},
    "auth-service":     {"avail": 0.9998, "p99": 50.0,  "req": 3000, "err_rate": 0.0002},
}

_DEFAULT_AVAIL    = 0.990
_DEFAULT_P99      = 200.0
_DEFAULT_REQ      = 500
_DEFAULT_ERR_RATE = 0.010


def fetch_raw_metrics(service_name: str, window_days: int, config) -> dict:
    """
    Fetch raw metrics for a service from the configured source.

    Parameters
    ----------
    service_name : str
        Name of the service to fetch metrics for.
    window_days : int
        Number of days of historical data to retrieve.
    config : Dynaconf
        Application config object with a ``metrics.source`` attribute.

    Returns
    -------
    dict
        Raw metrics dictionary with keys ``avail``, ``p99``, ``req``, and ``err``,
        each containing a list of hourly values for the window.

    Notes
    -----
    Source selection:
    - ``"prometheus"`` -> queries Prometheus range API with ``httpx``.
    - ``"datadog"`` -> queries Datadog metrics API with ``httpx``.
    - Any other value or unavailable source -> returns mock data.
    Both real sources silently fall back to mock on any exception.
    """
    source = _get_source(config)
    logger.debug("fetch_raw_metrics: service={} source={} window={}d", service_name, source, window_days)

    if source == "prometheus":
        return _fetch_from_prometheus(service_name, window_days, config)
    if source == "datadog":
        return _fetch_from_datadog(service_name, window_days, config)
    return _fetch_mock(service_name, window_days)


def _get_source(config) -> str:
    """
    Read the configured metrics source string from config.

    Parameters
    ----------
    config : Dynaconf
        Application config object.

    Returns
    -------
    str
        Source string, defaulting to ``"mock"`` if the config attribute is
        missing or raises an exception.

    Notes
    -----
    Uses ``getattr`` with a default to handle both missing attributes and
    Dynaconf lazy-load exceptions gracefully.
    """
    try:
        return getattr(getattr(config, "metrics", None), "source", "mock") or "mock"
    except Exception:
        return "mock"


def _fetch_from_prometheus(service_name: str, window_days: int, config) -> dict:
    """
    Fetch metrics from Prometheus using the range query API.

    Parameters
    ----------
    service_name : str
        Service label value used in PromQL label selectors.
    window_days : int
        Number of days for the query time range.
    config : Dynaconf
        Config object with ``metrics.prometheus_url``.

    Returns
    -------
    dict
        Raw metrics dict with hourly series for avail, p99, req, and err.
        Falls back to mock data if the query fails.

    Notes
    -----
    Queries four PromQL expressions: ``up`` for availability, ``histogram_quantile``
    for p99 latency, ``http_requests_total`` for request counts, and a 5xx-filtered
    variant for error counts. Step interval is 1 hour.
    """
    try:
        base_url = getattr(getattr(config, "metrics", None), "prometheus_url", "")
        if not base_url:
            raise ValueError("config.metrics.prometheus_url not set")

        window_s = window_days * 86400
        step = "1h"
        now_ts = _now_ts()
        start_ts = now_ts - window_s

        def _query(expr: str) -> list[float]:
            url = f"{base_url.rstrip('/')}/api/v1/query_range"
            resp = httpx.get(
                url,
                params={"query": expr, "start": start_ts, "end": now_ts, "step": step},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", {}).get("result", [])
            if not results:
                return []
            values = results[0].get("values", [])
            return [float(v[1]) for v in values]

        avail_series = _query(
            f'avg_over_time(up{{service="{service_name}"}}[5m])'
        )
        p99_series = _query(
            f'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m])) * 1000'
        )
        req_series = _query(
            f'rate(http_requests_total{{service="{service_name}"}}[5m])'
        )
        err_series = _query(
            f'rate(http_requests_total{{service="{service_name}",status=~"5.."}}[5m])'
        )

        n = window_days * 24
        if not avail_series:
            raise ValueError("No availability data from Prometheus")

        return {
            "avail": avail_series[-n:] or [1.0] * n,
            "p99":   p99_series[-n:]   or [100.0] * n,
            "req":   [int(r) for r in req_series[-n:]] or [100] * n,
            "err":   [int(e) for e in err_series[-n:]] or [0] * n,
        }

    except Exception as exc:
        logger.warning("Prometheus fetch failed for '{}', falling back to mock: {}", service_name, exc)
        return _fetch_mock(service_name, window_days)


def _fetch_from_datadog(service_name: str, window_days: int, config) -> dict:
    """
    Fetch metrics from Datadog using the metrics query API.

    Parameters
    ----------
    service_name : str
        Datadog service tag value used in metric queries.
    window_days : int
        Number of days for the query time range.
    config : Dynaconf
        Config object with ``metrics.datadog_api_key`` and
        ``metrics.datadog_app_key``.

    Returns
    -------
    dict
        Raw metrics dict with hourly series for avail, p99, req, and err.
        Falls back to mock data if the query fails.

    Notes
    -----
    Queries four Datadog metric expressions: APM request hits for availability
    proxy, trace duration p99 for latency, request hits count, and error count.
    Authentication uses DD-API-KEY and DD-APPLICATION-KEY headers.
    """
    try:
        metrics_cfg = getattr(config, "metrics", None)
        api_key = getattr(metrics_cfg, "datadog_api_key", "") or ""
        app_key = getattr(metrics_cfg, "datadog_app_key", "") or ""
        if not api_key or not app_key:
            raise ValueError("config.metrics.datadog_api_key / datadog_app_key not set")

        now_ts = _now_ts()
        from_ts = now_ts - window_days * 86400
        headers = {"DD-API-KEY": api_key, "DD-APPLICATION-KEY": app_key}

        def _dd_query(metric_expr: str) -> list[float]:
            resp = httpx.get(
                "https://api.datadoghq.com/api/v1/query",
                params={"query": metric_expr, "from": from_ts, "to": now_ts},
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            series = resp.json().get("series", [])
            if not series:
                return []
            return [float(pt[1]) for pt in series[0].get("pointlist", [])]

        n = window_days * 24
        avail_series = _dd_query(f"avg:trace.web.request.hits{{service:{service_name}}}.as_count()")
        if not avail_series:
            raise ValueError("No data from Datadog")

        p99_series   = _dd_query(f"p99:trace.web.request.duration{{service:{service_name}}} * 1000")
        req_series   = _dd_query(f"sum:trace.web.request.hits{{service:{service_name}}}.as_count()")
        err_series   = _dd_query(f"sum:trace.web.request.errors{{service:{service_name}}}.as_count()")

        return {
            "avail": avail_series[-n:] or [1.0] * n,
            "p99":   p99_series[-n:]   or [100.0] * n,
            "req":   [int(r) for r in req_series[-n:]] or [100] * n,
            "err":   [int(e) for e in err_series[-n:]] or [0] * n,
        }

    except Exception as exc:
        logger.warning("Datadog fetch failed for '{}', falling back to mock: {}", service_name, exc)
        return _fetch_mock(service_name, window_days)


def _fetch_mock(service_name: str, window_days: int = 30) -> dict:
    """
    Generate synthetic mock metrics for a service.

    Parameters
    ----------
    service_name : str
        Service name used to look up base mock parameters.
    window_days : int, optional
        Number of days of hourly data to generate. Defaults to 30.

    Returns
    -------
    dict
        Synthetic metrics dict with keys ``avail``, ``p99``, ``req``, and ``err``,
        each containing ``window_days * 24`` hourly values.

    Notes
    -----
    Known services (``api-gateway``, ``checkout-service``, ``auth-service``) use
    their calibrated base parameters. Unknown services use default parameters.
    Gaussian noise is added to availability and p99. Request counts use uniform
    noise. Error counts are derived from error rate with additional Gaussian noise.
    """
    base = _MOCK_BASE.get(service_name)
    n = window_days * 24

    if base:
        avail_mean  = base["avail"]
        p99_mean    = base["p99"]
        req_base    = base["req"]
        err_mean    = base["err_rate"]
    else:
        avail_mean  = _DEFAULT_AVAIL
        p99_mean    = _DEFAULT_P99
        req_base    = _DEFAULT_REQ
        err_mean    = _DEFAULT_ERR_RATE

    avail   = [max(0.0, min(1.0, avail_mean + random.gauss(0, 2e-3)))  for _ in range(n)]
    p99     = [max(1.0, p99_mean + random.gauss(0, p99_mean * 0.1))    for _ in range(n)]
    req     = [max(1, int(req_base + random.randint(-req_base // 10, req_base // 10))) for _ in range(n)]
    err     = [max(0, int(r * err_mean + random.gauss(0, r * err_mean * 0.2)))         for r in req]

    return {"avail": avail, "p99": p99, "req": req, "err": err}


def _now_ts() -> int:
    """
    Return the current UTC timestamp as an integer Unix epoch.

    Returns
    -------
    int
        Current UTC time as integer seconds since the Unix epoch.

    Notes
    -----
    Used to compute Prometheus and Datadog query time ranges.
    """
    from datetime import datetime
    return int(datetime.now(tz=UTC).timestamp())
