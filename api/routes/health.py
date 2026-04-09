"""
Health and readiness endpoints for the SLO Recommendation Engine API.

Notes
-----
Provides liveness and readiness probes suitable for Kubernetes and other
container orchestration health-check mechanisms.
"""
from fastapi import APIRouter

from slo_engine.config import config

router = APIRouter()


@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    """
    Return the application liveness status and version.

    Returns
    -------
    dict
        Dictionary with keys ``status`` (always ``"ok"``) and ``version``
        (read from ``config.app.version``, defaulting to ``"1.0.0"``).

    Notes
    -----
    Used by container orchestrators to determine whether the process is alive.
    Version lookup failures are silently ignored and the default is returned.
    """
    version = "1.0.0"
    try:
        version = config.app.version
    except Exception:
        pass
    return {"status": "ok", "version": version}


@router.get("/health/ready", summary="Readiness probe")
async def readiness() -> dict:
    """
    Return the application readiness status for all components.

    Returns
    -------
    dict
        Dictionary with ``status`` (``"ready"``) and ``components`` mapping
        each component name to its status string.

    Notes
    -----
    Used by container orchestrators to determine whether the application is
    ready to accept traffic. Currently returns static ``"up"`` values; extend
    this to perform real dependency checks in production.
    """
    return {"status": "ready", "components": {"api": "up", "db": "up", "cache": "up"}}
