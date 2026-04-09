"""
SLOs API for accepting, modifying, and tracking active SLOs.

Notes
-----
Provides REST endpoints for creating and retrieving SLO records associated
with services. SLOs can be set directly or accepted from recommendations
produced by the agent pipeline.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from slo_engine.db.database import get_db
from slo_engine.db.models import SLO, SLOStatus, Service

router = APIRouter()


class SLOIn(BaseModel):
    """
    Request body for setting or accepting an SLO for a service.

    Attributes
    ----------
    availability_target : float or None
        Target availability expressed as a fraction between 0.0 and 1.0.
    latency_percentile : float or None
        Latency percentile for the latency target (e.g. 0.99 for p99).
    latency_target_ms : float or None
        Target latency threshold in milliseconds, must be strictly positive.
    error_rate_target : float or None
        Target error rate as a fraction between 0.0 and 1.0.
    window_days : int
        Rolling window duration in days. Must be between 1 and 365.
        Defaults to 30.
    name : str
        Human-readable SLO name. Defaults to ``"default"``.

    Notes
    -----
    At least one of ``availability_target``, ``latency_target_ms``, or
    ``error_rate_target`` should be provided for the SLO to be meaningful.
    """

    availability_target: float | None = Field(None, ge=0.0, le=1.0)
    latency_percentile: float | None = Field(None, ge=0.0, le=1.0)
    latency_target_ms: float | None = Field(None, gt=0)
    error_rate_target: float | None = Field(None, ge=0.0, le=1.0)
    window_days: int = Field(default=30, ge=1, le=365)
    name: str = "default"


@router.post(
    "/services/{service_id}/slos",
    status_code=201,
    summary="Accept or set an SLO for a service",
)
async def set_slo(
    service_id: str,
    body: SLOIn,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Create or accept an SLO record for a service.

    Parameters
    ----------
    service_id : str
        Unique name of the service to attach the SLO to.
    body : SLOIn
        SLO target values and configuration.
    db : AsyncSession
        Injected async SQLAlchemy database session.

    Returns
    -------
    dict
        Created SLO record with id, service name, availability target,
        latency target, status, and created_at timestamp.

    Notes
    -----
    Raises HTTP 404 if no service with the given name exists. The new SLO
    is created with ``ACTIVE`` status and flushed (but not committed) so that
    the returned id is available immediately.
    """
    result = await db.execute(select(Service).where(Service.name == service_id))
    svc = result.scalar_one_or_none()
    if not svc:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")

    slo = SLO(
        service_id=svc.id,
        name=body.name,
        status=SLOStatus.ACTIVE,
        availability_target=body.availability_target,
        latency_percentile=body.latency_percentile,
        latency_target_ms=body.latency_target_ms,
        error_rate_target=body.error_rate_target,
        window_days=body.window_days,
    )
    db.add(slo)
    await db.flush()

    return {
        "id": str(slo.id),
        "service": service_id,
        "availability_target": slo.availability_target,
        "latency_target_ms": slo.latency_target_ms,
        "status": slo.status.value,
        "created_at": slo.created_at.isoformat() if slo.created_at else None,
    }


@router.get("/services/{service_id}/slos", summary="Get active SLOs for a service")
async def get_slos(service_id: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    """
    Return all active SLOs for a service.

    Parameters
    ----------
    service_id : str
        Unique name of the service to retrieve SLOs for.
    db : AsyncSession
        Injected async SQLAlchemy database session.

    Returns
    -------
    list of dict
        List of active SLO dictionaries, each containing id, name,
        availability_target, latency_target_ms, window_days,
        error_budget_remaining, and burn_rate_1h.

    Notes
    -----
    Raises HTTP 404 if no service with the given name exists. Only SLOs
    with status ``ACTIVE`` are returned.
    """
    result = await db.execute(
        select(Service).where(Service.name == service_id)
    )
    svc = result.scalar_one_or_none()
    if not svc:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")

    slos_result = await db.execute(
        select(SLO).where(SLO.service_id == svc.id, SLO.status == SLOStatus.ACTIVE)
    )
    slos = slos_result.scalars().all()
    return [
        {
            "id": str(s.id),
            "name": s.name,
            "availability_target": s.availability_target,
            "latency_target_ms": s.latency_target_ms,
            "window_days": s.window_days,
            "error_budget_remaining": s.error_budget_remaining,
            "burn_rate_1h": s.burn_rate_1h,
        }
        for s in slos
    ]
