"""
Services API for dependency ingestion and graph management.

Notes
-----
Provides REST endpoints for ingesting service dependency graphs, listing
services, retrieving full graph data in agent-ready format, and fetching
individual service details. Persists data to the async SQLAlchemy ORM layer
and runs graph analysis via the dependency agent tools.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from slo_engine.db.database import get_db
from slo_engine.db.models import DependencyType, Service, ServiceDependency, ServiceTier

router = APIRouter()


class DependencyIn(BaseModel):
    """
    A single outgoing dependency edge from a service.

    Attributes
    ----------
    name : str
        Name of the target service.
    dep_type : str
        Dependency type: ``"synchronous"``, ``"asynchronous"``, or ``"external"``.
        Defaults to ``"synchronous"``.
    weight : float
        Edge weight between 0.0 and 1.0 representing dependency strength.
        Defaults to ``1.0``.

    Notes
    -----
    Used inside ``ServiceIn.depends_on`` for inline dependency declaration.
    """

    name: str
    dep_type: str = "synchronous"
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class ServiceIn(BaseModel):
    """
    Service node with inline dependency edges for graph ingestion.

    Attributes
    ----------
    service : str
        Unique service identifier / name.
    display_name : str or None
        Human-readable service name. Defaults to the ``service`` field if omitted.
    tier : str
        Service criticality tier: ``"critical"``, ``"high"``, ``"medium"``, or ``"low"``.
        Defaults to ``"medium"``.
    team : str or None
        Owning team name.
    depends_on : list of DependencyIn
        List of outgoing dependency edges declared inline.
    p99_latency_ms : float or None
        Observed p99 latency in milliseconds used for critical path analysis.
    metadata : dict
        Arbitrary key-value metadata stored alongside the service record.

    Notes
    -----
    Flat edges from ``DependencyGraphIn.edges`` are merged with ``depends_on``
    during ingestion to produce a unified dependency list for graph analysis.
    """

    service: str
    display_name: str | None = None
    tier: str = "medium"
    team: str | None = None
    depends_on: list[DependencyIn] = Field(default_factory=list)
    p99_latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EdgeIn(BaseModel):
    """
    Flat edge format for specifying service dependencies separately.

    Attributes
    ----------
    source : str
        Source service name (field alias ``"from"``).
    target : str
        Target service name (field alias ``"to"``).
    dep_type : str
        Dependency type. Defaults to ``"synchronous"``.
    weight : float
        Edge weight. Defaults to ``1.0``.

    Notes
    -----
    Allows consumers to specify edges in a flat ``{"from": "a", "to": "b"}``
    format as an alternative to inline ``depends_on`` lists in ``ServiceIn``.
    """

    source: str = Field("", alias="from")
    target: str = Field("", alias="to")
    dep_type: str = "synchronous"
    weight: float = 1.0

    model_config = {"populate_by_name": True}


class DependencyGraphIn(BaseModel):
    """
    Full dependency graph payload with services and optional flat edges.

    Attributes
    ----------
    services : list of ServiceIn
        List of service nodes, each optionally with inline ``depends_on`` edges.
    edges : list of EdgeIn
        Flat edge list. These are merged with inline ``depends_on`` entries
        before the graph is analysed.

    Notes
    -----
    The two edge sources (inline ``depends_on`` and flat ``edges``) are
    deduplicated by target service name during ingestion.
    """

    services: list[ServiceIn]
    edges: list[EdgeIn] = Field(default_factory=list)


class ServiceOut(BaseModel):
    """
    Serialisable service record returned by list and detail endpoints.

    Attributes
    ----------
    id : str
        UUID string of the service database record.
    name : str
        Unique service identifier.
    display_name : str
        Human-readable service name.
    tier : str
        Criticality tier value.
    pagerank_score : float
        PageRank score from the most recent graph analysis.
    betweenness_centrality : float
        Betweenness centrality from the most recent graph analysis.
    fan_in : int
        Number of services that depend on this service.
    fan_out : int
        Number of services this service depends on.

    Notes
    -----
    Graph metrics are updated by the dependency agent after each ingestion.
    """

    id: str
    name: str
    display_name: str
    tier: str
    pagerank_score: float
    betweenness_centrality: float
    fan_in: int
    fan_out: int


@router.post(
    "/services/dependencies",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest service dependency graph",
)
async def ingest_dependencies(
    payload: DependencyGraphIn,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Ingest or update the service dependency graph.

    Parameters
    ----------
    payload : DependencyGraphIn
        Dependency graph with services and optional flat edges.
    db : AsyncSession
        Injected async SQLAlchemy database session.

    Returns
    -------
    dict
        Ingestion result including service count, edge count, circular
        dependencies detected, critical path, and critical path latency.

    Notes
    -----
    Upserts service records and dependency edges into the database, merges
    flat edges with inline ``depends_on`` entries, then delegates to the
    dependency agent tools for in-memory graph analysis (PageRank,
    betweenness centrality, critical path). Database writes use
    INSERT-or-UPDATE semantics via primary key matching.
    """
    import json as _json

    from sqlalchemy import select as _select

    from slo_engine.agents.dependency_agent.tools.tools import (
        analyse_dependency_graph,
        ingest_service_dependencies,
    )
    for svc_in in payload.services:
        result = await db.execute(_select(Service).where(Service.name == svc_in.service))
        existing = result.scalar_one_or_none()
        if existing:
            existing.display_name = svc_in.display_name or svc_in.service
            existing.tier = ServiceTier(svc_in.tier)
            existing.team = svc_in.team
            existing.metadata_ = svc_in.metadata
        else:
            svc = Service(
                name=svc_in.service,
                display_name=svc_in.display_name or svc_in.service,
                tier=ServiceTier(svc_in.tier),
                team=svc_in.team,
                metadata_=svc_in.metadata,
            )
            db.add(svc)
    await db.flush()

    depends_on_map: dict[str, list[dict]] = {}
    for edge in payload.edges:
        src = edge.source
        if src:
            depends_on_map.setdefault(src, []).append({
                "name": edge.target,
                "dep_type": edge.dep_type,
                "weight": edge.weight,
            })

    raw_payload = []
    for s in payload.services:
        d = s.model_dump()
        extra_deps = depends_on_map.get(s.service, [])
        existing_names = {dep["name"] for dep in d.get("depends_on", [])}
        for dep in extra_deps:
            if dep["name"] not in existing_names:
                d.setdefault("depends_on", []).append(dep)
        raw_payload.append(d)

    for edge in payload.edges:
        if edge.source and edge.target:
            from sqlalchemy import select as _sel2
            src_row = (await db.execute(_sel2(Service).where(Service.name == edge.source))).scalar_one_or_none()
            tgt_row = (await db.execute(_sel2(Service).where(Service.name == edge.target))).scalar_one_or_none()
            if src_row and tgt_row:
                existing_dep = (await db.execute(
                    _sel2(ServiceDependency).where(
                        ServiceDependency.source_id == src_row.id,
                        ServiceDependency.target_id == tgt_row.id,
                    )
                )).scalar_one_or_none()
                if not existing_dep:
                    db.add(ServiceDependency(
                        source_id=src_row.id,
                        target_id=tgt_row.id,
                        dep_type=DependencyType(edge.dep_type) if edge.dep_type in ("synchronous", "asynchronous", "external") else DependencyType.SYNCHRONOUS,
                        weight=edge.weight,
                    ))
    await db.flush()

    ingest_service_dependencies(_json.dumps(raw_payload))
    graph_result = _json.loads(analyse_dependency_graph("{}"))

    inline_edge_count = sum(len(s.depends_on) for s in payload.services)
    total_edges = len(payload.edges) + inline_edge_count

    return {
        "status": "accepted",
        "services_ingested": len(payload.services),
        "edges": total_edges,
        "circular_dependencies": graph_result.get("circular_deps", []),
        "critical_path": graph_result.get("critical_path", []),
        "critical_path_latency_ms": graph_result.get("critical_path_latency_ms", 0.0),
        "task": "graph_analysis_queued",
    }


@router.get("/services", summary="List all services")
async def list_services(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """
    Return a paginated list of all services in the database.

    Parameters
    ----------
    db : AsyncSession
        Injected async SQLAlchemy database session.

    Returns
    -------
    list of dict
        List of service dictionaries with id, name, display_name, tier, and
        pagerank_score fields.

    Notes
    -----
    Limited to 100 results. Pagination is not yet implemented.
    """
    from sqlalchemy import select
    result = await db.execute(select(Service).limit(100))
    services = result.scalars().all()
    return [
        {
            "id": str(s.id),
            "name": s.name,
            "display_name": s.display_name,
            "tier": s.tier.value,
            "pagerank_score": s.pagerank_score,
        }
        for s in services
    ]


@router.get("/services/graph", summary="Full dependency graph — services + edges in agent-ready format")
async def get_graph(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """
    Return all services with outgoing dependency edges in agent-ready format.

    Parameters
    ----------
    db : AsyncSession
        Injected async SQLAlchemy database session.

    Returns
    -------
    list of dict
        List of service dictionaries, each with ``service``, ``tier``, and
        ``depends_on`` list matching the format read by the dependency planner
        agent from ``state["graph_payload"]``.

    Notes
    -----
    Used by the Streamlit UI to dynamically populate ``graph_payload`` before
    pipeline runs. The format matches the ``SAMPLE_GRAPH`` services structure.
    Limited to 200 services.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(Service)
        .options(selectinload(Service.outgoing_deps).selectinload(ServiceDependency.target))
        .limit(200)
    )
    services = result.scalars().all()

    graph = []
    for svc in services:
        graph.append({
            "service": svc.name,
            "tier": svc.tier.value,
            "depends_on": [
                {
                    "name": dep.target.name,
                    "dep_type": dep.dep_type.value,
                    "weight": dep.weight,
                }
                for dep in svc.outgoing_deps
                if dep.target is not None
            ],
        })
    return graph


@router.get("/services/{service_name}", summary="Get service details")
async def get_service(service_name: str, db: AsyncSession = Depends(get_db)) -> dict:
    """
    Return detailed information for a single service.

    Parameters
    ----------
    service_name : str
        Unique name of the service to retrieve.
    db : AsyncSession
        Injected async SQLAlchemy database session.

    Returns
    -------
    dict
        Service detail dictionary including id, name, display_name, tier, team,
        pagerank_score, betweenness_centrality, fan_in, and fan_out.

    Notes
    -----
    Raises HTTP 404 if no service with the given name exists in the database.
    """
    from sqlalchemy import select
    result = await db.execute(select(Service).where(Service.name == service_name))
    svc = result.scalar_one_or_none()
    if not svc:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    return {
        "id": str(svc.id),
        "name": svc.name,
        "display_name": svc.display_name,
        "tier": svc.tier.value,
        "team": svc.team,
        "pagerank_score": svc.pagerank_score,
        "betweenness_centrality": svc.betweenness_centrality,
        "fan_in": svc.fan_in,
        "fan_out": svc.fan_out,
    }
