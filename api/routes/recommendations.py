"""
Recommendations API routes for the SLO Recommendation Engine.

Notes
-----
Provides SLO-specific REST wrappers around the tool layer that seed the agent
state and return structured results. The ADK agent pipeline with LLM reasoning
is served at /run and /run_sse by get_fast_api_app. These routes provide direct
tool-layer access without the LLM agent loop.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from slo_engine.agents.dependency_agent.tools.tools import (
    ingest_service_dependencies,
    analyse_dependency_graph,
)
from slo_engine.agents.metrics_agent.tools.tools import query_service_metrics
from slo_engine.agents.recommendation_agent.tools.tools import (
    generate_slo_recommendation,
    check_slo_feasibility,
    run_milp_optimization,
    retrieve_knowledge_for_slo,
)

router = APIRouter()


class RecommendationRequestBody(BaseModel):
    """
    Request body for bulk SLO recommendation generation.

    Attributes
    ----------
    services : list of dict
        Service dependency graph payload, each item describing one service
        with its dependencies, tier, and latency characteristics.
    focus_service : str or None
        Optional service name to compute recommendations for only that service.
        When ``None``, all services in the graph are processed.
    window_days : int
        Observation window in days for historical metrics. Must be between 7
        and 90 inclusive.

    Notes
    -----
    The ``services`` list format matches the output of the dependency ingestion
    tools and the ``SAMPLE_GRAPH`` in the Streamlit UI.
    """

    services: list[dict[str, Any]] = Field(..., description="Service dependency graph payload")
    focus_service: str | None = None
    window_days: int = Field(default=30, ge=7, le=90)


class ImpactAnalysisBody(BaseModel):
    """
    Request body for SLO change impact analysis.

    Attributes
    ----------
    service_name : str
        Name of the service for which the proposed SLO change is being evaluated.
    proposed_availability : float
        Target availability SLO value, between 0.0 and 1.0 inclusive.
    proposed_latency_p99_ms : float
        Target p99 latency in milliseconds, must be strictly positive.

    Notes
    -----
    Impact analysis checks both the feasibility of the proposed change for the
    target service and cascading effects on all upstream services that depend
    on it.
    """

    service_name: str
    proposed_availability: float = Field(..., ge=0.0, le=1.0)
    proposed_latency_p99_ms: float = Field(..., gt=0)


@router.get(
    "/services/{service_id}/slo-recommendations",
    summary="Get SLO recommendations for a service",
)
async def get_slo_recommendations(service_id: str) -> dict:
    """
    Run a single-service SLO recommendation using direct tool-layer calls.

    Parameters
    ----------
    service_id : str
        Identifier of the service to generate recommendations for.

    Returns
    -------
    dict
        Dictionary with keys ``service``, ``recommendation``, and
        ``metrics_summary``.

    Notes
    -----
    Queries the metrics tool, retrieves relevant knowledge context, then
    generates an SLO recommendation without invoking the LLM agent loop.
    For full multi-agent pipeline with LLM reasoning, use the ADK /run endpoint.
    """
    metrics_raw = query_service_metrics(json.dumps({
        "service_name": service_id,
        "window_days": 30,
    }))
    metrics = json.loads(metrics_raw)
    if metrics.get("status") == "error":
        raise HTTPException(status_code=500, detail=metrics["message"])

    knowledge_raw = retrieve_knowledge_for_slo(json.dumps({
        "service_name": service_id,
        "service_type": "generic",
        "tier": "medium",
        "drift_detected": metrics.get("drift_detected", False),
        "anomaly_severity": metrics.get("anomaly_severity", "none"),
        "has_external_deps": False,
        "top_k": 3,
    }))
    knowledge = json.loads(knowledge_raw)

    rec_raw = generate_slo_recommendation(json.dumps({
        "service_name": service_id,
        "metrics_summary": metrics,
        "graph_analysis": {},
        "dep_slos": {},
        "knowledge_context": knowledge.get("context_summary", ""),
        "knowledge_sources": knowledge.get("source_ids", []),
    }))
    rec = json.loads(rec_raw)
    if rec.get("status") == "error":
        raise HTTPException(status_code=500, detail=rec["message"])

    return {
        "service": service_id,
        "recommendation": rec,
        "metrics_summary": metrics,
    }


@router.post(
    "/recommendations/bulk",
    summary="Generate SLO recommendations for a full service graph",
)
async def bulk_recommendations(body: RecommendationRequestBody) -> dict:
    """
    Run the full tool-layer SLO pipeline for an entire service graph.

    Parameters
    ----------
    body : RecommendationRequestBody
        Request body with service graph, optional focus service, and window.

    Returns
    -------
    dict
        Dictionary with ``services_analysed``, ``recommendations`` per service,
        ``portfolio_optimization`` from MILP, and ``graph_summary``.

    Notes
    -----
    Executes the pipeline inline without the LLM agent loop: ingests the
    dependency graph, queries metrics for each service, retrieves knowledge
    context, generates per-service recommendations, and runs MILP portfolio
    optimisation. For streaming and LLM reasoning, use the ADK /run_sse endpoint.
    """
    ingest_service_dependencies(json.dumps(body.services))
    graph_raw = analyse_dependency_graph("{}")
    graph = json.loads(graph_raw)

    service_names = [s.get("service", "") for s in body.services]
    recommendations: dict[str, Any] = {}
    all_hist: dict[str, float] = {}
    all_weights: dict[str, float] = {}
    pagerank = graph.get("pagerank", {})

    for svc_data in body.services:
        svc = svc_data.get("service", "")
        if not svc:
            continue
        metrics_raw = query_service_metrics(json.dumps({
            "service_name": svc,
            "window_days": body.window_days,
        }))
        metrics = json.loads(metrics_raw)
        has_external_deps = any(
            d.get("dep_type") == "external"
            for d in svc_data.get("depends_on", [])
        )
        knowledge_raw = retrieve_knowledge_for_slo(json.dumps({
            "service_name": svc,
            "service_type": "generic",
            "tier": "medium",
            "drift_detected": metrics.get("drift_detected", False),
            "anomaly_severity": metrics.get("anomaly_severity", "none"),
            "has_external_deps": has_external_deps,
            "top_k": 3,
        }))
        knowledge = json.loads(knowledge_raw)
        rec_raw = generate_slo_recommendation(json.dumps({
            "service_name": svc,
            "metrics_summary": metrics,
            "graph_analysis": graph,
            "dep_slos": {},
            "knowledge_context": knowledge.get("context_summary", ""),
            "knowledge_sources": knowledge.get("source_ids", []),
        }))
        rec = json.loads(rec_raw)
        recommendations[svc] = rec
        all_hist[svc] = metrics.get("posterior_mean", 0.99)
        all_weights[svc] = pagerank.get(svc, 1.0)

    sync_deps: dict[str, list[str]] = {}
    for svc_data in body.services:
        svc = svc_data.get("service", "")
        sync_deps[svc] = [
            d["name"] for d in svc_data.get("depends_on", [])
            if d.get("dep_type", "synchronous") == "synchronous"
        ]

    portfolio_raw = run_milp_optimization(json.dumps({
        "services": service_names,
        "historical_availability": all_hist,
        "importance_weights": all_weights,
        "sync_deps": sync_deps,
        "error_budget": 0.001,
    }))
    portfolio = json.loads(portfolio_raw)

    return {
        "services_analysed": service_names,
        "recommendations": recommendations,
        "portfolio_optimization": portfolio,
        "graph_summary": {
            "dag_is_valid": graph.get("dag_is_valid", True),
            "critical_path": graph.get("critical_path", []),
            "critical_path_latency_ms": graph.get("critical_path_latency_ms", 0),
            "circular_deps": graph.get("circular_deps", []),
        },
    }


@router.post("/slos/impact-analysis", summary="Check dependency impact of a proposed SLO change")
async def impact_analysis(body: ImpactAnalysisBody) -> dict:
    """
    Evaluate the feasibility and cascade impact of a proposed SLO change.

    Parameters
    ----------
    body : ImpactAnalysisBody
        Request body with the target service name and proposed SLO values.

    Returns
    -------
    dict
        Dictionary containing feasibility assessment, historical availability,
        sync dependency list, and cascade impact on upstream services.

    Notes
    -----
    Performs two analyses: (1) checks the feasibility of the proposed SLO for
    the target service against its own historical performance and its dependency
    capabilities; (2) computes the cascade ceiling change for each upstream
    service that depends on the target service using the log-product availability
    model.
    """
    import json as _json
    from slo_engine.agents.dependency_agent.tools.tools import _graph_cache
    from slo_engine.agents.metrics_agent.tools.tools import query_service_metrics
    from slo_engine.agents.recommendation_agent.tools.tools import check_slo_feasibility

    metrics_raw = query_service_metrics(_json.dumps({
        "service_name": body.service_name,
        "window_days": 30,
    }))
    metrics = _json.loads(metrics_raw)
    historical_avail = metrics.get("posterior_mean", 0.99)

    graph = _graph_cache.get("graph", {})
    edges = _graph_cache.get("edges", [])
    sync_dep_names = [
        t for (s, t, _) in edges
        if s == body.service_name
    ]

    dep_availabilities: dict[str, float] = {}
    for dep in sync_dep_names:
        dep_metrics_raw = query_service_metrics(_json.dumps({"service_name": dep, "window_days": 30}))
        dep_metrics = _json.loads(dep_metrics_raw)
        dep_availabilities[dep] = dep_metrics.get("posterior_mean", 0.99)

    feas_raw = check_slo_feasibility(_json.dumps({
        "service_name": body.service_name,
        "proposed_availability": body.proposed_availability,
        "proposed_latency_p99_ms": body.proposed_latency_p99_ms,
        "historical_availability": historical_avail,
        "dep_availabilities": dep_availabilities,
    }))
    feas = _json.loads(feas_raw)

    upstream_services = [
        s for (s, t, _) in edges
        if t == body.service_name
    ]

    cascade: dict[str, dict] = {}
    for upstream in upstream_services:
        up_metrics_raw = query_service_metrics(_json.dumps({"service_name": upstream, "window_days": 30}))
        up_metrics = _json.loads(up_metrics_raw)
        up_hist = up_metrics.get("posterior_mean", 0.99)

        import math
        new_ceiling = math.exp(math.log(max(body.proposed_availability, 1e-9)) + math.log(max(up_hist, 1e-9)))
        new_ceiling = min(new_ceiling, 0.9999)
        cascade[upstream] = {
            "current_availability": up_hist,
            "new_availability_ceiling": round(new_ceiling, 6),
            "impact": "tightened" if new_ceiling < up_hist else "relaxed",
        }

    return {
        "service_name": body.service_name,
        "proposed_availability": body.proposed_availability,
        "proposed_latency_p99_ms": body.proposed_latency_p99_ms,
        "feasible": feas.get("is_feasible", True),
        "feasibility_score": feas.get("feasibility_score", 1.0),
        "availability_ceiling": feas.get("availability_ceiling"),
        "blocking_reason": feas.get("issues", []),
        "historical_availability": historical_avail,
        "sync_dependencies": sync_dep_names,
        "dep_availabilities": dep_availabilities,
        "upstream_affected": upstream_services,
        "cascade": cascade,
    }
