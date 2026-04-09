"""
Pydantic schemas for recommendation agent tools.

Notes
-----
Defines input and output validation schemas used by the SLO generation,
feasibility checking, and MILP portfolio optimization tools.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SLORecommendationInput(BaseModel):
    """
    Input schema for the SLO recommendation generation tool.

    Attributes
    ----------
    service_name : str
        Name of the target service.
    metrics_summary : dict
        Bayesian query results including posterior mean, burn rates, and drift.
    graph_analysis : dict
        Dependency analysis results including PageRank and blast radius.
    current_slo : dict or None
        Currently active SLO configuration, or None for new services.
    dep_slos : dict
        Mapping of dependency service names to their current SLO dicts.
    knowledge_context : str
        Retrieved runbook or incident context from the RAG knowledge store.
    knowledge_sources : list of str
        Knowledge source IDs retrieved from ChromaDB.

    Notes
    -----
    Serialised to JSON and passed to ``generate_slo_recommendation``. The
    ``dep_slos`` field is used to propagate dependency constraints into the
    series reliability formula.
    """

    service_name: str
    metrics_summary: dict[str, Any]
    graph_analysis: dict[str, Any]
    current_slo: dict[str, Any] | None = None
    dep_slos: dict[str, dict[str, Any]] = Field(default_factory=dict)
    knowledge_context: str = ""
    knowledge_sources: list[str] = Field(default_factory=list)


class SLORecommendationOutput(BaseModel):
    """
    Output schema for the SLO recommendation generation tool.

    Attributes
    ----------
    service_name : str
        Name of the service for which the SLO was generated.
    recommended_availability : float
        Recommended availability SLO value between 0.0 and 1.0.
    recommended_latency_p99_ms : float
        Recommended p99 latency SLO in milliseconds (must be > 0).
    recommended_error_rate : float
        Complement of the availability SLO (1 - availability).
    confidence_score : float
        Confidence in the recommendation between 0.0 and 1.0.
    reasoning : str
        Engineer-facing explanation of how the SLO was derived.
    math_details : dict
        Raw mathematical output including series/parallel reliability,
        Monte Carlo estimates, CLT latency bounds, and Wilson interval.
    upstream_impact : dict
        Estimated impact of this SLO on upstream services.
    data_sources : list of str
        Identifiers of data sources used in the recommendation.
    requires_human_review : bool
        Whether engineer review is required. Defaults to False.
    review_reason : str
        Explanation of why human review is required, or empty string.

    Notes
    -----
    Returned as JSON by ``generate_slo_recommendation``. The ``math_details``
    dict contains keys for each mathematical computation performed.
    """

    service_name: str
    recommended_availability: float = Field(..., ge=0.0, le=1.0)
    recommended_latency_p99_ms: float = Field(..., gt=0)
    recommended_error_rate: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    math_details: dict[str, Any] = Field(default_factory=dict)
    upstream_impact: dict[str, Any] = Field(default_factory=dict)
    data_sources: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    review_reason: str = ""


class FeasibilityCheckInput(BaseModel):
    """
    Input schema for the SLO feasibility check tool.

    Attributes
    ----------
    service_name : str
        Name of the service whose proposed SLO is being checked.
    proposed_availability : float
        Proposed availability SLO value to evaluate.
    proposed_latency_p99_ms : float
        Proposed p99 latency SLO in milliseconds.
    historical_availability : float
        Historically observed availability used as the upper bound.
    dep_availabilities : dict
        Mapping of dependency service names to their availability SLOs,
        used to compute the series reliability ceiling.

    Notes
    -----
    Passed to ``check_slo_feasibility``. The availability ceiling is computed
    as the product of all dependency availabilities using the series reliability
    model: A_series = product(A_i for all i).
    """

    service_name: str
    proposed_availability: float
    proposed_latency_p99_ms: float
    historical_availability: float
    dep_availabilities: dict[str, float] = Field(default_factory=dict)


class FeasibilityCheckOutput(BaseModel):
    """
    Output schema for the SLO feasibility check tool.

    Attributes
    ----------
    is_feasible : bool
        Whether the proposed SLO is achievable given constraints.
    feasibility_score : float
        Numeric feasibility score between 0 and 1.
    availability_ceiling : float
        Maximum achievable availability given dependency constraints.
    latency_floor_ms : float
        Minimum achievable latency given critical path constraints.
    issues : list of str
        Human-readable descriptions of feasibility violations.
    adjusted_recommendation : dict or None
        Adjusted SLO values when the proposed SLO is not feasible,
        or None when feasible.

    Notes
    -----
    Returned as JSON by ``check_slo_feasibility``. When ``is_feasible=False``,
    the orchestrator uses ``adjusted_recommendation`` to override the
    generation result with a feasible alternative.
    """

    is_feasible: bool
    feasibility_score: float
    availability_ceiling: float
    latency_floor_ms: float
    issues: list[str]
    adjusted_recommendation: dict[str, float] | None
