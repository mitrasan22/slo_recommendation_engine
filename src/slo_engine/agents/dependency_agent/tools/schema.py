"""
Pydantic schemas for dependency agent tools.

Notes
-----
Defines the input and output validation schemas used by the dependency graph
ingestion, analysis, and impact computation tools.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DependencyType(str, Enum):
    """
    Enumeration of service dependency relationship types.

    Attributes
    ----------
    SYNCHRONOUS : str
        Synchronous blocking call; failure propagates directly.
    ASYNCHRONOUS : str
        Asynchronous call; failure is buffered or retried.
    EXTERNAL : str
        External third-party dependency with its own SLA.
    DATASTORE : str
        Database or cache dependency.

    Notes
    -----
    Used in ``DependencyEdge`` and throughout the graph analysis pipeline
    to classify edges for availability formula selection.
    """

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    EXTERNAL = "external"
    DATASTORE = "datastore"


class DependencyEdge(BaseModel):
    """
    A single directed dependency edge from one service to another.

    Attributes
    ----------
    name : str
        Target service name.
    dep_type : DependencyType
        Type of dependency relationship. Defaults to ``SYNCHRONOUS``.
    weight : float
        Edge weight between 0.0 and 1.0 representing dependency strength.

    Notes
    -----
    Used within service node dictionaries in the ``depends_on`` list field.
    """

    name: str = Field(..., description="Target service name")
    dep_type: DependencyType = DependencyType.SYNCHRONOUS
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class IngestDependenciesInput(BaseModel):
    """
    Input schema for the service dependency ingestion tool.

    Attributes
    ----------
    services : list of dict
        List of service dependency entries. Each entry must have a ``service``
        key, an optional ``depends_on`` list of dependency edges, an optional
        ``p99_latency_ms`` float, and an optional ``tier`` string.

    Notes
    -----
    The services list is serialised to JSON before being passed to the
    ``ingest_service_dependencies`` tool function.
    """

    services: list[dict[str, Any]] = Field(
        ...,
        description="List of service dependency entries",
        examples=[[{
            "service": "api-gateway",
            "depends_on": [{"name": "auth-service", "dep_type": "synchronous", "weight": 1.0}],
            "p99_latency_ms": 250.0,
            "tier": "critical",
        }]],
    )


class GraphAnalysisInput(BaseModel):
    """
    Input schema for the dependency graph analysis tool.

    Attributes
    ----------
    service_name : str or None
        If provided, focuses the analysis on this specific service.
    include_transitive : bool
        Whether to include transitive dependencies in the analysis.
    latency_map : dict or None
        Optional override mapping of service name to p99 latency in milliseconds.

    Notes
    -----
    When ``service_name`` is None, all services in the cached graph are analysed.
    """

    service_name: str | None = Field(None, description="Focus analysis on specific service")
    include_transitive: bool = Field(True, description="Include transitive dependencies")
    latency_map: dict[str, float] | None = None


class GraphAnalysisOutput(BaseModel):
    """
    Output schema for the dependency graph analysis tool.

    Attributes
    ----------
    pagerank : dict
        Mapping of service name to PageRank score.
    betweenness : dict
        Mapping of service name to betweenness centrality score.
    fan_in : dict
        Mapping of service name to number of incoming dependency edges.
    fan_out : dict
        Mapping of service name to number of outgoing dependency edges.
    circular_deps : list of list of str
        Each inner list is a cycle of service names.
    critical_path : list of str
        Ordered service names on the critical latency path.
    critical_path_latency_ms : float
        Total latency of the critical path in milliseconds.
    blast_radius : dict
        Mapping of service name to blast radius score (fraction of the graph
        that would be affected by this service failing).
    topological_order : list of str
        Services in topological sort order (empty if graph has cycles).
    dag_is_valid : bool
        Whether the graph is a valid directed acyclic graph.

    Notes
    -----
    Returned as JSON by the ``analyse_dependency_graph`` tool function.
    """

    pagerank: dict[str, float]
    betweenness: dict[str, float]
    fan_in: dict[str, int]
    fan_out: dict[str, int]
    circular_deps: list[list[str]]
    critical_path: list[str]
    critical_path_latency_ms: float
    blast_radius: dict[str, float]
    topological_order: list[str]
    dag_is_valid: bool


class ImpactAnalysisInput(BaseModel):
    """
    Input schema for the dependency impact analysis tool.

    Attributes
    ----------
    service_name : str
        Name of the service whose proposed SLO change is being evaluated.
    proposed_slo_availability : float
        Target availability SLO value between 0.0 and 1.0.
    proposed_slo_latency_p99_ms : float or None
        Target p99 latency in milliseconds, or None to skip latency analysis.

    Notes
    -----
    Used to compute the cascade impact on upstream services that depend on
    the target service.
    """

    service_name: str
    proposed_slo_availability: float = Field(..., ge=0.0, le=1.0)
    proposed_slo_latency_p99_ms: float | None = None


class ImpactAnalysisOutput(BaseModel):
    """
    Output schema for the dependency impact analysis tool.

    Attributes
    ----------
    service_name : str
        Name of the analysed service.
    upstream_services : list of str
        Services that depend on this service (would be affected by changes).
    downstream_services : list of str
        Services this service depends on.
    availability_propagation : dict
        Mapping of upstream service name to new availability ceiling under the
        proposed SLO change.
    latency_propagation : dict
        Mapping of upstream service name to new latency impact.
    blast_radius_score : float
        Fraction of the graph affected if this service fails.
    critical_path_impact : bool
        Whether this service is on the critical latency path.
    recommendation : str
        Engineer-facing recommendation about the proposed change.

    Notes
    -----
    Returned as JSON by the ``compute_dependency_impact`` tool function.
    """

    service_name: str
    upstream_services: list[str]
    downstream_services: list[str]
    availability_propagation: dict[str, float]
    latency_propagation: dict[str, float]
    blast_radius_score: float
    critical_path_impact: bool
    recommendation: str
