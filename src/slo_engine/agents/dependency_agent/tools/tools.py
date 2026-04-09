"""
Dependency agent tools implementing service graph mathematics.

Notes
-----
Graph math is implemented directly in these functions, which are registered
with Google ADK as FunctionTool instances. No intermediate math_engine layer.

Algorithms used:
- PageRank power iteration: PR(u) = (1-d)/N + d * sum(PR(v)/L(v) for v->u)
- Tarjan SCC via NetworkX (O(V+E)) for circular dependency detection
- DAG longest-path dynamic programming for critical latency path computation
- Brandes betweenness centrality: BC(v) = sum(sigma(s,t|v)/sigma(s,t)) for blast radius
"""
from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np

from slo_engine.agents.dependency_agent.tools.schema import (
    GraphAnalysisInput,
    GraphAnalysisOutput,
    ImpactAnalysisInput,
    ImpactAnalysisOutput,
)
from slo_engine.config.settings import settings

_graph_cache: dict[str, Any] = {}

_DAMPING = float(settings.compute_logic.pagerank_damping)
_PR_TOL  = float(settings.compute_logic.pagerank_tol)
_PR_ITER = int(settings.compute_logic.pagerank_max_iter)


def ingest_service_dependencies(
    services_json: str = "",
    services: list[Any] | None = None,
) -> str:
    """
    Parse and store a service dependency graph from a JSON payload.

    Parameters
    ----------
    services_json : str, optional
        JSON-encoded list of service dictionaries, each with ``service``,
        ``depends_on``, ``p99_latency_ms``, and ``tier`` fields.
    services : list, optional
        Pre-parsed list of service dictionaries. Used when ``services_json``
        is empty.

    Returns
    -------
    str
        JSON-encoded summary with ``status``, ``services_ingested``,
        ``edges_ingested``, and ``nodes`` fields.

    Notes
    -----
    Builds a NetworkX DiGraph from the payload and stores it in the module-level
    ``_graph_cache`` dictionary alongside the edge list and latency map. Both
    ``services_json`` and direct ``services`` list are accepted to support
    both ADK tool invocation and direct Python calls. On error, returns a
    JSON object with ``status="error"`` and a ``message`` field.
    """
    try:
        if services_json:
            payload = json.loads(services_json)
        elif services is not None:
            payload = services
        else:
            payload = []
        if isinstance(payload, dict):
            payload = payload.get("services", [])

        edges: list[tuple[str, str, float]] = []
        latency_map: dict[str, float] = {}

        for item in payload:
            svc = item["service"]
            latency_map[svc] = float(item.get("p99_latency_ms", 0.0))
            for dep in item.get("depends_on", []):
                edges.append((svc, dep["name"], float(dep.get("weight", 1.0))))

        G = nx.DiGraph()
        for src, tgt, w in edges:
            G.add_edge(src, tgt, weight=w)

        _graph_cache["edges"] = edges
        _graph_cache["latency_map"] = latency_map
        _graph_cache["graph"] = G

        return json.dumps({
            "status": "success",
            "services_ingested": len(latency_map),
            "edges_ingested": len(edges),
            "nodes": list(latency_map.keys()),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def analyse_dependency_graph(
    analysis_input_json: str = "{}",
    service_name: str | None = None,
    include_transitive: bool = True,
    latency_map: dict[str, float] | None = None,
) -> str:
    """
    Run PageRank, Tarjan SCC, DAG longest-path, and betweenness centrality.

    Parameters
    ----------
    analysis_input_json : str, optional
        JSON-encoded ``GraphAnalysisInput`` object. Pass ``"{}"`` to use
        defaults or supply individual keyword arguments instead.
    service_name : str, optional
        If provided, focuses the analysis on this specific service.
    include_transitive : bool, optional
        Whether to include transitive dependencies. Defaults to True.
    latency_map : dict, optional
        Optional override mapping of service name to p99 latency in milliseconds.

    Returns
    -------
    str
        JSON-encoded ``GraphAnalysisOutput`` with pagerank, betweenness,
        blast_radius, critical_path, critical_path_latency_ms, fan_in,
        fan_out, circular_deps, topological_order, and dag_is_valid fields.

    Notes
    -----
    PageRank uses power iteration: PR(u) = (1-d)/N + d * sum(PR(v)/L(v)).
    Convergence tolerance and max iterations are read from settings.
    Betweenness centrality uses the Brandes algorithm via NetworkX.
    Critical path uses DP on the topological sort; falls back to empty list
    if the graph contains cycles. Blast radius is betweenness normalised by
    the maximum betweenness value. On error, returns JSON with ``status="error"``.
    """
    try:
        if analysis_input_json and analysis_input_json != "{}":
            inp = GraphAnalysisInput.model_validate_json(analysis_input_json)
        else:
            inp = GraphAnalysisInput(
                service_name=service_name,
                include_transitive=include_transitive,
                latency_map=latency_map,
            )
        G: nx.DiGraph = _graph_cache.get("graph", nx.DiGraph())
        merged_latency_map: dict[str, float] = _graph_cache.get("latency_map", {})
        if inp.latency_map:
            merged_latency_map = {**merged_latency_map, **inp.latency_map}

        if G.number_of_nodes() == 0:
            return json.dumps({"status": "error", "message": "No graph ingested yet."})

        nodes = list(G.nodes())
        N = len(nodes)
        node_idx = {n: i for i, n in enumerate(nodes)}

        pr = np.full(N, 1.0 / N)
        out_deg = np.array([max(G.out_degree(n), 1) for n in nodes], dtype=float)
        dangling = np.array([G.out_degree(n) == 0 for n in nodes])

        for _ in range(_PR_ITER):
            pr_new = np.zeros(N)
            pr_new += _DAMPING * pr[dangling].sum() / N
            for u, v in G.edges():
                pr_new[node_idx[v]] += _DAMPING * pr[node_idx[u]] / out_deg[node_idx[u]]
            pr_new += (1.0 - _DAMPING) / N
            if np.abs(pr_new - pr).sum() < _PR_TOL:
                pr = pr_new
                break
            pr = pr_new
        pagerank = {nodes[i]: float(pr[i]) for i in range(N)}

        betweenness = nx.betweenness_centrality(G, normalized=True, weight="weight")

        sccs = list(nx.strongly_connected_components(G))
        circular_deps = [sorted(scc) for scc in sccs if len(scc) > 1]
        dag_valid = len(circular_deps) == 0

        critical_path: list[str] = []
        critical_latency = 0.0
        if dag_valid and merged_latency_map:
            try:
                topo = list(nx.topological_sort(G))
                dist = {n: merged_latency_map.get(n, 0.0) for n in nodes}
                pred: dict[str, str | None] = {n: None for n in nodes}
                for u in topo:
                    for v in G.successors(u):
                        candidate = dist[u] + merged_latency_map.get(v, 0.0)
                        if candidate > dist[v]:
                            dist[v] = candidate
                            pred[v] = u
                end = max(dist, key=dist.__getitem__)
                critical_latency = dist[end]
                cur: str | None = end
                while cur is not None:
                    critical_path.append(cur)
                    cur = pred[cur]
                critical_path.reverse()
            except nx.NetworkXUnfeasible:
                pass

        max_bc = max(betweenness.values(), default=1.0) or 1.0
        blast_radius = {k: v / max_bc for k, v in betweenness.items()}

        fan_in  = dict(G.in_degree())
        fan_out = dict(G.out_degree())

        topo_order = list(nx.topological_sort(G)) if dag_valid else nodes

        output = GraphAnalysisOutput(
            pagerank=pagerank,
            betweenness=betweenness,
            fan_in=fan_in,
            fan_out=fan_out,
            circular_deps=circular_deps,
            critical_path=critical_path,
            critical_path_latency_ms=critical_latency,
            blast_radius=blast_radius,
            topological_order=topo_order,
            dag_is_valid=dag_valid,
        )
        return output.model_dump_json()
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def compute_dependency_impact(impact_input_json: str) -> str:
    """
    Compute the cascading impact of a proposed SLO change on the dependency graph.

    Parameters
    ----------
    impact_input_json : str
        JSON-encoded ``ImpactAnalysisInput`` with ``service_name``,
        ``proposed_slo_availability``, and optional ``proposed_slo_latency_p99_ms``.

    Returns
    -------
    str
        JSON-encoded ``ImpactAnalysisOutput`` with upstream and downstream
        service lists, availability and latency propagation dictionaries,
        blast radius score, critical path flag, and recommendation string.

    Notes
    -----
    Availability propagation uses the series reliability model: each upstream
    service's effective availability ceiling is constrained by the proposed SLO
    weighted by that upstream's betweenness centrality. On error, returns JSON
    with ``status="error"`` and a ``message`` field.
    """
    try:
        inp = ImpactAnalysisInput.model_validate_json(impact_input_json)
        G: nx.DiGraph = _graph_cache.get("graph", nx.DiGraph())
        svc = inp.service_name

        if svc not in G:
            return json.dumps({"status": "error", "message": f"Service '{svc}' not in graph."})

        upstream   = list(G.predecessors(svc))
        downstream = list(G.successors(svc))

        betweenness = nx.betweenness_centrality(G, normalized=True)
        avail_propagation = {
            up: round(betweenness.get(up, 0.5) * inp.proposed_slo_availability, 5)
            for up in upstream
        }

        latency_propagation: dict[str, float] = {}
        if inp.proposed_slo_latency_p99_ms:
            lmap = _graph_cache.get("latency_map", {})
            for up in upstream:
                latency_propagation[up] = round(
                    lmap.get(up, 0.0) + inp.proposed_slo_latency_p99_ms, 1
                )

        critical_path = _graph_cache.get("critical_path", [])
        on_critical = svc in critical_path
        blast = betweenness.get(svc, 0.0)

        rec = (
            f"'{svc}' blast-radius={blast:.3f}. "
            + ("ON critical latency path. " if on_critical else "")
            + (f"Upstream constrained: {avail_propagation}" if upstream else "No upstream callers.")
        )

        output = ImpactAnalysisOutput(
            service_name=svc,
            upstream_services=upstream,
            downstream_services=downstream,
            availability_propagation=avail_propagation,
            latency_propagation=latency_propagation,
            blast_radius_score=round(blast, 4),
            critical_path_impact=on_critical,
            recommendation=rec,
        )
        return output.model_dump_json()
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def detect_circular_dependencies(dummy: str = "{}") -> str:
    """
    Detect and describe all circular dependency cycles in the cached graph.

    Parameters
    ----------
    dummy : str, optional
        Unused parameter required by ADK FunctionTool signature conventions.
        Defaults to ``"{}"``.

    Returns
    -------
    str
        JSON-encoded object with a ``circular_deps`` list of cycle descriptions
        and a ``count`` field. Returns ``{"circular_deps": [], "status": "clean"}``
        when no cycles are found.

    Notes
    -----
    Uses ``networkx.simple_cycles`` which enumerates all elementary circuits.
    Results are capped at 10 cycles to avoid excessively large payloads. Each
    cycle description includes the cycle node list, length, and a recommendation
    to break the cycle via async messaging. Series reliability is invalid for
    any service that participates in a cycle. On error, returns JSON with
    ``status="error"`` and a ``message`` field.
    """
    try:
        G: nx.DiGraph = _graph_cache.get("graph", nx.DiGraph())
        cycles = list(nx.simple_cycles(G))
        if not cycles:
            return json.dumps({"circular_deps": [], "status": "clean"})

        descriptions = [
            {
                "cycle": cycle,
                "length": len(cycle),
                "recommendation": (
                    f"Break cycle {' -> '.join(cycle)} via async messaging "
                    f"between '{cycle[-1]}' and '{cycle[0]}'."
                ),
            }
            for cycle in cycles[:10]
        ]
        return json.dumps({"circular_deps": descriptions, "count": len(cycles)})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
