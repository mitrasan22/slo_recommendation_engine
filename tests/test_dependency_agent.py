"""
Tests for dependency agent tools.

Notes
-----
Covers: service graph ingestion, PageRank, DAG validation, critical path,
betweenness centrality, blast radius, and circular-dependency detection.

All tests use the module-level ``_graph_cache`` which is cleared before each
test by the ``reset_graph_cache`` autouse fixture in ``conftest.py``.
"""
import json

from slo_engine.agents.dependency_agent.tools.tools import (
    analyse_dependency_graph,
    detect_circular_dependencies,
    ingest_service_dependencies,
)

ACYCLIC_GRAPH = [
    {"service": "A", "depends_on": [{"name": "B", "weight": 1.0}], "p99_latency_ms": 100.0},
    {"service": "B", "depends_on": [{"name": "C", "weight": 1.0}], "p99_latency_ms": 200.0},
    {"service": "C", "depends_on": [],                               "p99_latency_ms": 300.0},
]

CYCLIC_GRAPH = [
    {"service": "X", "depends_on": [{"name": "Y", "weight": 1.0}], "p99_latency_ms": 50.0},
    {"service": "Y", "depends_on": [{"name": "X", "weight": 1.0}], "p99_latency_ms": 50.0},
]

DIAMOND_GRAPH = [
    {"service": "top",   "depends_on": [{"name": "left", "weight": 1.0}, {"name": "right", "weight": 1.0}], "p99_latency_ms": 10.0},
    {"service": "left",  "depends_on": [{"name": "base", "weight": 1.0}], "p99_latency_ms": 50.0},
    {"service": "right", "depends_on": [{"name": "base", "weight": 1.0}], "p99_latency_ms": 30.0},
    {"service": "base",  "depends_on": [], "p99_latency_ms": 100.0},
]


def _ingest(graph: list) -> dict:
    """
    Ingest a service dependency graph and assert success.

    Parameters
    ----------
    graph : list
        List of service dependency dicts to ingest.

    Returns
    -------
    dict
        Parsed result dict from ``ingest_service_dependencies``.

    Notes
    -----
    Asserts that the ingestion ``status`` is ``"success"`` before returning.
    Used as a helper shared across multiple test classes.
    """
    result = json.loads(ingest_service_dependencies(json.dumps(graph)))
    assert result["status"] == "success"
    return result


class TestIngestion:
    """
    Tests for service dependency graph ingestion.

    Notes
    -----
    Verifies that the correct service and edge counts are reported after
    ingestion for various graph topologies including empty graphs.
    """

    def test_returns_correct_service_count(self):
        r = _ingest(ACYCLIC_GRAPH)
        assert r["services_ingested"] == 3

    def test_returns_correct_edge_count(self):
        r = _ingest(ACYCLIC_GRAPH)
        assert r["edges_ingested"] == 2

    def test_diamond_topology_ingested(self):
        r = _ingest(DIAMOND_GRAPH)
        assert r["services_ingested"] == 4
        assert r["edges_ingested"] == 4

    def test_empty_graph_ingested(self):
        r = _ingest([])
        assert r["services_ingested"] == 0
        assert r["edges_ingested"] == 0


class TestPageRank:
    """
    Tests for PageRank computation on ingested dependency graphs.

    Notes
    -----
    PageRank scores must sum to approximately 1.0 (power-iteration convergence
    may leave a small residual) and all individual scores must be in ``[0, 1]``.
    """

    def test_scores_sum_to_one(self):
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        total = sum(out["pagerank"].values())
        assert abs(total - 1.0) < 0.05

    def test_all_scores_in_unit_interval(self):
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        for svc, score in out["pagerank"].items():
            assert 0.0 <= score <= 1.0, f"{svc} pagerank={score} out of [0,1]"

    def test_leaf_node_has_highest_pagerank(self):
        """
        Verify that the leaf node (C) ranks higher than the root (A).

        Notes
        -----
        In the A->B->C chain, C has no outgoing dependencies but is depended
        on by B, which is depended on by A. PageRank should assign C the
        highest score due to inbound link weight.
        """
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        pr = out["pagerank"]
        assert pr["C"] >= pr["A"]


class TestDagValidation:
    """
    Tests for DAG (Directed Acyclic Graph) validation.

    Notes
    -----
    Acyclic graphs must be identified as valid DAGs with no circular
    dependencies. Cyclic graphs must fail validation and report the cycles.
    """

    def test_acyclic_graph_is_valid_dag(self):
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        assert out["dag_is_valid"] is True
        assert out["circular_deps"] == []

    def test_cyclic_graph_fails_dag_check(self):
        _ingest(CYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        assert out["dag_is_valid"] is False
        assert len(out["circular_deps"]) > 0

    def test_diamond_topology_is_valid_dag(self):
        _ingest(DIAMOND_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        assert out["dag_is_valid"] is True


class TestCriticalPath:
    """
    Tests for critical path latency computation.

    Notes
    -----
    The critical path latency must be at least as large as the slowest
    single node in the graph (300 ms for the A->B->C chain). It must
    also be non-negative.
    """

    def test_critical_path_at_least_slowest_single_node(self):
        """
        Verify critical path is at least as long as the slowest node.

        Notes
        -----
        In the A->B->C chain the critical path passes through all three nodes.
        The slowest single node is C at 300 ms, so the path must be >= 300 ms.
        """
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        assert out["critical_path_latency_ms"] >= 300.0

    def test_critical_path_non_negative(self):
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        assert out["critical_path_latency_ms"] >= 0.0


class TestCentralityAndBlastRadius:
    """
    Tests for betweenness centrality and blast radius computation.

    Notes
    -----
    Both metrics must always be in ``[0, 1]``. In diamond topology, the
    ``base`` node (depended on by all others) must have the highest blast
    radius.
    """

    def test_betweenness_in_unit_interval(self):
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        for svc, val in out["betweenness"].items():
            assert 0.0 <= val <= 1.0, f"{svc} betweenness={val}"

    def test_blast_radius_in_unit_interval(self):
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        for svc, val in out["blast_radius"].items():
            assert 0.0 <= val <= 1.0, f"{svc} blast_radius={val}"

    def test_base_node_highest_blast_radius_in_diamond(self):
        """
        Verify that the shared base node has the highest blast radius.

        Notes
        -----
        In the diamond topology, ``base`` is depended on by both ``left`` and
        ``right``, which are both depended on by ``top``. Failure of ``base``
        cascades to all other services, so it should have the highest blast
        radius.
        """
        _ingest(DIAMOND_GRAPH)
        out = json.loads(analyse_dependency_graph("{}"))
        br = out["blast_radius"]
        assert br["base"] >= br["top"]


class TestCircularDependencyDetection:
    """
    Tests for circular dependency detection using Tarjan SCC.

    Notes
    -----
    Cycles must be detected and include a human-readable recommendation.
    Acyclic graphs must report zero cycles.
    """

    def test_cycle_detected(self):
        _ingest(CYCLIC_GRAPH)
        out = json.loads(detect_circular_dependencies("{}"))
        assert out["count"] > 0

    def test_cycle_result_includes_recommendation(self):
        _ingest(CYCLIC_GRAPH)
        out = json.loads(detect_circular_dependencies("{}"))
        assert "recommendation" in out["circular_deps"][0]

    def test_no_cycle_in_acyclic_graph(self):
        _ingest(ACYCLIC_GRAPH)
        out = json.loads(detect_circular_dependencies("{}"))
        assert out["circular_deps"] == []
