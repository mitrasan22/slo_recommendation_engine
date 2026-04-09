"""
Tests for the RAG knowledge store (ChromaDB backend).

Notes
-----
Uses the module-level singleton which runs against an EphemeralClient
(in-memory) when ``PYTEST_CURRENT_TEST`` is set — no disk I/O, no cleanup
needed.

Covers:

  - Document loading (JSON KB + Markdown docs)
  - ChromaDB collection integrity
  - MMR retrieval correctness
  - Adaptive lambda selection
  - Document type filtering (pushed to ChromaDB where clause)
  - Service-context retrieval
"""
from __future__ import annotations

import pytest

from slo_engine.rag.knowledge_store import (
    KnowledgeStore,
    _infer_mmr_lambda,
    knowledge_store,
)


class TestKnowledgeBaseLoading:
    """
    Tests for document loading from JSON and Markdown sources.

    Notes
    -----
    Verifies that the module-level singleton loaded documents at startup
    and that the internal ``_docs`` list is consistent with the ChromaDB
    collection count.
    """

    def test_singleton_has_documents(self):
        assert knowledge_store.document_count > 0

    def test_document_count_at_least_200(self):
        assert knowledge_store.document_count >= 200

    def test_internal_docs_list_matches_collection(self):
        assert len(knowledge_store._docs) == knowledge_store.document_count

    def test_all_docs_have_required_fields(self):
        for doc in knowledge_store._docs:
            for field in ("id", "type", "title", "content", "tags"):
                assert field in doc, f"Doc {doc.get('id')} missing field '{field}'"

    def test_doc_ids_are_unique(self):
        ids = [d["id"] for d in knowledge_store._docs]
        assert len(ids) == len(set(ids)), "Duplicate document IDs found"

    def test_json_kb_types_present(self):
        """
        Verify that all four structured KB types are present.

        Notes
        -----
        The JSON knowledge base files must contribute documents of types
        ``runbook``, ``template``, ``incident``, and ``guideline``. Missing
        types indicate a KB loading failure.
        """
        expected = {"runbook", "template", "incident", "guideline"}
        present  = {d["type"] for d in knowledge_store._docs}
        assert expected & present == expected, f"Missing KB types: {expected - present}"

    def test_markdown_docs_present(self):
        md_docs = [d for d in knowledge_store._docs if d["type"] == "documentation"]
        assert len(md_docs) >= 5, f"Expected >=5 markdown docs, got {len(md_docs)}"

    def test_id_lookup_table_built(self):
        assert len(knowledge_store._id_to_doc) == len(knowledge_store._docs)


class TestChromaCollection:
    """
    Tests for ChromaDB collection integrity.

    Notes
    -----
    Verifies that the collection uses cosine similarity space and that
    a stale collection (wrong document count) triggers a rebuild.
    """

    def test_collection_count_nonzero(self):
        assert knowledge_store._collection.count() > 0

    def test_collection_count_equals_doc_list(self):
        assert knowledge_store._collection.count() == len(knowledge_store._docs)

    def test_collection_uses_cosine_space(self):
        meta = knowledge_store._collection.metadata or {}
        assert meta.get("hnsw:space") == "cosine"

    def test_rebuild_on_stale_collection(self, tmp_path):
        """
        Verify that a KnowledgeStore detects stale collections and rebuilds.

        Notes
        -----
        Manually adds a fake document to a fresh EphemeralClient collection
        to simulate a stale state (count mismatch). A second KnowledgeStore
        pointing at a PersistentClient must then rebuild to the correct count.
        """
        store = KnowledgeStore(_ephemeral=True)
        original_count = store.document_count
        assert original_count > 0

        store._collection.add(
            ids=["fake-doc-99999"],
            documents=["fake content"],
            metadatas=[{"type": "runbook", "title": "Fake", "tags": "fake"}],
        )
        assert store._collection.count() == original_count + 1

        store2 = KnowledgeStore(_ephemeral=False, chroma_path=str(tmp_path))
        assert store2.document_count == original_count


class TestMmrRetrieval:
    """
    Tests for MMR-diversified document retrieval.

    Notes
    -----
    Covers result count, field presence, relevance score bounds, lambda
    propagation, document type filtering, diversity vs relevance trade-off,
    and empty query handling.
    """

    def test_returns_results_for_valid_query(self):
        results = knowledge_store.retrieve("api gateway SLO template")
        assert len(results) > 0

    def test_empty_query_returns_empty(self):
        assert knowledge_store.retrieve("") == []
        assert knowledge_store.retrieve("   ") == []

    def test_result_count_respects_top_k(self):
        for k in (1, 3, 5):
            results = knowledge_store.retrieve("slo burn rate", top_k=k)
            assert len(results) <= k

    def test_result_has_required_fields(self):
        results = knowledge_store.retrieve("payment service slo", top_k=1)
        assert len(results) == 1
        r = results[0]
        for field in ("id", "type", "title", "content", "relevance", "mmr_lambda_used"):
            assert field in r, f"Result missing field: {field}"

    def test_relevance_scores_in_unit_interval(self):
        results = knowledge_store.retrieve("error budget burn rate")
        for r in results:
            assert 0.0 <= r["relevance"] <= 1.0, (
                f"Relevance {r['relevance']} out of [0, 1]"
            )

    def test_mmr_lambda_returned_in_results(self):
        results = knowledge_store.retrieve("incident response database", top_k=2)
        for r in results:
            assert 0.0 <= r["mmr_lambda_used"] <= 1.0

    def test_explicit_mmr_lambda_respected(self):
        results = knowledge_store.retrieve("slo template", top_k=2, mmr_lambda=0.9)
        for r in results:
            assert abs(r["mmr_lambda_used"] - 0.9) < 1e-6

    def test_doc_type_filter_runbook(self):
        results = knowledge_store.retrieve("burn rate alert", top_k=5, doc_type="runbook")
        for r in results:
            assert r["type"] == "runbook"

    def test_doc_type_filter_template(self):
        results = knowledge_store.retrieve("payment service", top_k=5, doc_type="template")
        for r in results:
            assert r["type"] == "template"

    def test_doc_type_filter_incident(self):
        results = knowledge_store.retrieve("cascade failure", top_k=5, doc_type="incident")
        for r in results:
            assert r["type"] == "incident"

    def test_doc_type_all_returns_mixed_types(self):
        results = knowledge_store.retrieve("slo reliability database", top_k=10, doc_type="all")
        types = {r["type"] for r in results}
        assert len(types) > 1, "Mixed query must return more than one document type"

    def test_nonexistent_doc_type_returns_empty(self):
        results = knowledge_store.retrieve("slo", doc_type="nonexistent_type")
        assert results == []

    def test_diversity_increases_with_lower_lambda(self):
        """
        Verify that lower lambda produces more diverse document types.

        Notes
        -----
        Lambda = 0.1 (high diversity) should yield at least as many distinct
        document types as lambda = 0.95 (high relevance). This is not strictly
        guaranteed but holds statistically for the current knowledge base.
        """
        diverse = knowledge_store.retrieve(
            "slo sre reliability service", top_k=6, mmr_lambda=0.1
        )
        focused = knowledge_store.retrieve(
            "slo sre reliability service", top_k=6, mmr_lambda=0.95
        )
        diverse_types = {r["type"] for r in diverse}
        focused_types = {r["type"] for r in focused}
        assert len(diverse_types) >= len(focused_types)

    def test_content_field_populated_from_source(self):
        """
        Verify that content comes from the original source document.

        Notes
        -----
        ChromaDB only stores truncated document strings in its index metadata.
        The full content must be retrieved from the ``_id_to_doc`` lookup and
        must not be empty.
        """
        results = knowledge_store.retrieve("error budget", top_k=1)
        assert len(results) == 1
        assert len(results[0]["content"]) > 0


class TestAdaptiveLambda:
    """
    Tests for automatic MMR lambda selection based on query intent signals.

    Notes
    -----
    Incident signals map to lambda = 0.60, exploratory signals to 0.50,
    and default queries to 0.70. Incident signals take priority over
    exploratory signals when both are present.
    """

    def test_incident_query_returns_incident_lambda(self):
        assert _infer_mmr_lambda("incident cascade failure outage") == 0.60

    def test_exploratory_query_returns_lower_lambda(self):
        assert _infer_mmr_lambda("what SLO should I set for a payment service") == 0.50

    def test_plain_query_returns_default_lambda(self):
        assert _infer_mmr_lambda("api-gateway slo template 99.9%") == 0.70

    def test_incident_signal_beats_exploratory(self):
        """
        Verify that incident signals override exploratory signals.

        Notes
        -----
        When a query contains both ``"incident"`` (incident signal) and
        ``"how"`` (exploratory signal), the incident priority should win
        and return lambda = 0.60.
        """
        lam = _infer_mmr_lambda("how should I respond to an incident")
        assert lam == 0.60

    def test_retrieve_uses_adaptive_lambda_for_incident_query(self):
        results = knowledge_store.retrieve("incident cascade failure", top_k=2)
        for r in results:
            assert r["mmr_lambda_used"] == 0.60

    def test_retrieve_uses_adaptive_lambda_for_exploratory_query(self):
        results = knowledge_store.retrieve("what should the SLO be", top_k=2)
        for r in results:
            assert r["mmr_lambda_used"] == 0.50


class TestRetrieveForService:
    """
    Tests for context-enriched service retrieval via ``retrieve_for_service``.

    Notes
    -----
    Verifies that drift flags and anomaly severity correctly set lambda to
    0.60 (incident-balanced), and that normal conditions allow adaptive
    lambda selection.
    """

    def test_returns_results_for_standard_service(self):
        results = knowledge_store.retrieve_for_service(
            service_name="api-gateway",
            service_type="api",
            metrics_summary={
                "tier": "critical",
                "drift_detected": False,
                "anomaly_severity": "none",
            },
        )
        assert len(results) > 0

    def test_drift_flag_sets_incident_lambda(self):
        """
        Verify that ``drift_detected=True`` forces lambda to 0.60.

        Notes
        -----
        When distribution drift is detected, the retrieval should use the
        incident-tuned lambda of 0.60 to balance relevance and diversity
        for incident response documents.
        """
        results = knowledge_store.retrieve_for_service(
            service_name="payment-service",
            service_type="payment",
            metrics_summary={
                "tier": "critical",
                "drift_detected": True,
                "anomaly_severity": "none",
                "has_external_deps": True,
            },
            top_k=3,
        )
        assert len(results) > 0
        for r in results:
            assert r["mmr_lambda_used"] == 0.60

    def test_anomaly_severity_sets_incident_lambda(self):
        results = knowledge_store.retrieve_for_service(
            service_name="auth-service",
            service_type="auth",
            metrics_summary={
                "tier": "critical",
                "drift_detected": False,
                "anomaly_severity": "critical",
            },
            top_k=3,
        )
        assert len(results) > 0
        for r in results:
            assert r["mmr_lambda_used"] == 0.60

    def test_normal_conditions_use_adaptive_lambda(self):
        """
        Verify adaptive lambda is used when no drift or anomaly is present.

        Notes
        -----
        With no drift and no anomaly, lambda is determined by the assembled
        query string. The query always contains ``"latency"`` which is in
        the incident signal set, so lambda will be 0.60.
        """
        results = knowledge_store.retrieve_for_service(
            service_name="inventory-service",
            service_type="inventory",
            metrics_summary={
                "tier": "medium",
                "drift_detected": False,
                "anomaly_severity": "none",
            },
            top_k=2,
        )
        assert len(results) > 0
        for r in results:
            assert r["mmr_lambda_used"] == 0.60

    def test_result_count_respects_top_k(self):
        for k in (1, 3, 5):
            results = knowledge_store.retrieve_for_service(
                service_name="db",
                service_type="database",
                metrics_summary={},
                top_k=k,
            )
            assert len(results) <= k

    def test_result_types_are_valid(self):
        valid = {"runbook", "template", "incident", "guideline", "documentation"}
        results = knowledge_store.retrieve_for_service(
            service_name="checkout",
            service_type="commerce",
            metrics_summary={"drift_detected": True, "anomaly_severity": "high"},
            top_k=5,
        )
        for r in results:
            assert r["type"] in valid
