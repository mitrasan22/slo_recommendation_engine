"""
RAG knowledge store backed by a ChromaDB persistent vector database.

Notes
-----
Documents are loaded from two on-disk sources (no inline knowledge in code):

  ``data/knowledge_base/*.json``
      Structured SRE documents (runbooks, templates, incidents, guidelines)
      with typed metadata.
  ``docs/*.md``
      Prose operational documents.

On first run, or when the document count changes, all documents are embedded
by sentence-transformers and stored in a ChromaDB persistent collection.
Subsequent process starts reuse the persisted index — startup is fast with no
re-embedding required.

Retrieval strategy:

  1. ChromaDB HNSW approximate nearest-neighbour search returns a candidate
     pool (``top_k * 10``) with cosine distances and document embeddings.
  2. MMR (Maximal Marginal Relevance) is applied on those embeddings in numpy
     for diversity-aware final selection.
  3. Document type filtering is pushed down into ChromaDB's ``where`` clause —
     not post-filtered in Python.
  4. Lambda is selected adaptively from query intent signals (exploratory /
     incident / lookup / default) unless overridden by the caller.

Storage / client selection (evaluated in order):

  ``PYTEST_CURRENT_TEST`` / ``ENVIRONMENT=test``
      EphemeralClient (in-memory, for tests).
  ``CHROMA_HOST`` set
      HttpClient (production K8s server pod).
  Fallback
      PersistentClient (local dev, ``CHROMA_PATH``).

Environment variables:
  ``CHROMA_HOST``  — hostname of the ChromaDB server pod.
  ``CHROMA_PORT``  — port of the ChromaDB server (default: 8000).
  ``CHROMA_PATH``  — local persist directory (default: ``<repo>/data/chromadb``).
  ``EMBED_MODEL``  — sentence-transformers model (default: all-MiniLM-L6-v2).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_MODEL_NAME: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

_REPO_ROOT       = Path(__file__).resolve().parents[3]
_KB_DIR          = _REPO_ROOT / "data" / "knowledge_base"
_DOCS_DIR        = _REPO_ROOT / "docs"
_CHROMA_PATH     = os.getenv("CHROMA_PATH", str(_REPO_ROOT / "data" / "chromadb"))
_CHROMA_HOST     = os.getenv("CHROMA_HOST", "")
_CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
_COLLECTION_NAME = "slo_knowledge_base"

_IS_TEST = (
    os.getenv("ENVIRONMENT") == "test"
    or "PYTEST_CURRENT_TEST" in os.environ
)

_MMR_LAMBDA_BY_INTENT: dict[str, float] = {
    "exploratory": 0.50,
    "lookup":      0.80,
    "incident":    0.60,
    "default":     0.70,
}

_EXPLORATORY_SIGNALS = frozenset([
    "what", "how", "should", "recommend", "best", "advice", "guide",
    "choose", "suggest", "why", "when",
])
_INCIDENT_SIGNALS = frozenset([
    "incident", "outage", "breach", "failure", "cascade", "error", "down",
    "degraded", "latency", "spike", "alert", "burn", "exhausted",
])


def _load_json_kb(kb_dir: Path) -> list[dict]:
    """
    Load all ``*.json`` files from the knowledge base directory.

    Parameters
    ----------
    kb_dir : Path
        Directory containing JSON knowledge base files.

    Returns
    -------
    list of dict
        Flattened list of all document dicts from all JSON files. Files that
        contain a JSON array contribute multiple documents each.

    Notes
    -----
    Files that fail to parse (invalid JSON or OS errors) are silently skipped.
    Returns an empty list when ``kb_dir`` does not exist.
    """
    docs: list[dict] = []
    if not kb_dir.exists():
        return docs
    for json_file in sorted(kb_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                docs.extend(data)
        except (json.JSONDecodeError, OSError):
            pass
    return docs


def _load_markdown_docs(docs_dir: Path) -> list[dict]:
    """
    Load ``*.md`` files as knowledge base entries.

    Parameters
    ----------
    docs_dir : Path
        Directory containing Markdown documentation files.

    Returns
    -------
    list of dict
        One document dict per Markdown file with keys ``id``, ``type``,
        ``title``, ``content``, and ``tags``.

    Notes
    -----
    Each file is mapped to a single document entry:

    - ``id``      = ``"doc-{stem}"``
    - ``type``    = ``"documentation"``
    - ``title``   = first H1 heading if present, otherwise stem in Title Case
    - ``content`` = full text truncated to 2000 characters
    - ``tags``    = ``["documentation"]`` plus stem words split by underscore

    Files that cannot be read due to OS errors are silently skipped.
    Returns an empty list when ``docs_dir`` does not exist.
    """
    docs: list[dict] = []
    if not docs_dir.exists():
        return docs
    for md_file in sorted(docs_dir.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        title = md_file.stem.replace("_", " ").title()
        for line in text.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
        tags    = ["documentation"] + md_file.stem.split("_")
        content = text[:2000] if len(text) > 2000 else text
        docs.append({
            "id":      f"doc-{md_file.stem}",
            "type":    "documentation",
            "title":   title,
            "content": content,
            "tags":    tags,
        })
    return docs


def _infer_mmr_lambda(query: str) -> float:
    """
    Select an MMR lambda value from query intent signals.

    Parameters
    ----------
    query : str
        Free-text query string from the caller.

    Returns
    -------
    float
        Lambda value in ``[0.0, 1.0]``: ``0.60`` for incident queries,
        ``0.50`` for exploratory queries, ``0.70`` otherwise.

    Notes
    -----
    Incident signals take priority over exploratory signals. Token matching is
    exact-word based (lowercased split), not substring based. The ``"lookup"``
    intent (``0.80``) is not inferred automatically — it must be set explicitly
    by the caller via the ``mmr_lambda`` parameter.
    """
    tokens = set(query.lower().split())
    if tokens & _INCIDENT_SIGNALS:
        return _MMR_LAMBDA_BY_INTENT["incident"]
    if tokens & _EXPLORATORY_SIGNALS:
        return _MMR_LAMBDA_BY_INTENT["exploratory"]
    return _MMR_LAMBDA_BY_INTENT["default"]


class KnowledgeStore:
    """
    Semantic knowledge store backed by ChromaDB persistent vector database.

    Attributes
    ----------
    _docs : list of dict
        All source documents loaded from disk at construction time.
    _id_to_doc : dict of str to dict
        Mapping from document ID to document dict for content retrieval.
    _embed_fn : SentenceTransformerEmbeddingFunction
        ChromaDB-compatible embedding function using sentence-transformers.
    _client : chromadb.ClientAPI
        Active ChromaDB client (Ephemeral, Http, or Persistent).
    _collection : chromadb.Collection
        ChromaDB collection holding the indexed documents.

    Notes
    -----
    Index lifecycle: on init the collection document count is compared against
    the current source files. A match means the persisted index is reused (fast
    path). A mismatch triggers a full delete-and-rebuild of the collection.
    ``_index_documents`` uses ``upsert`` (not ``add``) so multiple API pods
    starting simultaneously against the same HttpClient server are safe.
    """

    def __init__(
        self,
        kb_dir:          Path = _KB_DIR,
        docs_dir:        Path = _DOCS_DIR,
        chroma_path:     str  = _CHROMA_PATH,
        collection_name: str  = _COLLECTION_NAME,
        _ephemeral:      bool = _IS_TEST,
    ) -> None:
        """
        Initialise the knowledge store by loading documents and connecting to ChromaDB.

        Parameters
        ----------
        kb_dir : Path, optional
            Directory of JSON knowledge base files. Defaults to ``data/knowledge_base``.
        docs_dir : Path, optional
            Directory of Markdown documentation files. Defaults to ``docs``.
        chroma_path : str, optional
            File-system path for PersistentClient storage. Defaults to
            ``data/chromadb`` relative to the repository root.
        collection_name : str, optional
            Name of the ChromaDB collection to use. Defaults to
            ``"slo_knowledge_base"``.
        _ephemeral : bool, optional
            Force an EphemeralClient (in-memory). Automatically ``True``
            during test runs.

        Returns
        -------
        None

        Notes
        -----
        Client selection priority: EphemeralClient (test) > HttpClient
        (``CHROMA_HOST`` set) > PersistentClient (default). The collection
        is built or verified via ``_get_or_build_collection`` before returning.
        """
        json_docs = _load_json_kb(kb_dir)
        md_docs   = _load_markdown_docs(docs_dir)
        self._docs: list[dict] = json_docs + md_docs

        self._id_to_doc: dict[str, dict] = {d["id"]: d for d in self._docs}

        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=_MODEL_NAME,
            normalize_embeddings=True,
        )

        if _ephemeral:
            self._client: chromadb.ClientAPI = chromadb.EphemeralClient()
        elif _CHROMA_HOST:
            self._client = chromadb.HttpClient(
                host=_CHROMA_HOST,
                port=_CHROMA_PORT,
            )
        else:
            self._client = chromadb.PersistentClient(path=chroma_path)

        self._collection = self._get_or_build_collection(collection_name)

    def _get_or_build_collection(self, name: str) -> chromadb.Collection:
        """
        Return the persisted collection if current, otherwise rebuild it.

        Parameters
        ----------
        name : str
            ChromaDB collection name.

        Returns
        -------
        chromadb.Collection
            Ready-to-query collection with all documents indexed.

        Notes
        -----
        Collection metadata ``hnsw:space=cosine`` means distance = 1 - cosine
        similarity. If the collection's document count matches the number of
        loaded source documents, the fast path returns immediately without
        re-embedding. Otherwise the collection is deleted and recreated from
        scratch via ``_index_documents``.
        """
        collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=cast(Any, self._embed_fn),
            metadata={"hnsw:space": "cosine"},
        )

        if collection.count() == len(self._docs):
            return collection

        try:
            self._client.delete_collection(name)
        except Exception:
            pass

        collection = self._client.create_collection(
            name=name,
            embedding_function=cast(Any, self._embed_fn),
            metadata={"hnsw:space": "cosine"},
        )
        self._index_documents(collection)
        return collection

    def _index_documents(self, collection: chromadb.Collection) -> None:
        """
        Batch-upsert all source documents into the ChromaDB collection.

        Parameters
        ----------
        collection : chromadb.Collection
            Target ChromaDB collection to upsert documents into.

        Returns
        -------
        None

        Notes
        -----
        Documents are processed in batches of 100 to avoid ChromaDB payload
        limits. Uses ``upsert`` (not ``add``) so that multiple API pods
        starting simultaneously against the same HttpClient server are safe —
        duplicate writes overwrite with identical data and are idempotent.
        Metadata stored per document: ``type``, ``title``, and ``tags``
        (space-joined for ChromaDB compatibility).
        """
        batch_size = 100
        for start in range(0, len(self._docs), batch_size):
            batch = self._docs[start : start + batch_size]
            collection.upsert(
                ids=[d["id"] for d in batch],
                documents=[
                    f"{d['title']} {d['content']} {' '.join(d['tags'])}"
                    for d in batch
                ],
                metadatas=[
                    {
                        "type":  d["type"],
                        "title": d["title"],
                        "tags":  " ".join(d["tags"]),
                    }
                    for d in batch
                ],
            )

    @property
    def document_count(self) -> int:
        """
        Total number of documents in the ChromaDB collection.

        Returns
        -------
        int
            Live document count queried directly from ChromaDB.

        Notes
        -----
        Reflects the ChromaDB collection's current state, which may differ
        from ``len(self._docs)`` during a concurrent rebuild.
        """
        return self._collection.count()

    def retrieve(
        self,
        query:      str,
        top_k:      int = 3,
        doc_type:   str = "all",
        mmr_lambda: float | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k documents via ChromaDB ANN search and MMR diversification.

        Parameters
        ----------
        query : str
            Free-text search query.
        top_k : int, optional
            Maximum number of results to return. Defaults to 3.
        doc_type : str, optional
            Filter by document type. One of ``"runbook"``, ``"template"``,
            ``"incident"``, ``"guideline"``, ``"documentation"``, or ``"all"``.
            Defaults to ``"all"``.
        mmr_lambda : float or None, optional
            MMR relevance vs diversity trade-off in ``[0.0, 1.0]``.
            ``1.0`` = pure relevance, ``0.0`` = pure diversity.
            ``None`` triggers intent-based automatic selection.

        Returns
        -------
        list of dict
            Each dict has keys ``id``, ``type``, ``title``, ``content``,
            ``relevance`` (cosine similarity), and ``mmr_lambda_used``.

        Notes
        -----
        Retrieval proceeds in three steps:

        1. ChromaDB HNSW search returns a candidate pool of ``top_k * 10``
           results with cosine distances and embeddings.
        2. Distances are converted to similarities via ``sim = 1 - dist``.
        3. MMR greedy selection iterates over the candidate pool, scoring each
           unselected candidate as
           ``lambda * sim[i] - (1-lambda) * max_sim_to_selected``.
           This is an ``O(n_candidates * dim)`` numpy dot-product, negligible
           in practice.

        On ChromaDB query failure with a ``where`` filter (older versions may
        reject ``n_results > filtered_count``), the call is retried with
        ``n_results = top_k`` as a fallback.
        """
        if not query or not query.strip():
            return []

        lam = mmr_lambda if mmr_lambda is not None else _infer_mmr_lambda(query)

        total = self._collection.count()
        if total == 0:
            return []

        n_candidates = min(top_k * 10, total)
        where = cast(Any, {"type": doc_type} if doc_type != "all" else None)

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_candidates,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=max(1, top_k),
                    where=where,
                    include=["documents", "metadatas", "distances", "embeddings"],
                )
            except Exception:
                return []

        result_map = cast(dict[str, Any], results)
        ids = cast(list[str], result_map["ids"][0])
        metadatas = cast(list[dict[str, Any]], result_map["metadatas"][0])
        distances = cast(list[float], result_map["distances"][0])
        embeddings = cast(list[list[float]], result_map["embeddings"][0])

        if not ids:
            return []

        similarities = [max(0.0, 1.0 - d) for d in distances]

        cand_embs = np.array(embeddings, dtype=np.float32)
        selected:  list[int] = []
        remaining: list[int] = list(range(len(ids)))
        n_select = min(top_k, len(ids))

        while len(selected) < n_select and remaining:
            if not selected:
                best = max(remaining, key=lambda i: similarities[i])
            else:
                sel_embs = cand_embs[selected]

                def _mmr(i: int, _l: float = lam, _s: np.ndarray = sel_embs) -> float:
                    return _l * similarities[i] - (1.0 - _l) * float(
                        (cand_embs[i] @ _s.T).max()
                    )

                best = max(remaining, key=_mmr)

            selected.append(best)
            remaining.remove(best)

        output = []
        for pos in selected:
            doc_id = ids[pos]
            meta   = metadatas[pos]
            src    = self._id_to_doc.get(doc_id, {})
            output.append({
                "id":              doc_id,
                "type":            meta.get("type", "unknown"),
                "title":           meta.get("title", src.get("title", "")),
                "content":         src.get("content", "").replace("\r\n", "\n").replace("\r", "\n"),
                "relevance":       round(float(similarities[pos]), 4),
                "mmr_lambda_used": round(lam, 2),
            })
        return output

    def retrieve_for_service(
        self,
        service_name:    str,
        service_type:    str,
        metrics_summary: dict,
        top_k:           int = 4,
    ) -> list[dict]:
        """
        Build a context-enriched query from service attributes and retrieve documents.

        Parameters
        ----------
        service_name : str
            Name of the target service (appended to the query for specificity).
        service_type : str
            Service type keyword (e.g. ``"api_gateway"``, ``"database"``).
        metrics_summary : dict
            Metrics context dict with optional keys ``tier``, ``drift_detected``,
            ``anomaly_severity``, and ``has_external_deps``.
        top_k : int, optional
            Maximum number of results to return. Defaults to 4.

        Returns
        -------
        list of dict
            Retrieved documents in MMR-ranked order. See ``retrieve`` for the
            per-document field schema.

        Notes
        -----
        Drift detected or non-trivial anomaly severity shifts lambda to ``0.60``
        (incident-balanced retrieval). Otherwise lambda is inferred automatically
        from the assembled query string. External dependency context appends
        circuit-breaker and cascade-failure keywords to improve retrieval.
        """
        tier    = metrics_summary.get("tier", "medium")
        drift   = metrics_summary.get("drift_detected", False)
        anomaly = metrics_summary.get("anomaly_severity", "none")
        has_ext = metrics_summary.get("has_external_deps", False)

        parts = [service_name, service_type, "availability latency slo recommendation", tier]
        if drift:
            parts.append("distribution drift change kl divergence detection")
        if anomaly not in ("none", "low"):
            parts.append(f"anomaly {anomaly} severity incident breach")
        if has_ext:
            parts.append("external dependency circuit breaker reliability cascade")

        lam = 0.60 if (drift or anomaly not in ("none", "low")) else None
        return self.retrieve(" ".join(parts), top_k=top_k, mmr_lambda=lam)


knowledge_store = KnowledgeStore()
