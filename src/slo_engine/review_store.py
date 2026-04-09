"""
Human-in-the-loop (HITL) review store for the SLO recommendation pipeline.

Notes
-----
Not an agent or workflow — a plain dict-backed store (Redis in production)
with five public functions:

  ``submit_for_human_review``   — queues a low-confidence recommendation
  ``get_review_status``         — polls status and handles timeout auto-approve
  ``submit_review_decision``    — engineer approves/rejects/modifies via REST
  ``list_pending_reviews``      — dashboard feed of outstanding reviews
  ``get_feedback_summary``      — aggregated approval stats from the feedback log

Called directly by:

  - ``slo_engine/agents/agent.py``  — ``_gate_and_finalize()``
  - ``api/routes/reviews.py``       — REST endpoints
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any

from pydantic import BaseModel, Field

from slo_engine.config import config as _config
from slo_engine.utils.pii_scrubber import scrub as _scrub

_FEEDBACK_FILE = pathlib.Path("feedback_log.jsonl")

logger = logging.getLogger(__name__)

_pending:   dict[str, dict[str, Any]] = {}
_completed: dict[str, dict[str, Any]] = {}


class ReviewRequest(BaseModel):
    """
    Pydantic schema for a recommendation submitted for human review.

    Attributes
    ----------
    recommendation_id : str
        Unique identifier for the recommendation being reviewed.
    service_name : str
        Name of the service the recommendation applies to.
    recommended_availability : float
        Proposed availability SLO target in the range ``[0.0, 1.0]``.
    recommended_latency_p99_ms : float
        Proposed p99 latency SLO target in milliseconds.
    confidence_score : float
        Model confidence score for the recommendation, in ``[0.0, 1.0]``.
    review_reason : str, optional
        Human-readable explanation of why the recommendation was flagged for
        review. Defaults to ``""``.

    Notes
    -----
    Validated by ``submit_for_human_review`` using ``model_validate_json``.
    All fields are stored verbatim in the in-memory pending store without
    further transformation.
    """

    recommendation_id: str
    service_name: str
    recommended_availability: float
    recommended_latency_p99_ms: float
    confidence_score: float
    review_reason: str = ""


class ReviewDecisionBody(BaseModel):
    """
    Pydantic schema for an engineer's review decision submitted via REST.

    Attributes
    ----------
    recommendation_id : str
        ID of the recommendation being decided on.
    decision : str
        One of ``"approve"``, ``"reject"``, or ``"modify"``.
    reviewer : str
        Identity of the engineer submitting the decision (PII-scrubbed in logs).
    comment : str, optional
        Free-text comment attached to the decision. Defaults to ``""``.
    modified_availability : float or None, optional
        Overridden availability target when ``decision == "modify"``.
    modified_latency_p99_ms : float or None, optional
        Overridden p99 latency target when ``decision == "modify"``.

    Notes
    -----
    The ``reviewer`` and ``comment`` fields are passed through ``_scrub`` before
    being written to the feedback log to avoid persisting PII.
    """

    recommendation_id: str
    decision: str
    reviewer: str
    comment: str = ""
    modified_availability: float | None = None
    modified_latency_p99_ms: float | None = None


def submit_for_human_review(review_request_json: str) -> str:
    """
    Queue a recommendation for human review and return immediately.

    Parameters
    ----------
    review_request_json : str
        JSON string conforming to the ``ReviewRequest`` schema.

    Returns
    -------
    str
        JSON string with keys ``recommendation_id``, ``status``
        (``"pending_review"``), ``review_deadline_seconds``, and
        ``review_url``. Returns ``{"status": "error", "message": ...}``
        on parse or validation failure.

    Notes
    -----
    The review timeout is read from ``config.hitl.review_timeout_seconds``
    (default: 3600 seconds). In production this function would additionally
    publish the request to a Redis pub/sub channel for real-time dashboard
    notification.
    """
    try:
        req = ReviewRequest.model_validate_json(review_request_json)
        timeout = int(getattr(getattr(_config, "hitl", None), "review_timeout_seconds", 3600))

        _pending[req.recommendation_id] = {
            "request":      req.model_dump(),
            "submitted_at": time.time(),
            "deadline":     time.time() + timeout,
        }

        return json.dumps({
            "recommendation_id":     req.recommendation_id,
            "status":                "pending_review",
            "review_deadline_seconds": timeout,
            "review_url":            f"/api/v1/reviews/{req.recommendation_id}",
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def get_review_status(recommendation_id: str) -> str:
    """
    Check the review status for a recommendation, auto-approving on timeout.

    Parameters
    ----------
    recommendation_id : str
        Unique identifier of the recommendation to check.

    Returns
    -------
    str
        JSON string with ``recommendation_id``, ``status``, and conditionally
        ``decision``, ``reviewer``, and ``timed_out`` fields.
        Returns ``{"status": "not_found"}`` for unknown IDs and
        ``{"status": "error", "message": ...}`` on exceptions.

    Notes
    -----
    When a pending recommendation's deadline has passed, it is automatically
    moved to completed with ``decision="approve"`` and ``timed_out=True``.
    This prevents the pipeline from blocking indefinitely when reviewers do
    not respond.
    """
    try:
        if recommendation_id in _completed:
            c = _completed[recommendation_id]
            return json.dumps({
                "recommendation_id": recommendation_id,
                "status":    c["status"],
                "decision":  c["decision"],
                "reviewer":  c.get("reviewer"),
                "timed_out": c.get("timed_out", False),
            })

        if recommendation_id in _pending:
            p = _pending[recommendation_id]
            if time.time() > p["deadline"]:
                _completed[recommendation_id] = {
                    "status":    "approved",
                    "decision":  "approve",
                    "reviewer":  "auto-timeout",
                    "timed_out": True,
                }
                del _pending[recommendation_id]
                return json.dumps({
                    "recommendation_id": recommendation_id,
                    "status": "approved", "decision": "approve", "timed_out": True,
                })
            return json.dumps({"recommendation_id": recommendation_id, "status": "pending"})

        return json.dumps({"recommendation_id": recommendation_id, "status": "not_found"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def _append_feedback(rec_id: str, body: ReviewDecisionBody, original_request: dict) -> None:
    """
    Append a review decision record to the JSONL feedback log.

    Parameters
    ----------
    rec_id : str
        Recommendation ID being logged.
    body : ReviewDecisionBody
        The engineer's decision body, including reviewer identity and comment.
    original_request : dict
        The original recommendation request dict from the pending store.

    Returns
    -------
    None

    Notes
    -----
    The feedback log at ``feedback_log.jsonl`` is used for future confidence
    score calibration. ``reviewer`` and ``comment`` are passed through
    ``_scrub`` before writing to avoid persisting PII. Write failures are
    logged at WARNING level and do not propagate.
    """
    record = {
        "timestamp": time.time(),
        "recommendation_id": rec_id,
        "service_name": original_request.get("service_name", "unknown"),
        "original_availability": original_request.get("recommended_availability"),
        "original_confidence": original_request.get("confidence_score"),
        "decision": body.decision,
        "reviewer": _scrub(body.reviewer),
        "modified_availability": body.modified_availability,
        "comment": _scrub(body.comment) if body.comment else body.comment,
    }
    try:
        with open(_FEEDBACK_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Feedback log write failed: %s", exc)


def submit_review_decision(body: ReviewDecisionBody) -> dict:
    """
    Record an engineer's approve, reject, or modify decision.

    Parameters
    ----------
    body : ReviewDecisionBody
        Validated decision body from the REST endpoint.

    Returns
    -------
    dict
        On success: ``{"status": "success", "recommendation_id": ..., "decision": ...}``.
        On not found: ``{"status": "error", "message": ...}``.

    Notes
    -----
    Moves the recommendation from ``_pending`` to ``_completed``. The
    completed entry's ``status`` field is the decision string with ``"d"``
    appended (``"approved"``, ``"rejected"``, ``"modified"``). For ``"modify"``
    decisions, the modified targets are stored under ``"modified_targets"``.
    The decision is appended to the feedback log via ``_append_feedback``.
    """
    rec_id = body.recommendation_id
    if rec_id not in _pending:
        return {"status": "error", "message": f"{rec_id} not found or already resolved."}

    modified = None
    if body.decision == "modify" and body.modified_availability:
        modified = {
            "availability":    body.modified_availability,
            "latency_p99_ms":  body.modified_latency_p99_ms,
        }

    _completed[rec_id] = {
        "status":          body.decision + "d",
        "decision":        body.decision,
        "reviewer":        body.reviewer,
        "comment":         body.comment,
        "modified_targets": modified,
        "timed_out":       False,
    }
    _append_feedback(rec_id, body, _pending[rec_id]["request"])
    del _pending[rec_id]
    logger.info("Review %s -> %s by %s", rec_id, body.decision, body.reviewer)
    return {"status": "success", "recommendation_id": rec_id, "decision": body.decision}


def get_feedback_summary() -> dict:
    """
    Read the feedback log and return approval/rejection stats aggregated by service.

    Returns
    -------
    dict
        Dict with ``total`` (int) and ``by_service`` (dict of service name to
        counts dict with keys ``approved``, ``rejected``, ``modified``, ``total``).

    Notes
    -----
    Reads ``feedback_log.jsonl`` from the current working directory line-by-line.
    Lines that fail JSON parsing are silently skipped. Returns
    ``{"total": 0, "by_service": {}}`` when the file does not exist.
    """
    if not _FEEDBACK_FILE.exists():
        return {"total": 0, "by_service": {}}
    records = []
    with open(_FEEDBACK_FILE) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass
    by_service: dict[str, dict] = {}
    for r in records:
        svc = r.get("service_name", "unknown")
        if svc not in by_service:
            by_service[svc] = {"approved": 0, "rejected": 0, "modified": 0, "total": 0}
        decision = r.get("decision", "")
        by_service[svc]["total"] += 1
        if decision == "approve":
            by_service[svc]["approved"] += 1
        elif decision == "reject":
            by_service[svc]["rejected"] += 1
        elif decision == "modify":
            by_service[svc]["modified"] += 1
    return {"total": len(records), "by_service": by_service}


def list_pending_reviews() -> list[dict]:
    """
    Return all pending reviews for the dashboard feed.

    Returns
    -------
    list of dict
        One dict per pending review with keys ``recommendation_id``,
        ``service_name``, ``confidence_score``, ``review_reason``,
        ``recommended_availability``, and ``time_remaining_seconds``.

    Notes
    -----
    ``time_remaining_seconds`` is clamped to zero — it will not go negative
    even if a timed-out entry has not yet been auto-approved by
    ``get_review_status``. The list reflects the in-memory state at call time.
    """
    now = time.time()
    return [
        {
            "recommendation_id":       rid,
            "service_name":            p["request"].get("service_name"),
            "confidence_score":        p["request"].get("confidence_score"),
            "review_reason":           p["request"].get("review_reason"),
            "recommended_availability": p["request"].get("recommended_availability"),
            "time_remaining_seconds":  max(0, int(p["deadline"] - now)),
        }
        for rid, p in _pending.items()
    ]
