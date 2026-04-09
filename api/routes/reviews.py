"""
Review endpoints for human-in-the-loop SLO recommendation approval.

Notes
-----
Provides REST endpoints for engineers to approve, reject, or modify
SLO recommendations that have been flagged for human review due to low
confidence scores or detected metric drift.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from slo_engine.review_store import (
    ReviewDecisionBody,
    get_feedback_summary,
    get_review_status,
    list_pending_reviews,
    submit_review_decision,
)

router = APIRouter()


@router.get("/reviews/feedback/summary", summary="Feedback loop summary — approval rates per service")
async def feedback_summary() -> dict:
    """
    Return a summary of the feedback loop including approval rates per service.

    Returns
    -------
    dict
        Dictionary with aggregate approval statistics and per-service breakdown.

    Notes
    -----
    Delegates to ``slo_engine.review_store.get_feedback_summary``. Useful for
    monitoring the HITL gate effectiveness over time.
    """
    return get_feedback_summary()


@router.get("/reviews/pending", summary="List all pending SLO reviews")
async def get_pending_reviews() -> list[dict]:
    """
    Return a list of all SLO recommendations awaiting human review.

    Returns
    -------
    list of dict
        List of pending review dictionaries, each containing recommendation
        details, confidence score, and time remaining before timeout.

    Notes
    -----
    Delegates to ``slo_engine.review_store.list_pending_reviews``.
    """
    return list_pending_reviews()


@router.get("/reviews/{recommendation_id}", summary="Get review status")
async def get_review(recommendation_id: str) -> dict:
    """
    Return the current review status for a specific recommendation.

    Parameters
    ----------
    recommendation_id : str
        Unique identifier of the SLO recommendation to look up.

    Returns
    -------
    dict
        Review status dictionary including decision, reviewer, and timestamps
        if the review has been completed.

    Notes
    -----
    Raises HTTP 404 if no review record exists for the given ID.
    """
    import json
    result = json.loads(get_review_status(recommendation_id))
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Review '{recommendation_id}' not found.")
    return result


@router.post("/reviews/{recommendation_id}/decision", summary="Submit review decision")
async def submit_decision(recommendation_id: str, body: ReviewDecisionBody) -> dict:
    """
    Submit a human review decision for a pending SLO recommendation.

    Parameters
    ----------
    recommendation_id : str
        Unique identifier of the recommendation being reviewed.
    body : ReviewDecisionBody
        Decision payload containing the decision (approve/reject/modify),
        reviewer identity, optional comment, and optional modified SLO values.

    Returns
    -------
    dict
        Result dictionary confirming the submitted decision.

    Notes
    -----
    The ``recommendation_id`` from the URL path is injected into the body
    before delegating to the review store, so callers do not need to duplicate
    it in the request body. Raises HTTP 404 if the recommendation is not found.
    """
    body.recommendation_id = recommendation_id
    result = submit_review_decision(body)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result
