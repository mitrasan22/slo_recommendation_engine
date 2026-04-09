"""
Schemas and state keys for the SLO recommendation agent workflow.

Notes
-----
Defines all ADK session state key constants and Pydantic output schemas used by
the SLO generator, feasibility checker, MILP optimizer, report synthesizer, and
orchestrator agents.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from google.adk.agents.base_agent import BaseAgentState
from pydantic import BaseModel, Field

REC_SERVICE_KEY             = "rec_service_name"
REC_SERVICES_LIST_KEY       = "rec_services_list"
REC_DEP_SLOS_KEY            = "rec_dep_slos"
REC_HIST_AVAIL_KEY          = "rec_historical_availability"
REC_WEIGHTS_KEY             = "rec_importance_weights"
REC_WORKFLOW_KEY            = "rec_workflow_step"
REC_GENERATION_KEY          = "rec_generation_output"
REC_FEASIBILITY_KEY         = "rec_feasibility_output"
REC_OPTIMIZER_KEY           = "rec_optimizer_output"
REC_REPORT_KEY              = "rec_report_output"
REC_KNOWLEDGE_KEY           = "rec_knowledge_output"
REC_KNOWLEDGE_RAW_KEY       = "rec_knowledge_raw_output"
REC_KNOWLEDGE_QUERY_KEY     = "rec_knowledge_query"
REC_PENDING_QUESTIONS_KEY   = "rec_pending_questions"
REC_USER_ANSWERS_KEY        = "rec_user_answers"


class RecWorkflowStep(StrEnum):
    """
    Enumeration of SLO recommendation workflow steps.

    Attributes
    ----------
    GENERATE : str
        Initial SLO generation step using Bayesian and reliability math.
    AWAIT_INPUT : str
        Paused state waiting for human tier assignment for new services.
    FEASIBILITY : str
        LP feasibility check and dependency ceiling validation step.
    OPTIMIZE : str
        MILP portfolio optimization step (skipped for single service).
    REPORT : str
        Final LLM report synthesis step.
    DONE : str
        Terminal state after the report has been written.

    Notes
    -----
    Steps are ordered and comparable using StrEnum. The orchestrator uses
    ``<=`` comparisons to skip already-completed steps on resume.
    """

    GENERATE    = "GENERATE"
    AWAIT_INPUT = "AWAIT_INPUT"
    FEASIBILITY = "FEASIBILITY"
    OPTIMIZE    = "OPTIMIZE"
    REPORT      = "REPORT"
    DONE        = "DONE"


class KnowledgeResult(BaseModel):
    """
    Structured knowledge retrieval result carried through session state.

    Attributes
    ----------
    source_ids : list of str
        Document IDs returned by the knowledge MCP server.
    context_summary : str
        Concatenated ``[id — TYPE] title\\ncontent`` blocks for all results.
    total_returned : int
        Number of documents actually returned (≤ top_k).
    kb_size : int
        Total documents in the knowledge base at query time.
    """

    source_ids:      list[str] = Field(default_factory=list)
    context_summary: str       = Field(default="")
    total_returned:  int       = Field(default=0)
    kb_size:         int       = Field(default=0)


class RecPlannerQuestion(BaseModel):
    """
    A structured question from the recommendation planner for human clarification.

    Attributes
    ----------
    question_id : str
        Stable snake_case identifier used to match answers to questions.
    question : str
        Human-readable question text shown to the engineer.
    options : list of str
        Suggested answer choices; the engineer may pick one or answer freely.

    Notes
    -----
    Serialised to JSON and stored in ``rec_pending_questions`` state key.
    The engineer's answers are stored in ``rec_user_answers`` on the next turn.
    """

    question_id: str   = Field(..., description="Stable identifier — used to match answers.")
    question: str      = Field(..., description="Question text shown to the user.")
    options: list[str] = Field(..., description="Suggested choices — user may pick one or answer freely.")


class SLOGeneratorDecisionSchema(BaseModel):
    """
    Structured output schema for the SLO generator decision step.

    Attributes
    ----------
    decision : Literal["READY", "NEEDS_INPUT"]
        Whether sufficient context is available to generate an SLO recommendation.
    questions_for_user : list of RecPlannerQuestion
        Non-empty only when ``decision="NEEDS_INPUT"``.

    Notes
    -----
    ``"NEEDS_INPUT"`` is triggered when a service is entirely new with no tier
    assignment. In this case ``questions_for_user`` is populated and the pipeline
    pauses at ``AWAIT_INPUT`` until the engineer responds.
    """

    decision: Literal["READY", "NEEDS_INPUT"] = Field(
        ...,
        description=(
            "READY — enough context to generate SLO recommendation. "
            "NEEDS_INPUT — service is new with no tier assignment; questions_for_user is populated."
        ),
    )
    questions_for_user: list[RecPlannerQuestion] = Field(
        default_factory=list,
        description="Non-empty only when decision=NEEDS_INPUT.",
    )


class SLORecommendationReport(BaseModel):
    """
    Structured output schema for the final SLO recommendation report.

    Attributes
    ----------
    service_name : str
        Name of the service for which the SLO is recommended.
    recommended_availability : float
        Recommended availability SLO value between 0 and 1.
    recommended_latency_p99_ms : float
        Recommended p99 latency SLO in milliseconds.
    recommended_error_rate : float
        Complement of the availability SLO (1 - availability).
    confidence_score : float
        Confidence in the recommendation between 0 and 1.
    is_feasible : bool
        Whether the proposed SLO passed the LP feasibility check.
    feasibility_score : float
        Numeric feasibility score between 0 and 1.
    availability_ceiling : float
        Maximum achievable availability based on dependency ceilings.
    requires_human_review : bool
        Whether engineer review is required before the SLO can be approved.
    review_reason : str
        Explanation of why human review is required, or empty string.
    pareto_optimal_slos : dict
        Mapping of service name to Pareto-optimal SLO from the MILP optimizer.
    error_budget_allocation : dict
        Mapping of service name to allocated error budget fraction.
    math_details : dict
        Raw mathematical output from series/parallel reliability, Monte Carlo,
        and CLT computations.
    reasoning : str
        Engineer-facing explanation of how the SLO was derived.
    summary : str
        2-3 sentence concise summary for display in review UIs.
    sources : list of str
        Knowledge source IDs cited in this recommendation.
    knowledge_context : str
        Retrieved runbook or incident context used for grounding.

    Notes
    -----
    Stored in state under ``REC_REPORT_KEY`` after the REPORT step completes.
    If feasibility fails and an ``adjusted_recommendation`` is present, the
    recommended availability and latency are overridden with the adjusted values.
    """

    service_name: str
    recommended_availability: float
    recommended_latency_p99_ms: float
    recommended_error_rate: float
    confidence_score: float
    is_feasible: bool
    feasibility_score: float
    availability_ceiling: float
    requires_human_review: bool
    review_reason: str
    pareto_optimal_slos: dict[str, float]
    error_budget_allocation: dict[str, float]
    math_details: dict[str, Any]
    reasoning: str
    summary: str
    sources: list[str] = Field(default_factory=list, description="Knowledge source IDs cited in this recommendation.")
    knowledge_context: str = Field(default="", description="Retrieved runbook/incident context used for grounding.")


class RecOrchestratorState(BaseAgentState):
    """
    Typed ADK session state for the recommendation orchestrator.

    Attributes
    ----------
    workflow_step : RecWorkflowStep
        Current step in the recommendation workflow.
    generation_output : dict or None
        Output from the GENERATE step, if completed.
    feasibility_output : dict or None
        Output from the FEASIBILITY step, if completed.
    optimizer_output : dict or None
        Output from the OPTIMIZE step, if completed.
    report_output : SLORecommendationReport or None
        Output from the REPORT step, if completed.
    pending_questions : list of RecPlannerQuestion
        Questions awaiting human answers when in AWAIT_INPUT state.
    user_answers : str
        Raw text of the engineer's answers to the planner's questions.

    Notes
    -----
    Extends ADK ``BaseAgentState`` for compatibility with the ADK session
    serialisation layer. Fields are persisted to SQLite via the ADK runtime.
    """

    workflow_step: RecWorkflowStep = RecWorkflowStep.GENERATE
    generation_output: dict | None = None
    feasibility_output: dict | None = None
    optimizer_output: dict | None = None
    report_output: SLORecommendationReport | None = None
    pending_questions: list[RecPlannerQuestion] = []
    user_answers: str = ""
