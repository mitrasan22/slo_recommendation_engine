"""
Schemas and state keys for the metrics analysis agent workflow.

Notes
-----
Defines all ADK session state key constants and Pydantic output schemas used by
the metrics query, anomaly detection, error budget, report synthesizer, and
orchestrator agents.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from google.adk.agents.base_agent import BaseAgentState
from pydantic import BaseModel, Field

METRICS_SERVICE_KEY             = "metrics_service_name"
METRICS_WINDOW_KEY              = "metrics_window_days"
METRICS_SLO_TARGET_KEY          = "metrics_slo_target"
METRICS_WORKFLOW_KEY            = "metrics_workflow_step"
METRICS_QUERY_KEY               = "metrics_query_output"
METRICS_ANOMALY_KEY             = "metrics_anomaly_output"
METRICS_BUDGET_KEY              = "metrics_budget_output"
METRICS_REPORT_KEY              = "metrics_report_output"
METRICS_PENDING_QUESTIONS_KEY   = "metrics_pending_questions"
METRICS_USER_ANSWERS_KEY        = "metrics_user_answers"


class MetricsWorkflowStep(StrEnum):
    """
    Enumeration of metrics analysis workflow steps.

    Attributes
    ----------
    QUERY : str
        Initial Bayesian metrics query step.
    AWAIT_INPUT : str
        Paused state waiting for human clarification when traffic is insufficient.
    ANALYZE : str
        Resume step after human answers have been received.
    ANOMALY : str
        Z-score anomaly detection step.
    BUDGET : str
        Error budget health computation step.
    REPORT : str
        Final LLM report synthesis step.
    DONE : str
        Terminal state after the report has been written.

    Notes
    -----
    Steps are ordered and comparable using StrEnum. The orchestrator uses
    ``<=`` comparisons to skip already-completed steps on resume.
    """

    QUERY       = "QUERY"
    AWAIT_INPUT = "AWAIT_INPUT"
    ANALYZE     = "ANALYZE"
    ANOMALY     = "ANOMALY"
    BUDGET      = "BUDGET"
    REPORT      = "REPORT"
    DONE        = "DONE"


class MetricsPlannerQuestion(BaseModel):
    """
    A structured question from the metrics planner for human clarification.

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
    Serialised to JSON and stored in ``metrics_pending_questions`` state key.
    The engineer's answers are stored in ``metrics_user_answers`` on the next turn.
    """

    question_id: str   = Field(..., description="Stable identifier — used to match answers.")
    question: str      = Field(..., description="Question text shown to the user.")
    options: list[str] = Field(..., description="Suggested choices — user may pick one or answer freely.")


class MetricsQueryAgentSchema(BaseModel):
    """
    Structured output schema for the metrics query agent.

    Attributes
    ----------
    decision : Literal["READY", "NEEDS_INPUT"]
        Whether there is sufficient data for Bayesian estimation.
    questions_for_user : list of MetricsPlannerQuestion
        Non-empty only when ``decision="NEEDS_INPUT"``.
    service_name : str
        Name of the queried service.
    window_days : int
        Observation window in days.
    posterior_mean : float
        Bayesian Beta-Binomial posterior mean availability estimate.
    credible_lower_95 : float
        Lower bound of the 95% credible interval.
    credible_upper_95 : float
        Upper bound of the 95% credible interval.
    smoothed_availability : float
        Kalman-filtered availability estimate.
    burn_rate_1h : float
        Google SRE 1-hour error budget burn rate.
    burn_rate_6h : float
        Google SRE 6-hour error budget burn rate.
    drift_detected : bool
        Whether KL-divergence drift has been detected.
    kl_divergence : float
        KL-divergence score measuring distribution shift.
    request_count_total : int
        Total number of requests in the observation window.
    notes : str
        Optional free-text notes from the query agent.

    Notes
    -----
    Stored in state under ``METRICS_QUERY_KEY`` after the QUERY step completes.
    When ``request_count_total < 100`` the decision is ``"NEEDS_INPUT"`` and
    the pipeline pauses at ``AWAIT_INPUT``.
    """

    decision: Literal["READY", "NEEDS_INPUT"] = Field(
        ...,
        description=(
            "READY — sufficient data for Bayesian estimation. "
            "NEEDS_INPUT — request_count_total < 100; questions_for_user is populated."
        ),
    )
    questions_for_user: list[MetricsPlannerQuestion] = Field(
        default_factory=list,
        description="Non-empty only when decision=NEEDS_INPUT.",
    )
    service_name: str
    window_days: int
    posterior_mean: float = Field(..., description="Bayesian posterior mean availability.")
    credible_lower_95: float
    credible_upper_95: float
    smoothed_availability: float = Field(..., description="Kalman-filtered availability.")
    burn_rate_1h: float
    burn_rate_6h: float
    drift_detected: bool
    kl_divergence: float
    request_count_total: int
    notes: str = ""


class MetricsAnomalyAgentSchema(BaseModel):
    """
    Structured output schema for the anomaly detection agent.

    Attributes
    ----------
    is_anomaly : bool
        Whether an anomaly was detected.
    severity : Literal
        Severity level: ``"none"``, ``"low"``, ``"medium"``, ``"high"``, or ``"critical"``.
    availability_z_score : float
        Z-score of the current availability relative to the baseline.
    latency_z_score : float
        Z-score of the current p99 latency relative to the baseline.
    requires_investigation : bool
        Whether the anomaly warrants immediate engineer investigation.
    message : str
        Human-readable description of the anomaly finding.

    Notes
    -----
    Stored in state under ``METRICS_ANOMALY_KEY`` after the ANOMALY step.
    Z-score thresholds: >=5 critical, >=4 high, >=3 medium, >=2 low, else none.
    """

    is_anomaly: bool
    severity: Literal["none", "low", "medium", "high", "critical"]
    availability_z_score: float
    latency_z_score: float
    requires_investigation: bool
    message: str = ""


class MetricsBudgetAgentSchema(BaseModel):
    """
    Structured output schema for the error budget agent.

    Attributes
    ----------
    burn_fraction : float
        Fraction of the error budget consumed in the window (0 to 1).
    burn_rate_per_day : float
        Average error budget consumption per day.
    days_to_exhaustion : float
        Estimated days until the error budget is fully consumed at current burn rate.
    prob_exhaust_in_window : float
        Gaussian probability of full budget exhaustion within the window.
    status : Literal
        Budget health status: ``"healthy"``, ``"warning"``, or ``"critical"``.

    Notes
    -----
    Stored in state under ``METRICS_BUDGET_KEY`` after the BUDGET step.
    ``prob_exhaust_in_window`` is derived from Gaussian approximation of the
    burn-rate distribution.
    """

    burn_fraction: float = Field(..., description="Fraction of error budget consumed (0-1).")
    burn_rate_per_day: float
    days_to_exhaustion: float
    prob_exhaust_in_window: float
    status: Literal["healthy", "warning", "critical"]


class MetricsReportSchema(BaseModel):
    """
    Structured output schema for the final metrics analysis report.

    Attributes
    ----------
    service_name : str
        Name of the analysed service.
    window_days : int
        Observation window in days.
    posterior_mean : float
        Bayesian posterior mean availability.
    credible_lower_95 : float
        Lower bound of the 95% credible interval.
    smoothed_availability : float
        Kalman-filtered availability.
    burn_rate_1h : float
        1-hour SRE burn rate.
    drift_detected : bool
        Whether KL-divergence drift was detected.
    kl_divergence : float
        KL-divergence score.
    is_anomaly : bool
        Whether an availability or latency anomaly was detected.
    anomaly_severity : str
        Severity string from anomaly detection.
    burn_fraction : float
        Fraction of error budget consumed.
    prob_exhaust_in_window : float
        Probability of budget exhaustion in the window.
    budget_status : str
        Error budget health status string.
    requires_human_review : bool
        Whether engineer review is required based on combined signals.
    summary : str
        Concise engineer-facing summary of all three analysis signals.

    Notes
    -----
    Stored in state under ``METRICS_REPORT_KEY`` after the REPORT step.
    ``requires_human_review`` is set to True when drift is detected, anomaly
    severity is high or critical, or burn_fraction exceeds 0.9.
    """

    service_name: str
    window_days: int
    posterior_mean: float
    credible_lower_95: float
    smoothed_availability: float
    burn_rate_1h: float
    drift_detected: bool
    kl_divergence: float
    is_anomaly: bool
    anomaly_severity: str
    burn_fraction: float
    prob_exhaust_in_window: float
    budget_status: str
    requires_human_review: bool
    summary: str


class MetricsOrchestratorState(BaseAgentState):
    """
    Typed ADK session state for the metrics analysis orchestrator.

    Attributes
    ----------
    workflow_step : MetricsWorkflowStep
        Current step in the metrics analysis workflow.
    query_output : MetricsQueryAgentSchema or None
        Output from the QUERY step, if completed.
    anomaly_output : MetricsAnomalyAgentSchema or None
        Output from the ANOMALY step, if completed.
    budget_output : MetricsBudgetAgentSchema or None
        Output from the BUDGET step, if completed.
    report_output : MetricsReportSchema or None
        Output from the REPORT step, if completed.
    pending_questions : list of MetricsPlannerQuestion
        Questions awaiting human answers when in AWAIT_INPUT state.
    user_answers : str
        Raw text of the engineer's answers to the planner's questions.

    Notes
    -----
    Extends ADK ``BaseAgentState`` for compatibility with the ADK session
    serialisation layer. Fields are persisted to SQLite via the ADK runtime.
    """

    workflow_step: MetricsWorkflowStep = MetricsWorkflowStep.QUERY
    query_output: MetricsQueryAgentSchema | None = None
    anomaly_output: MetricsAnomalyAgentSchema | None = None
    budget_output: MetricsBudgetAgentSchema | None = None
    report_output: MetricsReportSchema | None = None
    pending_questions: list[MetricsPlannerQuestion] = []
    user_answers: str = ""
