"""
Metrics Analysis Agent — BaseAgent subclass pattern.

Notes
-----
Workflow: QUERY -> [AWAIT_INPUT] -> ANALYZE -> ANOMALY -> BUDGET -> REPORT.
Each step is a single agent that combines tool use and structured output. The
shared BaseAgent wiring handles Gemini-compatible response JSON schema setup, so
no extra orchestration wrapper is needed here.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator, Optional, Type, TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools.function_tool import FunctionTool
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel, ValidationError

from slo_engine.agents.base import BaseAgent
from slo_engine.agents.metrics_agent.prompt import (
    metrics_anomaly_tool_prompt,
    metrics_budget_tool_prompt,
    metrics_orchestrator_prompt,
    metrics_query_tool_prompt,
    metrics_report_prompt,
)
from slo_engine.agents.metrics_agent.schema import (
    METRICS_ANOMALY_KEY,
    METRICS_BUDGET_KEY,
    METRICS_PENDING_QUESTIONS_KEY,
    METRICS_QUERY_KEY,
    METRICS_REPORT_KEY,
    METRICS_SERVICE_KEY,
    METRICS_SLO_TARGET_KEY,
    METRICS_USER_ANSWERS_KEY,
    METRICS_WINDOW_KEY,
    METRICS_WORKFLOW_KEY,
    MetricsAnomalyAgentSchema,
    MetricsBudgetAgentSchema,
    MetricsPlannerQuestion,
    MetricsQueryAgentSchema,
    MetricsReportSchema,
    MetricsWorkflowStep,
)
from slo_engine.agents.metrics_agent.tools.tools import (
    compute_error_budget_status,
    detect_metric_anomaly,
    query_service_metrics,
)

logger = logger.bind(name=__name__)

T = TypeVar("T", bound=BaseModel)


def _parse_state(raw: str | dict | None, schema: Type[T]) -> Optional[T]:
    """
    Parse a raw state value into a Pydantic schema instance.

    Parameters
    ----------
    raw : str or dict or None
        Raw state value from ADK session state, either JSON string, dict, or None.
    schema : Type[T]
        Pydantic model class to validate against.

    Returns
    -------
    T or None
        Validated Pydantic model instance, or None if parsing fails or input is None.

    Notes
    -----
    Accepts both pre-parsed dicts and JSON strings. Returns None silently on
    ValidationError or ValueError to allow callers to apply fallback defaults.
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, dict):
            return schema.model_validate(raw)
        return schema.model_validate_json(raw)
    except (ValidationError, ValueError):
        return None


def _extract_user_message(ctx: InvocationContext) -> str:
    """
    Pull the latest user text from the invocation context.

    Parameters
    ----------
    ctx : InvocationContext
        ADK invocation context containing the current user content.

    Returns
    -------
    str
        Concatenated text from all text parts of the latest user message,
        or an empty string if no text content is present.

    Notes
    -----
    Iterates over ``ctx.user_content.parts`` and joins all parts that have a
    non-empty ``text`` attribute. Exceptions during extraction are silently
    swallowed and an empty string is returned.
    """
    try:
        if ctx.user_content and ctx.user_content.parts:
            return " ".join(
                p.text for p in ctx.user_content.parts if hasattr(p, "text") and p.text
            )
    except Exception:
        pass
    return ""


def _format_questions(questions: list[MetricsPlannerQuestion]) -> str:
    """
    Format a list of planner questions into human-readable markdown text.

    Parameters
    ----------
    questions : list of MetricsPlannerQuestion
        Structured question objects from the metrics planner.

    Returns
    -------
    str
        Markdown-formatted string with numbered questions and lettered options
        ready to display to the engineer.

    Notes
    -----
    Each question is prefixed with its ``question_id`` in brackets and options
    are labelled with letters starting from 'b'. A usage example is appended
    at the end to guide the engineer on how to answer.
    """
    lines = ["I need a clarification before proceeding with metrics analysis:\n"]
    for i, q in enumerate(questions, 1):
        opts = "\n".join(f"  {chr(96+j+1)}) {o}" for j, o in enumerate(q.options))
        lines.append(f"**Q{i}** [{q.question_id}]\n{q.question}\n{opts}")
    lines.append("\nPlease answer each question by question_id, e.g.:\n"
                 "  low_traffic_payment_api: (a) Proceed with low confidence")
    return "\n\n".join(lines)


def _log_questions(questions: list[MetricsPlannerQuestion], prefix: str) -> None:
    """
    Log each question with its full text and every option on its own line.

    Parameters
    ----------
    questions : list of MetricsPlannerQuestion
        Structured question objects to log.
    prefix : str
        Log line prefix string prepended to each log entry.

    Returns
    -------
    None

    Notes
    -----
    Uses loguru ``logger.info`` for each question and each option. Intended
    for observability during the AWAIT_INPUT pause in the metrics workflow.
    """
    for i, q in enumerate(questions, 1):
        logger.info("{} Q{} [{}]: {}", prefix, i, q.question_id, q.question)
        for j, opt in enumerate(q.options, 1):
            logger.info("{}   option {}: {}", prefix, j, opt)


class MetricsReportAgent(BaseAgent):
    """
    LLM synthesizer reading raw query/anomaly/budget from state.

    Attributes
    ----------
    name : str
        Agent identifier used by ADK routing.
    description : str
        Short description shown in agent cards.
    instruction : str
        Prompt template referencing state keys.
    output_schema : type
        Pydantic model class enforcing structured output.
    output_key : str
        ADK session state key where the report JSON is written.
    disallow_transfer_to_parent : bool
        Prevents ADK from transferring control back to parent agent.
    disallow_transfer_to_peers : bool
        Prevents ADK from transferring control to sibling agents.

    Notes
    -----
    Reads ``METRICS_QUERY_KEY``, ``METRICS_ANOMALY_KEY``, and
    ``METRICS_BUDGET_KEY`` from session state via the prompt template and
    produces a ``MetricsReportSchema`` JSON object stored in ``METRICS_REPORT_KEY``.
    """

    name          = "metrics_report_agent"
    description   = "Synthesizes metrics analysis results into a structured report."
    instruction   = metrics_report_prompt
    output_schema = MetricsReportSchema
    output_key    = METRICS_REPORT_KEY
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True


class _MetricsQueryAgent(BaseAgent):
    """
    Calls query_service_metrics and writes JSON result to state via output_key.

    Attributes
    ----------
    name : str
        Agent identifier used by ADK routing.
    description : str
        Short description of the agent's function.
    instruction : str
        Prompt template used to invoke the query tool.
    output_key : str
        ADK session state key where the query result is written.
    disallow_transfer_to_parent : bool
        Prevents ADK from transferring control back to parent agent.
    disallow_transfer_to_peers : bool
        Prevents ADK from transferring control to sibling agents.

    Notes
    -----
    Registered as a FunctionTool wrapper around ``query_service_metrics``.
    Computes Beta-Binomial posterior, Kalman-smoothed availability, Google SRE
    burn rates, and KL-divergence drift detection.
    """

    name          = "metrics_query_agent"
    description   = "Queries Bayesian posterior, Kalman-smoothed availability, burn rates, and KL drift."
    instruction   = metrics_query_tool_prompt
    output_key    = METRICS_QUERY_KEY
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True

    def get_tools(self):
        """
        Return the FunctionTool list for this agent.

        Returns
        -------
        list of FunctionTool
            Single-element list wrapping ``query_service_metrics``.

        Notes
        -----
        Called by ``BaseAgent.build()`` during agent construction to register
        tools with the ADK runner.
        """
        return [FunctionTool(query_service_metrics)]


def _build_metrics_query_agent():
    """
    Build and return the metrics query sub-agent.

    Returns
    -------
    google.adk.agents.base_agent.BaseAgent
        A built ADK agent instance ready for ``run_async`` calls.

    Notes
    -----
    Convenience factory used by ``MetricsOrchestratorAgent.__init__`` to
    construct the sub-agent once at initialisation time.
    """
    return _MetricsQueryAgent().build()


class _MetricsAnomalyAgent(BaseAgent):
    """
    Calls detect_metric_anomaly and writes JSON result to state via output_key.

    Attributes
    ----------
    name : str
        Agent identifier used by ADK routing.
    description : str
        Short description of the anomaly detection function.
    instruction : str
        Prompt template used to invoke the anomaly tool.
    output_key : str
        ADK session state key where the anomaly result is written.
    disallow_transfer_to_parent : bool
        Prevents ADK from transferring control back to parent agent.
    disallow_transfer_to_peers : bool
        Prevents ADK from transferring control to sibling agents.

    Notes
    -----
    Registered as a FunctionTool wrapper around ``detect_metric_anomaly``.
    Performs Z-score anomaly detection on availability and latency metrics.
    """

    name          = "metrics_anomaly_agent"
    description   = "Z-score anomaly detection for availability and latency drops."
    instruction   = metrics_anomaly_tool_prompt
    output_key    = METRICS_ANOMALY_KEY
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True

    def get_tools(self):
        """
        Return the FunctionTool list for this agent.

        Returns
        -------
        list of FunctionTool
            Single-element list wrapping ``detect_metric_anomaly``.

        Notes
        -----
        Called by ``BaseAgent.build()`` during agent construction to register
        tools with the ADK runner.
        """
        return [FunctionTool(detect_metric_anomaly)]


def _build_metrics_anomaly_agent():
    """
    Build and return the metrics anomaly detection sub-agent.

    Returns
    -------
    google.adk.agents.base_agent.BaseAgent
        A built ADK agent instance ready for ``run_async`` calls.

    Notes
    -----
    Convenience factory used by ``MetricsOrchestratorAgent.__init__`` to
    construct the sub-agent once at initialisation time.
    """
    return _MetricsAnomalyAgent().build()


class _MetricsBudgetAgent(BaseAgent):
    """
    Calls compute_error_budget_status and writes JSON result to state via output_key.

    Attributes
    ----------
    name : str
        Agent identifier used by ADK routing.
    description : str
        Short description of the budget computation function.
    instruction : str
        Prompt template used to invoke the budget tool.
    output_key : str
        ADK session state key where the budget result is written.
    disallow_transfer_to_parent : bool
        Prevents ADK from transferring control back to parent agent.
    disallow_transfer_to_peers : bool
        Prevents ADK from transferring control to sibling agents.

    Notes
    -----
    Registered as a FunctionTool wrapper around ``compute_error_budget_status``.
    Computes error budget health percentage and Gaussian exhaustion forecast.
    """

    name          = "metrics_budget_agent"
    description   = "Computes error budget health and Gaussian exhaustion forecast."
    instruction   = metrics_budget_tool_prompt
    output_key    = METRICS_BUDGET_KEY
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True

    def get_tools(self):
        """
        Return the FunctionTool list for this agent.

        Returns
        -------
        list of FunctionTool
            Single-element list wrapping ``compute_error_budget_status``.

        Notes
        -----
        Called by ``BaseAgent.build()`` during agent construction to register
        tools with the ADK runner.
        """
        return [FunctionTool(compute_error_budget_status)]


def _build_metrics_budget_agent():
    """
    Build and return the metrics error budget sub-agent.

    Returns
    -------
    google.adk.agents.base_agent.BaseAgent
        A built ADK agent instance ready for ``run_async`` calls.

    Notes
    -----
    Convenience factory used by ``MetricsOrchestratorAgent.__init__`` to
    construct the sub-agent once at initialisation time.
    """
    return _MetricsBudgetAgent().build()


class MetricsOrchestratorAgent(BaseAgent):
    """
    Top-level metrics agent driving the full analysis workflow.

    Attributes
    ----------
    name : str
        Agent identifier used by ADK routing.
    description : str
        Human-readable description of the agent's capabilities.
    orchestration : str
        ADK orchestration mode; ``"single"`` for direct control flow.
    instruction : str
        Prompt template for the orchestrator's own LLM calls.
    disallow_transfer_to_parent : bool
        Prevents ADK from transferring control back to parent agent.
    disallow_transfer_to_peers : bool
        Prevents ADK from transferring control to sibling agents.

    Notes
    -----
    Drives QUERY -> [AWAIT_INPUT] -> ANALYZE -> ANOMALY -> BUDGET -> REPORT.
    Tools are called directly (bypassing the LLM wrapper) because the LLM
    wraps raw results as ``{"query_service_metrics_response": ...}`` which
    does not match the target Pydantic schemas. The REPORT step uses an LLM
    synthesizer sub-agent with ``output_schema=MetricsReportSchema``.
    If the QUERY step returns fewer than 100 requests the pipeline pauses,
    yields a clarification question to the engineer, and returns. On the
    next session turn the engineer's answer is read from state and the
    pipeline advances from the ANALYZE step.
    """

    name = "metrics_agent"
    description = (
        "Bayesian metrics analysis: Beta-Binomial posterior, Kalman smoothing, "
        "KL drift detection, Google SRE burn rates, error budget forecasting. "
        "Asks the engineer for guidance when traffic volume is insufficient."
    )
    orchestration = "single"
    instruction = metrics_orchestrator_prompt
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers = True

    def __init__(self, **overrides):
        """
        Initialise the orchestrator and build all sub-agents.

        Parameters
        ----------
        **overrides : dict
            Keyword arguments forwarded to ``BaseAgent.__init__``.

        Returns
        -------
        None

        Notes
        -----
        Sub-agents are built once at construction time and reused across
        invocations. ``_report_agent`` is built directly from
        ``MetricsReportAgent`` while the three tool agents use factory functions.
        """
        super().__init__(**overrides)
        self._query_agent   = _build_metrics_query_agent()
        self._anomaly_agent = _build_metrics_anomaly_agent()
        self._budget_agent  = _build_metrics_budget_agent()
        self._report_agent  = MetricsReportAgent().build()

    async def run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the metrics analysis workflow step by step.

        Parameters
        ----------
        ctx : InvocationContext
            ADK invocation context providing session state and user content.

        Returns
        -------
        AsyncGenerator of Event
            Yields ADK events including the final structured report event and
            an optional clarification-question event when NEEDS_INPUT is triggered.

        Notes
        -----
        Each workflow step (QUERY, AWAIT_INPUT, ANALYZE, ANOMALY, BUDGET, REPORT)
        reads from and writes to ``ctx.session.state`` using the schema key
        constants defined in ``metrics_agent.schema``. The step cursor is stored
        in ``METRICS_WORKFLOW_KEY`` so that the pipeline can resume from the
        correct step on subsequent session turns. The REPORT step delegates to
        the ``MetricsReportAgent`` LLM synthesizer via ``run_async``.
        """
        state = ctx.session.state
        step          = MetricsWorkflowStep(state.get(METRICS_WORKFLOW_KEY, MetricsWorkflowStep.QUERY))
        original_step = step
        logger.info("MetricsOrchestratorAgent: step={}", step)

        if step == MetricsWorkflowStep.QUERY:
            svc    = state.get(METRICS_SERVICE_KEY, "?")
            window = int(state.get(METRICS_WINDOW_KEY, 30))
            target = float(state.get(METRICS_SLO_TARGET_KEY, 0.999))
            logger.info(
                "METRICS QUERY | calling query_service_metrics directly: service={} window_days={} slo_target={}",
                svc, window, target,
            )
            query_result_raw = query_service_metrics(
                service_name=svc,
                window_days=window,
                include_percentiles=True,
                include_burn_rates=True,
            )
            logger.info("METRICS QUERY | raw result: {}", query_result_raw[:500])

            try:
                q_data = json.loads(query_result_raw)
                request_count = q_data.get("request_count_total", 0)
                decision = "READY" if request_count >= 100 else "NEEDS_INPUT"
                query_schema_data = {
                    "decision": decision,
                    "questions_for_user": [],
                    "service_name":          q_data.get("service_name", svc),
                    "window_days":           q_data.get("window_days", window),
                    "posterior_mean":        q_data.get("posterior_mean", 0.99),
                    "credible_lower_95":     q_data.get("credible_lower_95", 0.98),
                    "credible_upper_95":     q_data.get("credible_upper_95", 1.0),
                    "smoothed_availability": q_data.get("smoothed_availability", 0.99),
                    "burn_rate_1h":          q_data.get("burn_rate_1h", 0.0),
                    "burn_rate_6h":          q_data.get("burn_rate_6h", 0.0),
                    "drift_detected":        q_data.get("drift_detected", False),
                    "kl_divergence":         q_data.get("kl_divergence", 0.0),
                    "request_count_total":   request_count,
                    "posterior_std":         q_data.get("std_availability", 0.01),
                    "smoothed_p99_ms":       q_data.get("smoothed_p99_ms", 200.0),
                    "p99_latency_ms":        q_data.get("p99_latency_ms", 200.0),
                    "data_quality_score":    q_data.get("data_quality_score", 1.0),
                    "notes": "",
                }
                state[METRICS_QUERY_KEY] = json.dumps(query_schema_data)
                query = MetricsQueryAgentSchema.model_validate(query_schema_data)
                logger.info(
                    "METRICS QUERY | posterior_mean={:.6f} CI95=[{:.6f}, {:.6f}] "
                    "smoothed={:.6f} burn_rate_1h={:.4f} drift_detected={} "
                    "kl_divergence={:.4f} request_count={} decision={}",
                    query.posterior_mean, query.credible_lower_95, query.credible_upper_95,
                    query.smoothed_availability, query.burn_rate_1h,
                    query.drift_detected, query.kl_divergence,
                    query.request_count_total, query.decision,
                )
            except Exception as e:
                logger.warning("METRICS QUERY | parse failed: {}. raw={}", e, query_result_raw[:300])
                query = None

            if query and query.decision == "NEEDS_INPUT" and query.questions_for_user:
                questions_text = _format_questions(query.questions_for_user)
                state[METRICS_PENDING_QUESTIONS_KEY] = json.dumps(
                    [q.model_dump() for q in query.questions_for_user]
                )
                state[METRICS_WORKFLOW_KEY] = MetricsWorkflowStep.AWAIT_INPUT
                logger.info("METRICS QUERY | NEEDS_INPUT — {} question(s):", len(query.questions_for_user))
                _log_questions(query.questions_for_user, "METRICS QUERY |")
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={
                        METRICS_WORKFLOW_KEY:          MetricsWorkflowStep.AWAIT_INPUT,
                        METRICS_PENDING_QUESTIONS_KEY: state[METRICS_PENDING_QUESTIONS_KEY],
                    }),
                    content=genai_types.Content(
                        parts=[genai_types.Part(text=questions_text)]
                    ),
                )
                return

            logger.info("METRICS QUERY | decision=READY — proceeding to ANALYZE")
            state[METRICS_WORKFLOW_KEY] = MetricsWorkflowStep.ANALYZE
            step = MetricsWorkflowStep.ANALYZE

        if step == MetricsWorkflowStep.AWAIT_INPUT:
            answers = _extract_user_message(ctx)
            logger.info("METRICS AWAIT_INPUT | user answers: {}", answers[:200] if answers else "(none yet)")
            if not answers:
                yield Event(
                    author=self.name,
                    content=genai_types.Content(
                        parts=[genai_types.Part(text="Still waiting for your answer to the clarification question.")]
                    ),
                )
                return

            state[METRICS_USER_ANSWERS_KEY]    = answers
            state[METRICS_WORKFLOW_KEY]        = MetricsWorkflowStep.ANALYZE
            state[METRICS_PENDING_QUESTIONS_KEY] = ""
            logger.info("METRICS AWAIT_INPUT | answers stored, advancing to ANALYZE")

        if step <= MetricsWorkflowStep.ANALYZE:
            if original_step == MetricsWorkflowStep.ANALYZE:
                logger.info("METRICS ANALYZE | re-running query after user answers")
                svc    = state.get(METRICS_SERVICE_KEY, "?")
                window = int(state.get(METRICS_WINDOW_KEY, 30))
                query_result_raw = query_service_metrics(
                    service_name=svc, window_days=window,
                    include_percentiles=True, include_burn_rates=True,
                )
                try:
                    q_data = json.loads(query_result_raw)
                    request_count = q_data.get("request_count_total", 0)
                    query_schema_data = {
                        "decision": "READY" if request_count >= 100 else "NEEDS_INPUT",
                        "questions_for_user": [],
                        "service_name":          q_data.get("service_name", svc),
                        "window_days":           q_data.get("window_days", window),
                        "posterior_mean":        q_data.get("posterior_mean", 0.99),
                        "credible_lower_95":     q_data.get("credible_lower_95", 0.98),
                        "credible_upper_95":     q_data.get("credible_upper_95", 1.0),
                        "smoothed_availability": q_data.get("smoothed_availability", 0.99),
                        "burn_rate_1h":          q_data.get("burn_rate_1h", 0.0),
                        "burn_rate_6h":          q_data.get("burn_rate_6h", 0.0),
                        "drift_detected":        q_data.get("drift_detected", False),
                        "kl_divergence":         q_data.get("kl_divergence", 0.0),
                        "request_count_total":   request_count,
                        "notes": "",
                    }
                    state[METRICS_QUERY_KEY] = json.dumps(query_schema_data)
                except Exception as e:
                    logger.warning("METRICS ANALYZE | re-query parse failed: {}", e)
            state[METRICS_WORKFLOW_KEY] = MetricsWorkflowStep.ANOMALY
            step = MetricsWorkflowStep.ANOMALY

        if step <= MetricsWorkflowStep.ANOMALY:
            q_state = _parse_state(state.get(METRICS_QUERY_KEY), MetricsQueryAgentSchema)
            cur_avail  = q_state.posterior_mean        if q_state else 0.99
            base_avail = q_state.smoothed_availability if q_state else 0.99
            logger.info(
                "METRICS ANOMALY | calling detect_metric_anomaly directly: current_avail={:.6f} baseline_avail={:.6f}",
                cur_avail, base_avail,
            )
            anomaly_result_raw = detect_metric_anomaly(
                service_name=state.get(METRICS_SERVICE_KEY, ""),
                current_availability=cur_avail,
                baseline_availability=base_avail,
                current_p99_ms=200.0,
                baseline_p99_ms=200.0,
            )
            state[METRICS_ANOMALY_KEY] = anomaly_result_raw
            logger.info("METRICS ANOMALY | raw result: {}", anomaly_result_raw)
            anomaly_out = _parse_state(anomaly_result_raw, MetricsAnomalyAgentSchema)
            if anomaly_out:
                logger.info(
                    "METRICS ANOMALY | is_anomaly={} severity={} avail_z={:.4f} latency_z={:.4f}",
                    anomaly_out.is_anomaly, anomaly_out.severity,
                    anomaly_out.availability_z_score, anomaly_out.latency_z_score,
                )
            state[METRICS_WORKFLOW_KEY] = MetricsWorkflowStep.BUDGET
            step = MetricsWorkflowStep.BUDGET

        if step <= MetricsWorkflowStep.BUDGET:
            svc    = state.get(METRICS_SERVICE_KEY, "?")
            target = float(state.get(METRICS_SLO_TARGET_KEY, 0.999))
            window = int(state.get(METRICS_WINDOW_KEY, 30))
            logger.info(
                "METRICS BUDGET | calling compute_error_budget_status directly: service={} slo_target={} window_days={}",
                svc, target, window,
            )
            budget_result_raw = compute_error_budget_status(
                service_name=svc,
                slo_target=target,
                window_days=window,
            )
            state[METRICS_BUDGET_KEY] = budget_result_raw
            logger.info("METRICS BUDGET | raw result: {}", budget_result_raw)
            budget_out = _parse_state(budget_result_raw, MetricsBudgetAgentSchema)
            if budget_out:
                logger.info(
                    "METRICS BUDGET | burn_fraction={:.4f} days_to_exhaustion={:.1f} prob_exhaust={:.4f} status={}",
                    budget_out.burn_fraction, budget_out.days_to_exhaustion,
                    budget_out.prob_exhaust_in_window, budget_out.status,
                )
            state[METRICS_WORKFLOW_KEY] = MetricsWorkflowStep.REPORT

        logger.info("METRICS REPORT | running LLM report synthesizer")
        async for event in self._report_agent.run_async(ctx):
            yield event

        state[METRICS_WORKFLOW_KEY] = MetricsWorkflowStep.DONE
        report_raw = state.get(METRICS_REPORT_KEY, "{}")
        try:
            r = json.loads(report_raw) if isinstance(report_raw, str) else report_raw
            logger.info(
                "METRICS REPORT | service={} posterior_mean={} drift={} "
                "anomaly={} burn_fraction={} requires_review={}",
                r.get("service_name", "?"), r.get("posterior_mean", "?"),
                r.get("drift_detected", "?"), r.get("anomaly_severity", "?"),
                r.get("burn_fraction", "?"), r.get("requires_human_review", "?"),
            )
        except Exception:
            logger.warning("METRICS REPORT | could not parse report for logging")


metrics_agent = MetricsOrchestratorAgent().build()

root_agent = metrics_agent
