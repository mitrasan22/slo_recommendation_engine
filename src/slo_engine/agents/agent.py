"""
SLO Recommendation Engine root agent and router.

Notes
-----
Implements the RouterDecisionAgent (lightweight LLM for routing decisions),
the SLORouterAgent (orchestrates the three-stage pipeline), and the ADK App
and root_agent singletons for serving. HITL is implemented as an inline
confidence gate after recommendation — not as a separate agent. Low-confidence
recommendations are queued via submit_for_human_review() and approved via
the REST /api/v1/reviews/{id} endpoint.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator, Literal

from google.adk.agents import SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel, Field

from slo_engine.agents.base import BaseAgent
from slo_engine.agents.prompt import router_decision_prompt, router_prompt
from slo_engine.agents.dependency_agent.agent import dependency_agent
from slo_engine.agents.dependency_agent.schema import (
    DEP_ANALYSIS_OUTPUT_KEY,
    DEP_CYCLES_OUTPUT_KEY,
    DEP_REPORT_OUTPUT_KEY,
    DEP_WORKFLOW_STEP_KEY,
    DEP_USER_ANSWERS_KEY,
    DEP_PENDING_QUESTIONS_KEY,
    GRAPH_PAYLOAD_KEY,
    TARGET_SUBGRAPH_KEY,
    DependencyWorkflowStep,
)
from slo_engine.agents.metrics_agent.agent import metrics_agent
from slo_engine.agents.metrics_agent.schema import (
    METRICS_ANOMALY_KEY,
    METRICS_BUDGET_KEY,
    METRICS_REPORT_KEY,
    METRICS_SERVICE_KEY,
    METRICS_WINDOW_KEY,
    METRICS_WORKFLOW_KEY,
    METRICS_SLO_TARGET_KEY,
    METRICS_QUERY_KEY,
    METRICS_USER_ANSWERS_KEY,
    METRICS_PENDING_QUESTIONS_KEY,
    MetricsWorkflowStep,
)
from slo_engine.agents.recommendation_agent.agent import recommendation_agent
from slo_engine.agents.recommendation_agent.schema import (
    REC_REPORT_KEY,
    REC_SERVICE_KEY,
    REC_SERVICES_LIST_KEY,
    REC_WORKFLOW_KEY,
    REC_DEP_SLOS_KEY,
    REC_HIST_AVAIL_KEY,
    REC_WEIGHTS_KEY,
    REC_GENERATION_KEY,
    REC_KNOWLEDGE_KEY,
    REC_FEASIBILITY_KEY,
    REC_OPTIMIZER_KEY,
    REC_USER_ANSWERS_KEY,
    REC_PENDING_QUESTIONS_KEY,
    RecWorkflowStep,
)
from slo_engine.review_store import ReviewRequest, submit_for_human_review
from slo_engine.integrations.webhook_sink import push_slo_result_sync

logger = logger.bind(name=__name__)

FINAL_SLO_KEY = "final_slo"

RouteTarget = Literal[
    "dependency_agent",
    "metrics_agent",
    "recommendation_agent",
    "full_pipeline",
]


class RouterDecisionSchema(BaseModel):
    """
    Structured output schema for the router decision agent.

    Attributes
    ----------
    route_target : RouteTarget
        Which agent or pipeline to invoke. One of ``"dependency_agent"``,
        ``"metrics_agent"``, ``"recommendation_agent"``, or ``"full_pipeline"``.
    response_message : str
        Brief natural-language explanation of the routing decision.

    Notes
    -----
    Validated by the RouterDecisionAgent from the LLM structured output.
    Falls back to state-based inference if the LLM call fails.
    """

    route_target: RouteTarget = Field(
        ...,
        description=(
            "Which agent/pipeline to invoke: "
            "'dependency_agent' | 'metrics_agent' | "
            "'recommendation_agent' | 'full_pipeline'"
        ),
    )
    response_message: str = Field(..., description="Brief explanation of routing decision.")


class RouterDecisionAgent(BaseAgent):
    """
    Lightweight LLM agent that decides which sub-agent to invoke.

    Attributes
    ----------
    name : str
        ADK agent name ``"slo_router_decision_agent"``.
    description : str
        Brief description used in ADK agent discovery.
    instruction : str
        Prompt string imported from the prompts module.
    output_schema : type
        Pydantic model for structured output validation.
    output_key : str
        Session state key where the routing decision is stored.
    disallow_transfer_to_parent : bool
        Prevents the agent from transferring back to the parent.
    disallow_transfer_to_peers : bool
        Prevents the agent from transferring to sibling agents.

    Notes
    -----
    Uses the ``router_decision_prompt`` which instructs the LLM to output a
    ``RouterDecisionSchema``-compatible JSON object. Transfer is disabled to
    ensure the decision is always returned to the ``SLORouterAgent``.
    """

    name = "slo_router_decision_agent"
    description = "Decides which sub-agent to invoke based on current state and user request."
    instruction = router_decision_prompt
    output_schema = RouterDecisionSchema
    output_key = "router_decision"
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers = True


class SLORouterAgent(BaseAgent):
    """
    Top-level SLO Recommendation Engine router and orchestrator.

    Attributes
    ----------
    name : str
        ADK agent name ``"slo_router"``.
    description : str
        Brief description used in ADK agent discovery.
    orchestration : str
        Orchestration mode; ``"single"`` means the agent handles its own loop.
    instruction : str
        Prompt string imported from the prompts module.
    disallow_transfer_to_parent : bool
        Prevents the agent from transferring back to the parent.
    disallow_transfer_to_peers : bool
        Prevents the agent from transferring to sibling agents.

    Notes
    -----
    Orchestrates three sequential stages: dependency graph analysis, Bayesian
    metrics analysis, and SLO recommendation generation. After stage 3, an
    inline confidence gate auto-approves recommendations with confidence
    >= 0.90 and queues those below 0.75 or with drift detected for human review
    via the REST /api/v1/reviews/{id} endpoint.
    """

    name = "slo_router"
    description = (
        "SLO Recommendation Engine. Orchestrates dependency graph analysis, "
        "Bayesian metrics, and SLO generation."
    )
    orchestration = "single"
    instruction = router_prompt
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers = True

    def __init__(self, **overrides):
        """
        Initialise the SLORouterAgent and build sub-agent instances.

        Parameters
        ----------
        **overrides : dict
            Keyword arguments forwarded to the BaseAgent constructor.

        Notes
        -----
        Instantiates the router decision agent and stores references to the
        pre-built dependency, metrics, and recommendation agent singletons.
        """
        super().__init__(**overrides)
        self._decision_agent       = RouterDecisionAgent().build()
        self._dependency_agent     = dependency_agent
        self._metrics_agent        = metrics_agent
        self._recommendation_agent = recommendation_agent

    def _infer_route(self, state: dict) -> RouteTarget:
        """
        Determine the pipeline route from session state without LLM calls.

        Parameters
        ----------
        state : dict
            Current ADK session state dictionary.

        Returns
        -------
        RouteTarget
            Inferred route target based on which stages have already completed.

        Notes
        -----
        Used as a fallback when the LLM-based routing decision fails. Checks
        which output keys are already populated in state to skip completed stages.
        """
        if state.get(REC_REPORT_KEY):
            return "full_pipeline"
        if state.get(METRICS_REPORT_KEY):
            return "recommendation_agent"
        if state.get(DEP_REPORT_OUTPUT_KEY):
            return "metrics_agent"
        return "full_pipeline"

    def _gate_and_finalize(self, state: dict) -> dict:
        """
        Apply the inline confidence gate and produce the final SLO record.

        Parameters
        ----------
        state : dict
            Current ADK session state containing recommendation and metrics outputs.

        Returns
        -------
        dict
            Final SLO dictionary with service_name, availability, latency_p99_ms,
            error_rate, confidence_score, status, review_id, and approved flag.

        Notes
        -----
        Auto-approves recommendations when ``requires_human_review`` is False and
        no drift is detected. Otherwise, calls ``submit_for_human_review`` to queue
        the recommendation and sets status to ``"pending_human_review"``. Also
        pushes the result to any registered webhooks and logs to Opik.
        """
        rec_raw = state.get(REC_REPORT_KEY, "{}")
        rec = json.loads(rec_raw) if isinstance(rec_raw, str) else rec_raw

        met_raw = state.get(METRICS_REPORT_KEY, "{}")
        met = json.loads(met_raw) if isinstance(met_raw, str) else met_raw

        avail   = rec.get("recommended_availability", 0.99)
        lat     = rec.get("recommended_latency_p99_ms", 200.0)
        conf    = rec.get("confidence_score", 0.5)
        svc     = rec.get("service_name", state.get(REC_SERVICE_KEY, "unknown"))
        drift   = met.get("drift_detected", False)
        needs_review = rec.get("requires_human_review", False) or drift

        review_status = "auto_approved"
        review_id = None

        if needs_review:
            review_id = str(uuid.uuid4())
            submit_for_human_review(json.dumps({
                "recommendation_id": review_id,
                "service_name": svc,
                "recommended_availability": avail,
                "recommended_latency_p99_ms": lat,
                "confidence_score": conf,
                "review_reason": rec.get("review_reason", "Low confidence or drift detected."),
            }))
            review_status = "pending_human_review"
            logger.info(
                "Review queued for '{}': id={}, confidence={:.2f}, drift={}",
                svc, review_id, conf, drift,
            )
        else:
            logger.info("Auto-approved SLO for '{}': confidence={:.2f}", svc, conf)

        final = {
            "service_name": svc,
            "availability": avail,
            "latency_p99_ms": lat,
            "error_rate": round(1.0 - avail, 6),
            "confidence_score": conf,
            "status": review_status,
            "review_id": review_id,
            "approved": not needs_review,
        }
        try:
            push_slo_result_sync(svc, final)
        except Exception as e:
            logger.warning("Webhook push failed for '{}': {}", svc, e)

        from slo_engine.observability.opik_tracer import log_recommendation_audit
        try:
            log_recommendation_audit(
                service_name=svc,
                recommendation=final,
                sources=rec.get("sources", rec.get("data_sources", [])),
                decision=review_status,
            )
        except Exception:
            pass

        return final

    @staticmethod
    def _build_subgraph(graph: list[dict], target: str) -> list[dict]:
        """
        Extract the subgraph reachable from a target service via BFS.

        Parameters
        ----------
        graph : list of dict
            Full service graph as a list of service node dictionaries, each
            containing a ``service`` key and a ``depends_on`` list.
        target : str
            Name of the root service to start the BFS from.

        Returns
        -------
        list of dict
            List of service node dictionaries for the target and all of its
            transitive dependencies.

        Notes
        -----
        Used to scope the dependency planner questions to only the services
        relevant to the target, reducing noise for large graphs.
        """
        index   = {s.get("service", s.get("service_name", "")): s for s in graph if isinstance(s, dict)}
        visited: set[str] = set()
        queue   = [target]
        while queue:
            svc = queue.pop(0)
            if svc in visited or svc not in index:
                continue
            visited.add(svc)
            for dep in index[svc].get("depends_on", []):
                dep_name = dep.get("name", "")
                if dep_name and dep_name not in visited:
                    queue.append(dep_name)
        return [index[s] for s in visited if s in index]

    def _extract_service_from_message(self, ctx: InvocationContext) -> str | None:
        """
        Extract the target service name from the current user message.

        Parameters
        ----------
        ctx : InvocationContext
            ADK invocation context containing the current user message.

        Returns
        -------
        str or None
            Extracted service name in lowercase, or ``None`` if no service
            name could be identified.

        Notes
        -----
        Uses two regex patterns: ``"for <service-name>"`` and
        ``"<service-name> service"``. Only inspects the current-turn user
        message (``ctx.user_content``) to avoid matching previous turns.
        """
        import re
        try:
            if ctx.user_content and ctx.user_content.parts:
                text = " ".join(
                    p.text for p in ctx.user_content.parts
                    if hasattr(p, "text") and p.text
                )
                m = re.search(r'\bfor\s+([\w][\w\-\.]+)', text, re.IGNORECASE)
                if m:
                    return m.group(1).lower().rstrip(".")
                m = re.search(r'([\w][\w\-\.]+)\s+service\b', text, re.IGNORECASE)
                if m:
                    return m.group(1).lower().rstrip(".")
        except Exception:
            pass
        return None

    async def run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Orchestrate the three-stage SLO pipeline with inline confidence gate.

        Parameters
        ----------
        ctx : InvocationContext
            ADK invocation context with session state and user message.

        Returns
        -------
        AsyncGenerator[Event, None]
            Async generator yielding ADK events from each pipeline stage.

        Notes
        -----
        Pre-populates all optional state keys with defaults so ADK variable
        substitution never raises KeyError. Routes via LLM on the first turn
        and via state-based inference on resume turns. The three stages are
        dependency analysis, metrics analysis, and SLO recommendation. After
        stage 3, the inline confidence gate auto-approves or queues for review.
        Each stage pause on AWAIT_INPUT returns control to the user immediately.
        """
        state = ctx.session.state
        logger.info("SLORouterAgent: session={}", ctx.session.id)

        state.setdefault(DEP_USER_ANSWERS_KEY, "")
        state.setdefault(DEP_PENDING_QUESTIONS_KEY, "")
        state.setdefault(METRICS_USER_ANSWERS_KEY, "")
        state.setdefault(METRICS_PENDING_QUESTIONS_KEY, "")
        state.setdefault(METRICS_QUERY_KEY, "")
        state.setdefault(METRICS_WINDOW_KEY, 30)
        state.setdefault(METRICS_SLO_TARGET_KEY, 0.999)
        state.setdefault(METRICS_WORKFLOW_KEY, MetricsWorkflowStep.QUERY)
        state.setdefault(REC_WORKFLOW_KEY, RecWorkflowStep.GENERATE)
        state.setdefault(DEP_WORKFLOW_STEP_KEY, DependencyWorkflowStep.PLAN)
        state.setdefault(REC_USER_ANSWERS_KEY, "")
        state.setdefault(REC_PENDING_QUESTIONS_KEY, "")
        state.setdefault(REC_GENERATION_KEY, "")
        state.setdefault(REC_KNOWLEDGE_KEY, "")
        state.setdefault(REC_FEASIBILITY_KEY, "")
        state.setdefault(REC_OPTIMIZER_KEY, "")
        state.setdefault(REC_DEP_SLOS_KEY, "")
        state.setdefault(REC_HIST_AVAIL_KEY, "")
        state.setdefault(REC_WEIGHTS_KEY, "")
        state.setdefault(DEP_REPORT_OUTPUT_KEY, "")
        state.setdefault(DEP_ANALYSIS_OUTPUT_KEY, "")
        state.setdefault(DEP_CYCLES_OUTPUT_KEY, "")
        state.setdefault(TARGET_SUBGRAPH_KEY, "[]")
        state.setdefault(METRICS_REPORT_KEY, "")
        state.setdefault(METRICS_ANOMALY_KEY, "")
        state.setdefault(METRICS_BUDGET_KEY, "")

        _any_await = any(
            "AWAIT_INPUT" in str(state.get(k, ""))
            for k in (DEP_WORKFLOW_STEP_KEY, METRICS_WORKFLOW_KEY, REC_WORKFLOW_KEY)
        )
        _dep_in_progress = str(state.get(DEP_WORKFLOW_STEP_KEY, "")) not in ("", "PLAN", "DONE")

        if _any_await or _dep_in_progress:
            route = "full_pipeline"
            logger.info("Router -> full_pipeline (resuming in-progress pipeline, step={})",
                        state.get(DEP_WORKFLOW_STEP_KEY, "?"))
        else:
            try:
                async for event in self._decision_agent.run_async(ctx):
                    yield event
                raw = state.get("router_decision", "{}")
                decision = RouterDecisionSchema.model_validate(
                    json.loads(raw) if isinstance(raw, str) else raw
                )
                route = decision.route_target
                logger.info("Router -> {} ({})", route, decision.response_message)
            except Exception as exc:
                logger.warning("Router LLM failed ({}), inferring from state.", exc)
                route = self._infer_route(state)

        if route == "dependency_agent":
            state[DEP_WORKFLOW_STEP_KEY] = DependencyWorkflowStep.PLAN
            async for event in self._dependency_agent.run_async(ctx):
                yield event
            return

        if route == "metrics_agent":
            state[METRICS_WORKFLOW_KEY] = MetricsWorkflowStep.QUERY
            async for event in self._metrics_agent.run_async(ctx):
                yield event
            return

        if route == "recommendation_agent":
            if not state.get(REC_SERVICE_KEY):
                svc_from_msg = self._extract_service_from_message(ctx)
                if svc_from_msg:
                    state[REC_SERVICE_KEY] = svc_from_msg
                    state[METRICS_SERVICE_KEY] = svc_from_msg
            state[REC_WORKFLOW_KEY] = RecWorkflowStep.GENERATE
            async for event in self._recommendation_agent.run_async(ctx):
                yield event
            final = self._gate_and_finalize(state)
            state[FINAL_SLO_KEY] = json.dumps(final)
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={FINAL_SLO_KEY: json.dumps(final)}),
                content=genai_types.Content(parts=[genai_types.Part(
                    text=f"SLO for '{final['service_name']}': "
                         f"availability={final['availability']:.4f}, "
                         f"latency_p99={final['latency_p99_ms']:.0f}ms, "
                         f"status={final['status']}."
                )]),
            )
            return

        logger.info("Full pipeline: dependency -> metrics -> recommendation -> gate")

        graph_payload = state.get(GRAPH_PAYLOAD_KEY, [])
        if isinstance(graph_payload, str):
            try:
                graph_payload = json.loads(graph_payload)
            except Exception:
                graph_payload = []

        graph_services = [
            s.get("service", s.get("service_name", ""))
            for s in graph_payload if isinstance(s, dict)
        ]

        already_set = bool(state.get(METRICS_SERVICE_KEY))
        if not already_set:
            target_svc = self._extract_service_from_message(ctx)
            if not target_svc:
                target_svc = next(
                    (s.get("service", s.get("service_name", ""))
                     for s in graph_payload if isinstance(s, dict)),
                    "unknown",
                )
            if not graph_services:
                graph_services = [target_svc]

            target_subgraph = self._build_subgraph(graph_payload, target_svc)

            yield Event(
                author=self.name,
                actions=EventActions(state_delta={
                    METRICS_SERVICE_KEY:   target_svc,
                    REC_SERVICE_KEY:       target_svc,
                    REC_SERVICES_LIST_KEY: graph_services,
                    TARGET_SUBGRAPH_KEY:   json.dumps(target_subgraph),
                    METRICS_WINDOW_KEY:    30,
                    METRICS_SLO_TARGET_KEY: 0.999,
                }),
            )
            state[METRICS_SERVICE_KEY]    = target_svc
            state[REC_SERVICE_KEY]        = target_svc
            state[REC_SERVICES_LIST_KEY]  = graph_services
            state[TARGET_SUBGRAPH_KEY]    = json.dumps(target_subgraph)
            state[METRICS_WINDOW_KEY]     = 30
            state[METRICS_SLO_TARGET_KEY] = 0.999

            logger.info(
                "Pipeline target service: {} | subgraph {} services | full graph {} services",
                target_svc, len(target_subgraph), len(graph_services),
            )
        else:
            target_svc = state[METRICS_SERVICE_KEY]
            if not graph_services:
                graph_services = [target_svc]
            logger.info(
                "Pipeline resuming — target service preserved: {} | full graph {} services",
                target_svc, len(graph_services),
            )

        dep_step = DependencyWorkflowStep(
            state.get(DEP_WORKFLOW_STEP_KEY, DependencyWorkflowStep.PLAN)
        )
        if dep_step in (DependencyWorkflowStep.DONE,):
            logger.info("Stage 1 (dependency) already DONE — skipping.")
        else:
            if dep_step not in (DependencyWorkflowStep.AWAIT_INPUT,
                                DependencyWorkflowStep.INGEST,
                                DependencyWorkflowStep.CYCLES,
                                DependencyWorkflowStep.REPORT):
                state[DEP_WORKFLOW_STEP_KEY] = DependencyWorkflowStep.PLAN
            logger.info("Stage 1 — dependency analysis (step={})", state[DEP_WORKFLOW_STEP_KEY])
            async for event in self._dependency_agent.run_async(ctx):
                yield event

        if DependencyWorkflowStep(state.get(DEP_WORKFLOW_STEP_KEY, "PLAN")) == DependencyWorkflowStep.AWAIT_INPUT:
            logger.info("Stage 1 paused — AWAIT_INPUT. Pipeline will resume on next user turn.")
            return

        logger.info("Stage 1 done — waiting 10s before metrics stage.")
        await asyncio.sleep(10)

        logger.info("Stage 2 — metrics analysis for target service: {}", target_svc)
        state[METRICS_SERVICE_KEY]  = target_svc
        state[METRICS_WORKFLOW_KEY] = MetricsWorkflowStep.QUERY
        async for event in self._metrics_agent.run_async(ctx):
            yield event

        logger.info("Stage 2 done — waiting 10s before recommendation stage.")
        await asyncio.sleep(10)

        logger.info("Stage 3 — SLO recommendation for: {}", target_svc)
        state[REC_WORKFLOW_KEY] = RecWorkflowStep.GENERATE
        async for event in self._recommendation_agent.run_async(ctx):
            yield event

        final = self._gate_and_finalize(state)
        state[FINAL_SLO_KEY] = json.dumps(final)

        summary = (
            f"Pipeline complete for {len(graph_services)} service(s). "
            f"SLO: availability={final['availability']:.4f}, "
            f"latency_p99={final['latency_p99_ms']:.0f}ms, "
            f"status={final['status']}."
        )
        logger.info(summary)
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={FINAL_SLO_KEY: json.dumps(final)}),
            content=genai_types.Content(parts=[genai_types.Part(text=summary)]),
        )


def build_root_agent() -> SequentialAgent:
    """
    Build and return the ADK root agent wrapping the SLO router.

    Returns
    -------
    SequentialAgent
        ADK SequentialAgent with the SLORouterAgent as its only sub-agent.

    Notes
    -----
    The SequentialAgent is a thin wrapper required by the ADK runtime. The
    actual orchestration logic lives in SLORouterAgent.run_async_impl.
    """
    router = SLORouterAgent().build()
    return SequentialAgent(
        name="root_agent",
        description="Root entrypoint for the SLO Recommendation Engine.",
        sub_agents=[router],
    )


root_agent = build_root_agent()

app = App(
    name="slo_recommendation_engine",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
