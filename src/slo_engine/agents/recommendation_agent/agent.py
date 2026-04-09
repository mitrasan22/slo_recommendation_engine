"""
SLO Recommendation Agent — BaseAgent subclass pattern.

Notes
-----
Workflow: GENERATE -> FEASIBILITY -> OPTIMIZE -> REPORT.
All math tools are called directly in Python (same pattern as dependency and
metrics agents). The LLM is used only for the final SLOReportAgent synthesizer,
which returns prompt-constrained JSON that is parsed and logged by the
orchestrator.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from loguru import logger

from slo_engine.agents.base import BaseAgent
from slo_engine.agents.dependency_agent.schema import DEP_REPORT_OUTPUT_KEY, GRAPH_PAYLOAD_KEY
from slo_engine.agents.metrics_agent.schema import METRICS_QUERY_KEY, METRICS_REPORT_KEY
from slo_engine.agents.recommendation_agent.prompt import slo_report_prompt
from slo_engine.agents.recommendation_agent.schema import (
    REC_DEP_SLOS_KEY,
    REC_FEASIBILITY_KEY,
    REC_GENERATION_KEY,
    REC_HIST_AVAIL_KEY,
    REC_KNOWLEDGE_KEY,
    REC_KNOWLEDGE_QUERY_KEY,
    REC_OPTIMIZER_KEY,
    REC_PENDING_QUESTIONS_KEY,
    REC_REPORT_KEY,
    REC_SERVICE_KEY,
    REC_SERVICES_LIST_KEY,
    REC_USER_ANSWERS_KEY,
    REC_WEIGHTS_KEY,
    REC_WORKFLOW_KEY,
    RecPlannerQuestion,
    RecWorkflowStep,
    SLORecommendationReport,
)
from slo_engine.agents.recommendation_agent.tools.tools import (
    check_slo_feasibility,
    generate_slo_recommendation,
    run_milp_optimization,
)
from slo_engine.mcp.client import knowledge_client

logger = logger.bind(name=__name__)

def _infer_service_type(service_name: str) -> str:
    """
    Infer the service type category from a service name string.

    Parameters
    ----------
    service_name : str
        Name of the service, e.g. ``"checkout-service"`` or ``"auth-api"``.

    Returns
    -------
    str
        Service type string used by the knowledge retrieval tool. One of
        ``"checkout"``, ``"payment"``, ``"auth"``, ``"database"``,
        ``"api_gateway"``, or ``"generic"``.

    Notes
    -----
    Simple keyword matching on the lowercased service name. Unknown services
    fall back to ``"generic"`` which applies the most conservative SLO defaults.
    """
    name = service_name.lower()
    if "order" in name or "checkout" in name:
        return "checkout"
    if "payment" in name:
        return "payment"
    if "auth" in name:
        return "auth"
    if "db" in name or "database" in name:
        return "database"
    if "gateway" in name or "api" in name:
        return "api_gateway"
    if "fraud" in name or "risk" in name:
        return "generic"
    if "inventory" in name or "warehouse" in name:
        return "generic"
    return "generic"


def _load_json(raw) -> dict:
    """
    Parse a state value to dict, handling str, dict, or None inputs.

    Parameters
    ----------
    raw : str or dict or None
        Raw value from ADK session state.

    Returns
    -------
    dict
        Parsed dictionary, or an empty dict if parsing fails or input is None.

    Notes
    -----
    Used throughout the orchestrator to safely read state values that may be
    stored as JSON strings or already-deserialized dicts.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        text = str(raw).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        return {}


class SLOReportAgent(BaseAgent):
    """
    LLM synthesizer producing the final structured SLO recommendation report.

    Attributes
    ----------
    name : str
        Agent identifier used by ADK routing.
    description : str
        Short description shown in agent cards.
    instruction : str
        Prompt template referencing state keys for all upstream tool results.
    output_key : str
        ADK session state key where the report JSON is written.
    disallow_transfer_to_parent : bool
        Prevents ADK from transferring control back to parent agent.
    disallow_transfer_to_peers : bool
        Prevents ADK from transferring control to sibling agents.

    Notes
    -----
    Reads generation, feasibility, optimizer, and knowledge results from
    session state via the prompt template and produces a
    ``SLORecommendationReport`` JSON object stored in ``REC_REPORT_KEY``.
    """

    name          = "slo_report_agent"
    description   = "Synthesizes SLO recommendation results into a structured final report."
    instruction   = slo_report_prompt
    output_key    = REC_REPORT_KEY
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True


class RecommendationOrchestratorAgent(BaseAgent):
    """
    Top-level recommendation agent driving the full SLO recommendation workflow.

    Attributes
    ----------
    name : str
        Agent identifier used by ADK routing.
    description : str
        Human-readable description of the agent's capabilities.
    orchestration : str
        ADK orchestration mode; ``"single"`` for direct control flow.
    instruction : str
        Prompt template for the orchestrator showing the current workflow step.
    disallow_transfer_to_parent : bool
        Prevents ADK from transferring control back to parent agent.
    disallow_transfer_to_peers : bool
        Prevents ADK from transferring control to sibling agents.

    Notes
    -----
    Drives GENERATE -> [AWAIT_INPUT] -> FEASIBILITY -> OPTIMIZE -> REPORT.
    GENERATE checks if the service is in the dependency graph; if not it pauses
    at AWAIT_INPUT and asks the engineer for tier and external-dep info.
    On resume the orchestrator reads ``rec_user_answers`` and continues.
    Knowledge retrieval uses the shared MCP client from ``slo_engine.mcp.client``
    and calls the ``retrieve_knowledge`` MCP tool directly. This preserves the
    MCP integration while making the retrieval step deterministic and ensuring
    the parsed ``source_ids`` and ``context_summary`` are available immediately.
    All other math (feasibility, MILP) is called directly in Python.
    REPORT delegates to the ``SLOReportAgent`` LLM synthesizer.
    """

    name = "recommendation_agent"
    description = (
        "Full SLO recommendation pipeline: Bayesian posterior, series/parallel reliability, "
        "Monte Carlo validation, CLT latency, LP feasibility, Pareto MILP optimization."
    )
    orchestration = "single"
    instruction   = "You are the SLO Recommendation Orchestrator. Current step: {rec_workflow_step}"
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True

    def __init__(self, **overrides):
        """
        Initialise the orchestrator and build the report sub-agent.

        Parameters
        ----------
        **overrides : dict
            Keyword arguments forwarded to ``BaseAgent.__init__``.

        Returns
        -------
        None

        Notes
        -----
        ``SLOReportAgent`` is built once and reused across invocations.
        The knowledge MCP subprocess is managed by the shared client helper so
        calls to ``retrieve_knowledge`` are reused across requests.
        """
        super().__init__(**overrides)
        self._report_agent = SLOReportAgent().build()

    async def run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the SLO recommendation workflow step by step.

        Parameters
        ----------
        ctx : InvocationContext
            ADK invocation context providing session state and user content.

        Returns
        -------
        AsyncGenerator of Event
            Yields ADK events including the final structured report event
            and an optional summary text event.

        Notes
        -----
        Each step (GENERATE, FEASIBILITY, OPTIMIZE, REPORT) reads from and
        writes to ``ctx.session.state`` using the schema key constants defined
        in ``recommendation_agent.schema``. The step cursor is stored in
        ``REC_WORKFLOW_KEY``. The REPORT step delegates to ``SLOReportAgent``
        via ``run_async``. After REPORT completes, a final Event is yielded
        with the report summary text and a state delta persisting the result.
        """
        state = ctx.session.state
        step  = RecWorkflowStep(state.get(REC_WORKFLOW_KEY, RecWorkflowStep.GENERATE))
        logger.info("RecommendationOrchestratorAgent: step={}", step)

        svc_name = state.get(REC_SERVICE_KEY, "unknown")

        # ── AWAIT_INPUT: resume after human answers ───────────────────────────
        if step == RecWorkflowStep.AWAIT_INPUT:
            answers = state.get(REC_USER_ANSWERS_KEY, "")
            if not answers:
                # Still waiting — re-yield the questions so the UI shows them
                pending_raw = state.get(REC_PENDING_QUESTIONS_KEY, "[]")
                pending = json.loads(pending_raw) if isinstance(pending_raw, str) else pending_raw
                questions_text = "\n".join(
                    f"[{q.get('question_id')}] {q.get('question')}\n  Options: {', '.join(q.get('options', []))}"
                    for q in (pending if isinstance(pending, list) else [])
                )
                yield Event(
                    author=self.name,
                    content=genai_types.Content(
                        parts=[genai_types.Part(text=f"Waiting for human input:\n{questions_text}")]
                    ),
                )
                return
            # Answers received — fall through to GENERATE
            logger.info("REC AWAIT_INPUT | answers received, continuing to GENERATE: {}", answers[:200])
            state[REC_WORKFLOW_KEY] = RecWorkflowStep.GENERATE
            step = RecWorkflowStep.GENERATE

        if step == RecWorkflowStep.GENERATE:
            graph_raw = state.get(GRAPH_PAYLOAD_KEY, [])
            graph_svcs = graph_raw if isinstance(graph_raw, list) else _load_json(graph_raw).get("services", [])
            if not isinstance(graph_svcs, list):
                graph_svcs = []

            tier = "medium"
            tier_found = False
            has_ext = False
            for s in graph_svcs:
                if isinstance(s, dict) and s.get("service") == svc_name:
                    tier = s.get("tier", "medium")
                    tier_found = True
                    has_ext = any(
                        d.get("dep_type") == "external"
                        for d in s.get("depends_on", [])
                        if isinstance(d, dict)
                    )
                    break

            # ── Decision: pause for human input if service is unknown ─────────
            user_answers = state.get(REC_USER_ANSWERS_KEY, "")
            if not tier_found and not user_answers:
                logger.info(
                    "REC GENERATE | service '{}' not in dependency graph — pausing for tier assignment",
                    svc_name,
                )
                questions = [
                    RecPlannerQuestion(
                        question_id="tier_assignment",
                        question=(
                            f"Service '{svc_name}' is not in the dependency graph. "
                            "What tier should it be assigned?"
                        ),
                        options=["critical", "high", "medium", "low"],
                    ),
                    RecPlannerQuestion(
                        question_id="has_external_deps",
                        question=f"Does '{svc_name}' have external dependencies (third-party APIs, SaaS)?",
                        options=["yes", "no"],
                    ),
                ]
                questions_json = json.dumps([q.model_dump() for q in questions])
                state[REC_PENDING_QUESTIONS_KEY] = questions_json
                state[REC_WORKFLOW_KEY] = RecWorkflowStep.AWAIT_INPUT

                questions_text = "\n".join(
                    f"[{q.question_id}] {q.question}\n  Options: {', '.join(q.options)}"
                    for q in questions
                )
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={
                        REC_PENDING_QUESTIONS_KEY: questions_json,
                        REC_WORKFLOW_KEY: RecWorkflowStep.AWAIT_INPUT,
                    }),
                    content=genai_types.Content(
                        parts=[genai_types.Part(
                            text=(
                                f"Pipeline paused: service '{svc_name}' needs tier assignment.\n\n"
                                f"{questions_text}"
                            )
                        )]
                    ),
                )
                return

            # Extract tier override from human answers if provided
            if user_answers and not tier_found:
                for line in user_answers.splitlines():
                    if "tier_assignment" in line.lower():
                        for t in ("critical", "high", "medium", "low"):
                            if t in line.lower():
                                tier = t
                                break
                    if "has_external_deps" in line.lower():
                        has_ext = "yes" in line.lower()

            metrics_query  = _load_json(state.get(METRICS_QUERY_KEY))
            metrics_report = _load_json(state.get(METRICS_REPORT_KEY))
            metrics_data   = {**metrics_query, **metrics_report}
            drift_detected   = str(metrics_data.get("drift_detected", False)).lower()
            anomaly_severity = str(metrics_data.get("anomaly_severity", "none"))
            service_type     = _infer_service_type(svc_name)

            query_parts = [f"{svc_name} SLO availability latency recommendation", service_type, tier]
            if drift_detected == "true":
                query_parts.append("drift kl divergence")
            if anomaly_severity not in ("none", "low"):
                query_parts.append(f"anomaly {anomaly_severity}")
            query = " ".join(query_parts)

            state[REC_KNOWLEDGE_QUERY_KEY] = query
            logger.info(
                "REC GENERATE | calling knowledge MCP client: "
                "service={} type={} tier={} drift={} anomaly={} has_ext={} query={}",
                svc_name, service_type, tier, drift_detected, anomaly_severity, has_ext, query,
            )
            knowledge_data = await knowledge_client.retrieve_knowledge(
                query=query,
                top_k=4,
                doc_type="all",
            )
            state[REC_KNOWLEDGE_KEY] = json.dumps(knowledge_data)
            logger.info("REC GENERATE | knowledge result: {}", str(knowledge_data)[:300])
            knowledge_ctx    = knowledge_data.get("context_summary", "")
            knowledge_srcs   = knowledge_data.get("source_ids", [])

            dep_data    = _load_json(state.get(DEP_REPORT_OUTPUT_KEY))
            dep_slos_raw = state.get(REC_DEP_SLOS_KEY, {})
            dep_slos    = dep_slos_raw if isinstance(dep_slos_raw, dict) else _load_json(dep_slos_raw)

            rec_input = json.dumps({
                "service_name":      svc_name,
                "metrics_summary":   metrics_data,
                "graph_analysis":    dep_data,
                "dep_slos":          dep_slos,
                "knowledge_context": knowledge_ctx,
                "knowledge_sources": knowledge_srcs,
            })

            logger.info(
                "REC GENERATE | calling generate_slo_recommendation: "
                "service={} metrics_keys={} graph_keys={} dep_slos_count={}",
                svc_name,
                list(metrics_data.keys())[:8],
                list(dep_data.keys())[:8],
                len(dep_slos),
            )
            gen_raw = generate_slo_recommendation(rec_input_json=rec_input)
            state[REC_GENERATION_KEY] = gen_raw
            logger.info("REC GENERATE | raw result: {}", gen_raw[:500])

            try:
                gen = json.loads(gen_raw)
                logger.info(
                    "REC GENERATE | availability={:.6f} latency_p99={:.0f}ms "
                    "confidence={:.4f} requires_review={} reason={}",
                    gen.get("recommended_availability", 0.0),
                    gen.get("recommended_latency_p99_ms", 0.0),
                    gen.get("confidence_score", 0.0),
                    gen.get("requires_human_review", False),
                    gen.get("review_reason", ""),
                )
            except Exception as e:
                logger.warning("REC GENERATE | parse failed: {}", e)

            state[REC_WORKFLOW_KEY] = RecWorkflowStep.FEASIBILITY
            step = RecWorkflowStep.FEASIBILITY

        if step == RecWorkflowStep.FEASIBILITY:
            gen  = _load_json(state.get(REC_GENERATION_KEY))
            met  = _load_json(state.get(METRICS_REPORT_KEY))
            dep_slos_raw = state.get(REC_DEP_SLOS_KEY, {})
            dep_slos     = dep_slos_raw if isinstance(dep_slos_raw, dict) else _load_json(dep_slos_raw)

            proposed_avail = gen.get("recommended_availability", 0.99)
            proposed_lat   = gen.get("recommended_latency_p99_ms", 200.0)
            hist_avail     = met.get("posterior_mean", met.get("smoothed_availability", 0.99))
            dep_avails     = {
                k: v.get("recommended_availability", 0.999) if isinstance(v, dict) else float(v)
                for k, v in dep_slos.items()
            }

            feas_input = json.dumps({
                "service_name":           svc_name,
                "proposed_availability":  proposed_avail,
                "proposed_latency_p99_ms": proposed_lat,
                "historical_availability": hist_avail,
                "dep_availabilities":     dep_avails,
            })

            logger.info(
                "REC FEASIBILITY | calling check_slo_feasibility: "
                "proposed_avail={:.6f} proposed_lat={:.0f}ms hist_avail={:.6f} dep_count={}",
                proposed_avail, proposed_lat, hist_avail, len(dep_avails),
            )
            feas_raw = check_slo_feasibility(feasibility_input_json=feas_input)
            state[REC_FEASIBILITY_KEY] = feas_raw
            logger.info("REC FEASIBILITY | raw result: {}", feas_raw[:400])

            try:
                feas = json.loads(feas_raw)
                logger.info(
                    "REC FEASIBILITY | is_feasible={} score={:.4f} ceiling={:.6f} adjusted={}",
                    feas.get("is_feasible"), feas.get("feasibility_score", 0.0),
                    feas.get("availability_ceiling", 0.0),
                    feas.get("adjusted_recommendation"),
                )
            except Exception as e:
                logger.warning("REC FEASIBILITY | parse failed: {}", e)

            state[REC_WORKFLOW_KEY] = RecWorkflowStep.OPTIMIZE
            step = RecWorkflowStep.OPTIMIZE

        if step == RecWorkflowStep.OPTIMIZE:
            svcs = state.get(REC_SERVICES_LIST_KEY, [])
            if isinstance(svcs, list) and len(svcs) > 1:
                hist_avail_raw = state.get(REC_HIST_AVAIL_KEY, {})
                hist_avail     = hist_avail_raw if isinstance(hist_avail_raw, dict) else _load_json(hist_avail_raw)
                weights_raw    = state.get(REC_WEIGHTS_KEY, {})
                weights        = weights_raw if isinstance(weights_raw, dict) else _load_json(weights_raw)

                if not hist_avail:
                    metrics_data = {
                        **_load_json(state.get(METRICS_QUERY_KEY)),
                        **_load_json(state.get(METRICS_REPORT_KEY)),
                    }
                    target_posterior = metrics_data.get("posterior_mean", 0.95)
                    hist_avail = {svc: 0.95 for svc in svcs}
                    hist_avail[svc_name] = target_posterior

                if not weights:
                    dep_data = _load_json(state.get(DEP_REPORT_OUTPUT_KEY))
                    blast_map = {
                        s["service"]: s["score"]
                        for s in dep_data.get("top_services_by_blast_radius", [])
                        if isinstance(s, dict)
                    }
                    weights = {svc: 1.0 + blast_map.get(svc, 0.0) for svc in svcs}

                error_budget = sum(
                    1.0 - hist_avail.get(svc, 0.95)
                    for svc in svcs
                )
                opt_input = json.dumps({
                    "services":               svcs,
                    "historical_availability": hist_avail,
                    "importance_weights":      weights,
                    "sync_deps":               {},
                    "error_budget":            error_budget,
                })
                logger.info(
                    "REC OPTIMIZE | calling run_milp_optimization: {} services",
                    len(svcs),
                )
                opt_raw = run_milp_optimization(opt_input_json=opt_input)
                state[REC_OPTIMIZER_KEY] = opt_raw
                logger.info("REC OPTIMIZE | raw result: {}", opt_raw[:400])

                try:
                    opt = json.loads(opt_raw)
                    logger.info(
                        "REC OPTIMIZE | optimal_slos={} budget_allocation={}",
                        dict(list(opt.get("optimal_slos", {}).items())[:5]),
                        dict(list(opt.get("error_budget_allocation", {}).items())[:5]),
                    )
                except Exception as e:
                    logger.warning("REC OPTIMIZE | parse failed: {}", e)
            else:
                logger.info("REC OPTIMIZE | skipped (single service or empty list)")
                state[REC_OPTIMIZER_KEY] = "{}"

            state[REC_WORKFLOW_KEY] = RecWorkflowStep.REPORT

        logger.info("REC REPORT | running LLM report synthesizer")
        async for event in self._report_agent.run_async(ctx):
            yield event

        state[REC_WORKFLOW_KEY] = RecWorkflowStep.DONE

        report_raw = state.get(REC_REPORT_KEY, "{}")
        r = _load_json(report_raw)
        try:
            logger.info(
                "REC REPORT | service={} availability={:.6f} latency_p99={:.0f}ms "
                "confidence={:.4f} feasible={} review_required={}",
                r.get("service_name", "?"),
                r.get("recommended_availability", 0.0),
                r.get("recommended_latency_p99_ms", 0.0),
                r.get("confidence_score", 0.0),
                r.get("is_feasible", "?"),
                r.get("requires_human_review", "?"),
            )
        except Exception:
            logger.warning("REC REPORT | could not parse report for logging")

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={REC_REPORT_KEY: report_raw}),
            content=genai_types.Content(
                parts=[genai_types.Part(text=r.get("summary", "SLO recommendation complete.") if isinstance(r, dict) else "SLO recommendation complete.")]
            ),
        )


recommendation_agent = RecommendationOrchestratorAgent().build()

root_agent = recommendation_agent
