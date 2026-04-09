"""
Dependency Analysis Agent for the SLO Recommendation Engine.

Notes
-----
Implements the dependency analysis workflow using the BaseAgent pattern.
The workflow progresses through: PLAN -> [AWAIT_INPUT] -> INGEST -> CYCLES -> REPORT.

AWAIT_INPUT is entered when the planner detects missing information such as
unknown external SLAs, missing latency data, or ambiguous cycles. The pipeline
pauses, surfaces questions to the user, and resumes on the next session turn
with answers stored in state.
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
from slo_engine.agents.dependency_agent.prompt import (
    dependency_analyzer_tool_prompt,
    dependency_cycle_tool_prompt,
    dependency_orchestrator_prompt,
    dependency_planner_prompt,
    dependency_report_prompt,
)
from slo_engine.agents.dependency_agent.schema import (
    DEP_ANALYSIS_OUTPUT_KEY,
    DEP_CYCLES_OUTPUT_KEY,
    DEP_PENDING_QUESTIONS_KEY,
    DEP_PLAN_OUTPUT_KEY,
    DEP_REPORT_OUTPUT_KEY,
    DEP_USER_ANSWERS_KEY,
    DEP_WORKFLOW_STEP_KEY,
    GRAPH_PAYLOAD_KEY,
    TARGET_SUBGRAPH_KEY,
    DependencyAnalysisSchema,
    DependencyCycleSchema,
    DependencyPlannerSchema,
    DependencyReportSchema,
    DependencyWorkflowStep,
    PlannerQuestion,
)
from slo_engine.agents.dependency_agent.tools.tools import (
    analyse_dependency_graph,
    detect_circular_dependencies,
    ingest_service_dependencies,
)

logger = logger.bind(name=__name__)

T = TypeVar("T", bound=BaseModel)


def _parse_output(raw, schema: Type[T]) -> Optional[T]:
    """
    Parse raw LLM output into a Pydantic schema instance.

    Parameters
    ----------
    raw : str or dict or None
        Raw output from the LLM agent, either a JSON string or dict.
    schema : type[T]
        Pydantic model class to validate against.

    Returns
    -------
    T or None
        Validated schema instance, or ``None`` if parsing fails.

    Notes
    -----
    Attempts direct validation first, then falls back to extracting the first
    JSON object (using brace-matching) from the raw string when initial
    validation fails. Returns ``None`` rather than raising on all failure modes.
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, dict):
            return schema.model_validate(raw)
        return schema.model_validate_json(raw)
    except (ValidationError, ValueError):
        start = str(raw).find("{")
        end   = str(raw).rfind("}") + 1
        if start != -1 and end > start:
            try:
                return schema.model_validate_json(str(raw)[start:end])
            except (ValidationError, ValueError):
                pass
    return None


def _extract_user_message(ctx: InvocationContext) -> str:
    """
    Extract the latest user text from the ADK invocation context.

    Parameters
    ----------
    ctx : InvocationContext
        ADK invocation context for the current turn.

    Returns
    -------
    str
        Concatenated text from all parts of the user message, or empty string
        if no text can be extracted.

    Notes
    -----
    Accesses ``ctx.user_content.parts`` and joins text parts. Exceptions
    during extraction are silently ignored.
    """
    try:
        if ctx.user_content and ctx.user_content.parts:
            return " ".join(
                p.text for p in ctx.user_content.parts if hasattr(p, "text") and p.text
            )
    except Exception:
        pass
    return ""


def _format_questions(questions: list[PlannerQuestion]) -> str:
    """
    Format a list of planner questions into a human-readable message.

    Parameters
    ----------
    questions : list of PlannerQuestion
        List of structured questions from the dependency planner agent.

    Returns
    -------
    str
        Multi-paragraph formatted string suitable for display to the engineer.

    Notes
    -----
    Each question includes its ID, text, and lettered options. A trailing
    instruction explains the expected answer format.
    """
    lines = ["I need a few clarifications before analysing the dependency graph:\n"]
    for i, q in enumerate(questions, 1):
        opts = "\n".join(f"  {chr(96+j+1)}) {o}" for j, o in enumerate(q.options))
        lines.append(f"**Q{i}** [{q.question_id}]\n{q.question}\n{opts}")
    lines.append("\nPlease answer each question by question_id, e.g.:\n"
                 "  payment_api_sla: 99% (standard external)\n"
                 "  cycle_resolution: Treat auth as one-way dep of checkout")
    return "\n\n".join(lines)


def _generate_fallback_questions(subgraph: list[dict]) -> list[PlannerQuestion]:
    """
    Generate dependency planner questions from the subgraph when the LLM
    returns NEEDS_INPUT without populating questions_for_user.

    Parameters
    ----------
    subgraph : list of dict
        List of service node dicts from ``TARGET_SUBGRAPH_KEY``.

    Returns
    -------
    list of PlannerQuestion
        Questions about external dependency SLAs and missing latency data.
    """
    questions: list[PlannerQuestion] = []
    for svc in subgraph:
        if not isinstance(svc, dict):
            continue
        svc_name = svc.get("service", svc.get("service_name", ""))
        for dep in svc.get("depends_on", []):
            if not isinstance(dep, dict):
                continue
            if dep.get("dep_type") == "external":
                dep_name = dep.get("name", "external-dep")
                questions.append(PlannerQuestion(
                    question_id=f"{dep_name.replace('-', '_')}_sla",
                    question=f"'{dep_name}' is an external dependency with no known SLA. What availability should be assumed?",
                    options=["99.99% (premium SLA)", "99.9% (standard SLA)", "99% (basic external)", "skip — treat as unconstrained"],
                ))
    # Ask latency for services with no latency field
    svc_names = [
        s.get("service", s.get("service_name", ""))
        for s in subgraph if isinstance(s, dict)
        and not s.get("latency_p99_ms") and not s.get("p99_latency_ms")
    ]
    if svc_names:
        sample = svc_names[0]
        questions.append(PlannerQuestion(
            question_id="default_latency_assumption",
            question=(
                f"Services in the subgraph (e.g. '{sample}') have no p99 latency data. "
                "What latency ceiling should be assumed for SLO analysis?"
            ),
            options=["100ms — fast internal service", "200ms — standard", "300ms — relaxed / DB-heavy", "500ms — slow external or batch"],
        ))
    return questions


def _log_questions(questions: list[PlannerQuestion], prefix: str) -> None:
    """
    Log each planner question with its text and all options.

    Parameters
    ----------
    questions : list of PlannerQuestion
        List of structured questions to log.
    prefix : str
        Log line prefix string (e.g. ``"DEP PLAN |"``).

    Returns
    -------
    None

    Notes
    -----
    Logs at INFO level. Each question and its options are logged on separate
    lines for readability in the run log file.
    """
    for i, q in enumerate(questions, 1):
        logger.info("{} Q{} [{}]: {}", prefix, i, q.question_id, q.question)
        for j, opt in enumerate(q.options, 1):
            logger.info("{}   option {}: {}", prefix, j, opt)


class DependencyReportAgent(BaseAgent):
    """
    LLM synthesizer that reads raw analysis and cycle data and outputs a structured report.

    Attributes
    ----------
    name : str
        ADK agent name ``"dependency_report_agent"``.
    description : str
        Brief description for ADK agent discovery.
    instruction : str
        Report synthesis prompt.
    output_key : str
        State key ``DEP_REPORT_OUTPUT_KEY`` where the report is stored.

    Notes
    -----
    Transfer to parent and peers is disabled to prevent the ADK runtime from
    routing this agent's output elsewhere.
    """

    name          = "dependency_report_agent"
    description   = "Synthesizes dependency analysis results into a structured report."
    instruction   = dependency_report_prompt
    output_key    = DEP_REPORT_OUTPUT_KEY
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True


class DependencyPlannerAgent(BaseAgent):
    """
    Planning agent that inspects the graph and asks for clarification when data is missing.

    Attributes
    ----------
    name : str
        ADK agent name ``"dependency_planner_agent"``.
    description : str
        Brief description for ADK agent discovery.
    instruction : str
        Planner prompt.
    output_schema : type
        Pydantic model ``DependencyPlannerSchema``.
    output_key : str
        State key ``DEP_PLAN_OUTPUT_KEY`` where the plan is stored.

    Notes
    -----
    Transfer to parent is allowed so the orchestrator can intercept the output.
    Transfer to peers is disabled.
    """

    name        = "dependency_planner_agent"
    description = "Plans graph analysis; asks for clarification when data is missing."
    instruction = dependency_planner_prompt
    output_schema = DependencyPlannerSchema
    output_key    = DEP_PLAN_OUTPUT_KEY
    disallow_transfer_to_parent = False
    disallow_transfer_to_peers  = True


class _DepAnalyzerAgent(BaseAgent):
    """
    Tool-calling agent that ingests the service graph and runs graph analysis.

    Attributes
    ----------
    name : str
        ADK agent name ``"dependency_analyzer_agent"``.
    description : str
        Brief description.
    instruction : str
        Analyzer tool-calling prompt.
    output_key : str
        State key ``DEP_ANALYSIS_OUTPUT_KEY``.

    Notes
    -----
    Calls ``ingest_service_dependencies`` and ``analyse_dependency_graph`` tools.
    PageRank, Tarjan SCC, Bellman-Ford critical path, and betweenness centrality
    are computed deterministically without LLM involvement.
    """

    name          = "dependency_analyzer_agent"
    description   = "Runs PageRank, Tarjan SCC, Bellman-Ford, Betweenness on the graph."
    instruction   = dependency_analyzer_tool_prompt
    output_key    = DEP_ANALYSIS_OUTPUT_KEY
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True

    def get_tools(self):
        """
        Return the graph ingestion and analysis tools.

        Returns
        -------
        list
            List containing FunctionTool wrappers for ingest and analysis functions.

        Notes
        -----
        Tools are wrapped in FunctionTool for ADK-compatible invocation.
        """
        return [
            FunctionTool(ingest_service_dependencies),
            FunctionTool(analyse_dependency_graph),
        ]


def _build_dependency_analyzer_agent():
    """
    Build and return the dependency analyzer agent instance.

    Returns
    -------
    ADKAgentType
        Built ADK agent for dependency analysis.

    Notes
    -----
    Called once during orchestrator initialisation.
    """
    return _DepAnalyzerAgent().build()


class _DepCycleAgent(BaseAgent):
    """
    Tool-calling agent that detects circular dependencies using Tarjan SCC.

    Attributes
    ----------
    name : str
        ADK agent name ``"dependency_cycle_detector_agent"``.
    description : str
        Brief description.
    instruction : str
        Cycle detection tool-calling prompt.
    output_key : str
        State key ``DEP_CYCLES_OUTPUT_KEY``.

    Notes
    -----
    Calls ``detect_circular_dependencies`` deterministically without LLM involvement.
    """

    name          = "dependency_cycle_detector_agent"
    description   = "Detects circular dependencies using Tarjan SCC."
    instruction   = dependency_cycle_tool_prompt
    output_key    = DEP_CYCLES_OUTPUT_KEY
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True

    def get_tools(self):
        """
        Return the circular dependency detection tool.

        Returns
        -------
        list
            List containing a FunctionTool wrapper for ``detect_circular_dependencies``.

        Notes
        -----
        Tool is wrapped in FunctionTool for ADK-compatible invocation.
        """
        return [FunctionTool(detect_circular_dependencies)]


def _build_dependency_cycle_agent():
    """
    Build and return the cycle detection agent instance.

    Returns
    -------
    ADKAgentType
        Built ADK agent for cycle detection.

    Notes
    -----
    Called once during orchestrator initialisation.
    """
    return _DepCycleAgent().build()


class DependencyOrchestratorAgent(BaseAgent):
    """
    Orchestrates the full dependency analysis workflow.

    Attributes
    ----------
    name : str
        ADK agent name ``"dependency_agent"``.
    description : str
        Full description including analysis capabilities.
    orchestration : str
        ``"single"`` orchestration mode.
    instruction : str
        Orchestrator system prompt.

    Notes
    -----
    Drives the PLAN -> [AWAIT_INPUT] -> INGEST -> CYCLES -> REPORT workflow.
    If the planner returns NEEDS_INPUT, the pipeline pauses and yields the
    questions to the user, returning immediately. On the next session turn
    the user's answers are read from the message and stored in state before
    advancing to INGEST. INGEST and CYCLES call the math tools directly in
    Python without LLM involvement. The REPORT step uses the LLM report
    synthesizer.
    """

    name        = "dependency_agent"
    description = (
        "Analyses microservice dependency graphs using PageRank, Tarjan SCC, "
        "Bellman-Ford critical path, and betweenness centrality. "
        "Asks the engineer for clarification when data is missing."
    )
    orchestration = "single"
    instruction   = dependency_orchestrator_prompt
    disallow_transfer_to_parent = True
    disallow_transfer_to_peers  = True

    def __init__(self, **overrides):
        """
        Initialise the orchestrator and build all sub-agent instances.

        Parameters
        ----------
        **overrides : dict
            Keyword arguments forwarded to the BaseAgent constructor.

        Notes
        -----
        Builds the planner, analyzer, cycle detector, and report agents
        during initialisation so they are ready when run_async_impl is called.
        """
        super().__init__(**overrides)
        self._planner_agent  = DependencyPlannerAgent().build()
        self._analyzer_agent = _build_dependency_analyzer_agent()
        self._cycle_agent    = _build_dependency_cycle_agent()
        self._report_agent   = DependencyReportAgent().build()

    async def run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the dependency analysis workflow for the current session turn.

        Parameters
        ----------
        ctx : InvocationContext
            ADK invocation context with session state and user message.

        Returns
        -------
        AsyncGenerator[Event, None]
            Async generator yielding ADK events from each workflow step.

        Notes
        -----
        Reads the current workflow step from state and resumes from that step.
        Each step updates the state step key before proceeding. The AWAIT_INPUT
        step stores user answers and advances to INGEST. INGEST and CYCLES call
        the graph math tools directly. REPORT delegates to the LLM report agent.
        """
        state = ctx.session.state
        step  = DependencyWorkflowStep(
            state.get(DEP_WORKFLOW_STEP_KEY, DependencyWorkflowStep.PLAN)
        )
        logger.info("DependencyOrchestratorAgent: step={}", step)

        if step == DependencyWorkflowStep.PLAN:
            graph_raw = state.get(GRAPH_PAYLOAD_KEY, [])
            graph_svcs = graph_raw if isinstance(graph_raw, list) else json.loads(graph_raw or "[]")
            logger.info(
                "DEP PLAN | graph has {} services: {}",
                len(graph_svcs),
                [s.get("service","?") for s in graph_svcs if isinstance(s, dict)][:10],
            )
            async for event in self._planner_agent.run_async(ctx):
                yield event

            plan = _parse_output(state.get(DEP_PLAN_OUTPUT_KEY), DependencyPlannerSchema)
            if plan:
                logger.info(
                    "DEP PLAN | decision={} service_count={} edge_count={} "
                    "critical_services={} analysis_priority={} notes={}",
                    plan.decision, plan.service_count, plan.edge_count,
                    plan.critical_services, plan.analysis_priority, plan.notes,
                )
            else:
                logger.warning("DEP PLAN | planner output parse failed. raw={}",
                               str(state.get(DEP_PLAN_OUTPUT_KEY))[:300])

            # If LLM returned NEEDS_INPUT but forgot to populate questions, generate them
            if plan and plan.decision == "NEEDS_INPUT" and not plan.questions_for_user:
                subgraph_raw = state.get(TARGET_SUBGRAPH_KEY, "[]")
                subgraph = json.loads(subgraph_raw) if isinstance(subgraph_raw, str) else subgraph_raw
                fallback = _generate_fallback_questions(subgraph if isinstance(subgraph, list) else [])
                if fallback:
                    plan = plan.model_copy(update={"questions_for_user": fallback})
                    logger.info("DEP PLAN | LLM returned NEEDS_INPUT with no questions — generated {} fallback question(s)", len(fallback))
                else:
                    logger.info("DEP PLAN | LLM returned NEEDS_INPUT with no questions and no fallbacks — treating as READY")

            if plan and plan.decision == "NEEDS_INPUT" and plan.questions_for_user:
                questions_text = _format_questions(plan.questions_for_user)
                state[DEP_PENDING_QUESTIONS_KEY] = json.dumps(
                    [q.model_dump() for q in plan.questions_for_user]
                )
                state[DEP_WORKFLOW_STEP_KEY] = DependencyWorkflowStep.AWAIT_INPUT
                logger.info("DEP PLAN | NEEDS_INPUT — {} question(s):", len(plan.questions_for_user))
                _log_questions(plan.questions_for_user, "DEP PLAN |")
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={
                        DEP_WORKFLOW_STEP_KEY:     DependencyWorkflowStep.AWAIT_INPUT,
                        DEP_PENDING_QUESTIONS_KEY: state[DEP_PENDING_QUESTIONS_KEY],
                    }),
                    content=genai_types.Content(
                        parts=[genai_types.Part(text=questions_text)]
                    ),
                )
                return

            logger.info("DEP PLAN | decision=READY — proceeding to INGEST")
            state[DEP_WORKFLOW_STEP_KEY] = DependencyWorkflowStep.INGEST

        if step == DependencyWorkflowStep.AWAIT_INPUT:
            answers = _extract_user_message(ctx)
            logger.info("DEP AWAIT_INPUT | user answers received: {}", answers[:200] if answers else "(none yet)")
            if not answers:
                yield Event(
                    author=self.name,
                    content=genai_types.Content(
                        parts=[genai_types.Part(text="Still waiting for your answers to the clarification questions.")]
                    ),
                )
                return

            state[DEP_USER_ANSWERS_KEY]   = answers
            state[DEP_WORKFLOW_STEP_KEY]  = DependencyWorkflowStep.INGEST
            state[DEP_PENDING_QUESTIONS_KEY] = ""
            logger.info("DEP AWAIT_INPUT | answers stored, advancing to INGEST")

        if step <= DependencyWorkflowStep.INGEST:
            user_ans = state.get(DEP_USER_ANSWERS_KEY, "")
            subgraph_raw = state.get(TARGET_SUBGRAPH_KEY)
            graph_raw    = state.get(GRAPH_PAYLOAD_KEY, [])
            graph_to_use = subgraph_raw if subgraph_raw else graph_raw

            if isinstance(graph_to_use, list):
                graph_json = json.dumps(graph_to_use)
            elif isinstance(graph_to_use, str):
                graph_json = graph_to_use
            else:
                graph_json = "[]"

            logger.info(
                "DEP INGEST | calling ingest_service_dependencies directly. "
                "graph_bytes={} user_answers={}",
                len(graph_json),
                user_ans[:100] if user_ans else "(none)",
            )
            ingest_result_raw = ingest_service_dependencies(services_json=graph_json)
            logger.info("DEP INGEST | ingest result: {}", ingest_result_raw)

            analysis_result_raw = analyse_dependency_graph()
            logger.info("DEP INGEST | analysis result: {}", analysis_result_raw[:1000])

            state[DEP_ANALYSIS_OUTPUT_KEY] = analysis_result_raw
            state[DEP_WORKFLOW_STEP_KEY]   = DependencyWorkflowStep.CYCLES

            try:
                a = json.loads(analysis_result_raw)
                pr = dict(sorted(a.get("pagerank", {}).items(), key=lambda x: x[1], reverse=True))
                bc = dict(sorted(a.get("betweenness", {}).items(), key=lambda x: x[1], reverse=True))
                br = dict(sorted(a.get("blast_radius", {}).items(), key=lambda x: x[1], reverse=True))
                logger.info(
                    "DEP INGEST | dag_valid={} critical_path={} latency_ms={}",
                    a.get("dag_is_valid", "?"), a.get("critical_path", []),
                    a.get("critical_path_latency_ms", 0),
                )
                logger.info("DEP INGEST | pagerank (all): {}", pr)
                logger.info("DEP INGEST | betweenness (all): {}", bc)
                logger.info("DEP INGEST | blast_radius (all): {}", br)
            except Exception as e:
                logger.warning("DEP INGEST | could not parse analysis output: {}", e)

        if step <= DependencyWorkflowStep.CYCLES:
            cycles_result_raw = detect_circular_dependencies()
            logger.info("DEP CYCLES | raw result: {}", cycles_result_raw)

            try:
                c_raw = json.loads(cycles_result_raw)
                c_norm: dict = {}
                raw_cycles = c_raw.get("circular_deps", c_raw.get("cycles", []))
                parsed_cycles: list[list[str]] = []
                for item in raw_cycles:
                    if isinstance(item, dict):
                        parsed_cycles.append(item.get("cycle", []))
                    elif isinstance(item, list):
                        parsed_cycles.append(item)
                c_norm["cycles"]               = parsed_cycles
                c_norm["count"]                = c_raw.get("count", len(parsed_cycles))
                c_norm["recommendations"]      = [
                    item.get("recommendation", "") for item in raw_cycles if isinstance(item, dict)
                ]
                c_norm["blocks_series_formula"] = len(parsed_cycles) > 0
                cycles_normalised = json.dumps(c_norm)
            except Exception as e:
                logger.warning("DEP CYCLES | normalisation failed: {}", e)
                cycles_normalised = cycles_result_raw

            state[DEP_CYCLES_OUTPUT_KEY]   = cycles_normalised
            state[DEP_WORKFLOW_STEP_KEY]   = DependencyWorkflowStep.REPORT

            try:
                c = json.loads(cycles_normalised)
                logger.info(
                    "DEP CYCLES | count={} blocks_series_formula={}",
                    c.get("count", 0), c.get("blocks_series_formula", False),
                )
                for i, cyc in enumerate(c.get("cycles", []), 1):
                    logger.info("DEP CYCLES |   cycle {}: {}", i, cyc)
                if not c.get("cycles"):
                    logger.info("DEP CYCLES |   no circular dependencies detected")
            except Exception:
                pass

        logger.info("DEP REPORT | running LLM report synthesizer")
        async for event in self._report_agent.run_async(ctx):
            yield event

        state[DEP_WORKFLOW_STEP_KEY] = DependencyWorkflowStep.DONE
        report_raw = state.get(DEP_REPORT_OUTPUT_KEY, "{}")
        try:
            r = json.loads(report_raw) if isinstance(report_raw, str) else report_raw
            logger.info(
                "DEP REPORT | services={} dag_valid={} cycles={} slo_notes={}",
                r.get("service_count", "?"), r.get("dag_is_valid", "?"),
                r.get("circular_deps_count", "?"), r.get("slo_ceiling_notes", []),
            )
        except Exception:
            logger.warning("DEP REPORT | could not parse report for logging")


dependency_agent = DependencyOrchestratorAgent().build()

root_agent = dependency_agent
