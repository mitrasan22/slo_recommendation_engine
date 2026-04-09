"""
Schemas and state keys for the dependency analysis agent workflow.

Notes
-----
Defines all ADK session state key constants and Pydantic output schemas used by
the dependency planner, analyzer, cycle detector, report synthesizer, and
orchestrator agents.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal, Optional

from google.adk.agents.base_agent import BaseAgentState
from pydantic import BaseModel, Field

GRAPH_PAYLOAD_KEY           = "graph_payload"
TARGET_SUBGRAPH_KEY         = "target_subgraph_payload"
DEP_WORKFLOW_STEP_KEY       = "dependency_workflow_step"
DEP_PLAN_OUTPUT_KEY         = "dependency_plan_output"
DEP_ANALYSIS_OUTPUT_KEY     = "dependency_analysis_output"
DEP_CYCLES_OUTPUT_KEY       = "dependency_cycles_output"
DEP_REPORT_OUTPUT_KEY       = "dependency_report_output"
DEP_PENDING_QUESTIONS_KEY   = "dep_pending_questions"
DEP_USER_ANSWERS_KEY        = "dep_user_answers"


class DependencyWorkflowStep(StrEnum):
    """
    Enumeration of dependency analysis workflow steps.

    Attributes
    ----------
    PLAN : str
        Initial planning step where the planner inspects the graph.
    AWAIT_INPUT : str
        Paused state waiting for human clarification answers.
    INGEST : str
        Graph ingestion and mathematical analysis step.
    CYCLES : str
        Circular dependency detection step.
    REPORT : str
        Final LLM report synthesis step.
    DONE : str
        Terminal state after the report has been written.

    Notes
    -----
    Steps are ordered and comparable using StrEnum. The orchestrator uses
    ``<=`` comparisons to skip already-completed steps on resume.
    """

    PLAN        = "PLAN"
    AWAIT_INPUT = "AWAIT_INPUT"
    INGEST      = "INGEST"
    CYCLES      = "CYCLES"
    REPORT      = "REPORT"
    DONE        = "DONE"


class PlannerQuestion(BaseModel):
    """
    A structured question from the dependency planner for human clarification.

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
    Serialised to JSON and stored in ``dep_pending_questions`` state key.
    The engineer's answers are stored in ``dep_user_answers`` on the next turn.
    """

    question_id: str = Field(..., description="Stable identifier — used to match answers.")
    question: str    = Field(..., description="Question text shown to the user.")
    options: list[str] = Field(..., description="Suggested choices — user may pick one or answer freely.")


class DependencyPlannerSchema(BaseModel):
    """
    Structured output schema for the dependency planner agent.

    Attributes
    ----------
    decision : Literal["READY", "NEEDS_INPUT"]
        Planner decision: ``"READY"`` to proceed or ``"NEEDS_INPUT"`` when
        critical information is missing.
    service_count : int
        Number of services in the graph.
    edge_count : int
        Number of dependency edges in the graph.
    critical_services : list of str
        Services identified as critical for SLO analysis.
    analysis_priority : list of str
        Ordered list of services to prioritise in analysis.
    questions_for_user : list of PlannerQuestion
        Non-empty only when ``decision="NEEDS_INPUT"``.
    notes : str
        Free-text planner notes for the engineer.

    Notes
    -----
    Stored in state under ``DEP_PLAN_OUTPUT_KEY`` after the planner agent runs.
    """

    decision: Literal["READY", "NEEDS_INPUT"] = Field(
        ...,
        description=(
            "READY — enough information to proceed with graph analysis. "
            "NEEDS_INPUT — critical information is missing; questions_for_user is populated."
        ),
    )
    service_count: int = Field(default=0)
    edge_count: int    = Field(default=0)
    critical_services: list[str] = Field(default_factory=list)
    analysis_priority: list[str] = Field(default_factory=list)
    questions_for_user: list[PlannerQuestion] = Field(
        default_factory=list,
        description="Non-empty only when decision=NEEDS_INPUT.",
    )
    notes: str = Field(default="")


class DependencyAnalysisSchema(BaseModel):
    """
    Structured output schema for graph analysis results.

    Attributes
    ----------
    pagerank : dict
        Mapping of service name to PageRank score.
    betweenness : dict
        Mapping of service name to betweenness centrality score.
    blast_radius : dict
        Mapping of service name to blast radius score.
    critical_path : list of str
        Ordered list of service names on the critical latency path.
    critical_path_latency_ms : float
        Total latency of the critical path in milliseconds.
    dag_is_valid : bool
        Whether the dependency graph is a valid directed acyclic graph.
    circular_deps : list
        List of detected circular dependency cycles.

    Notes
    -----
    Stored in state under ``DEP_ANALYSIS_OUTPUT_KEY`` after the INGEST step.
    The ``circular_deps`` field accepts both ``list[list[str]]`` (from the tool)
    and ``list[dict]`` (from the LLM formatter).
    """

    pagerank: dict[str, float]      = Field(default_factory=dict)
    betweenness: dict[str, float]   = Field(default_factory=dict)
    blast_radius: dict[str, float]  = Field(default_factory=dict)
    critical_path: list[str]        = Field(default_factory=list)
    critical_path_latency_ms: float = Field(default=0.0)
    dag_is_valid: bool              = Field(default=True)
    circular_deps: list             = Field(default_factory=list)


class DependencyCycleSchema(BaseModel):
    """
    Structured output schema for circular dependency detection results.

    Attributes
    ----------
    count : int
        Total number of circular dependency cycles detected.
    cycles : list of list of str
        Each inner list is an ordered sequence of service names forming a cycle.
    recommendations : list of str
        Suggested resolutions for each detected cycle.
    blocks_series_formula : bool
        Whether any cycle blocks the use of the series availability formula.

    Notes
    -----
    Stored in state under ``DEP_CYCLES_OUTPUT_KEY`` after the CYCLES step.
    """

    count: int                      = Field(default=0)
    cycles: list[list[str]]         = Field(default_factory=list)
    recommendations: list[str]      = Field(default_factory=list)
    blocks_series_formula: bool     = Field(default=False)


class DependencyReportSchema(BaseModel):
    """
    Structured output schema for the dependency analysis report.

    Attributes
    ----------
    service_count : int
        Total number of services in the analysed graph.
    edge_count : int
        Total number of dependency edges.
    dag_is_valid : bool
        Whether the graph is a valid DAG.
    critical_path : list of str
        Service names on the critical latency path.
    critical_path_latency_ms : float
        Critical path latency in milliseconds.
    top_services_by_pagerank : list of dict
        Top 5 services by PageRank score as ``[{"service": name, "score": value}]``.
    top_services_by_blast_radius : list of dict
        Top 5 services by blast radius as ``[{"service": name, "score": value}]``.
    circular_deps_count : int
        Number of circular dependency cycles detected.
    slo_ceiling_notes : list of str
        Engineer-facing notes about SLO implications from the graph structure.
    user_inputs_applied : dict
        Dictionary recording any human answers that were applied.
    summary : str
        Concise engineer-facing summary of graph health and SLO implications.

    Notes
    -----
    Stored in state under ``DEP_REPORT_OUTPUT_KEY`` after the REPORT step.
    """

    service_count: int
    edge_count: int
    dag_is_valid: bool
    critical_path: list[str]
    critical_path_latency_ms: float
    top_services_by_pagerank: list[dict]
    top_services_by_blast_radius: list[dict]
    circular_deps_count: int
    slo_ceiling_notes: list[str]
    user_inputs_applied: dict       = Field(default_factory=dict)
    summary: str


class DependencyOrchestratorState(BaseAgentState):
    """
    Typed ADK session state for the dependency analysis orchestrator.

    Attributes
    ----------
    workflow_step : DependencyWorkflowStep
        Current step in the dependency analysis workflow.
    plan_output : DependencyPlannerSchema or None
        Output from the planner agent, if completed.
    analysis_output : DependencyAnalysisSchema or None
        Output from the graph analysis step, if completed.
    cycles_output : DependencyCycleSchema or None
        Output from the cycle detection step, if completed.
    report_output : DependencyReportSchema or None
        Output from the report synthesis step, if completed.
    pending_questions : list of PlannerQuestion
        Questions awaiting human answers when in AWAIT_INPUT state.
    user_answers : str
        Raw text of the engineer's answers to the planner's questions.

    Notes
    -----
    Extends ADK ``BaseAgentState`` for compatibility with the ADK session
    serialisation layer. Fields are persisted to SQLite via the ADK runtime.
    """

    workflow_step: DependencyWorkflowStep = DependencyWorkflowStep.PLAN
    plan_output: Optional[DependencyPlannerSchema] = None
    analysis_output: Optional[DependencyAnalysisSchema] = None
    cycles_output: Optional[DependencyCycleSchema] = None
    report_output: Optional[DependencyReportSchema] = None
    pending_questions: list[PlannerQuestion] = []
    user_answers: str = ""
