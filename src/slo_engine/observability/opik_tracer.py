"""
Comet Opik instrumentation for the SLO Recommendation Engine.

Notes
-----
Wraps agent pipeline runs with traces so every LLM call, token count,
and recommendation outcome is visible in the Opik dashboard. Opik is
an optional dependency: if the ``opik`` package is not installed or
``OPIK_API_KEY``/``OPIK_URL`` env vars are absent, all tracing calls
become silent no-ops so the engine runs unaffected.

Usage::

    from slo_engine.observability.opik_tracer import trace_pipeline

    async with trace_pipeline("slo_recommendation", inputs={...}) as tracer:
        tracer.log_step("dependency_agent", output={...})
        tracer.log_recommendation(service, rec, sources, decision)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from loguru import logger

try:
    import opik
    _OPIK_ENABLED = bool(os.getenv("OPIK_API_KEY") or os.getenv("OPIK_URL"))
    if _OPIK_ENABLED:
        opik.configure(
            api_key=os.getenv("OPIK_API_KEY", ""),
            workspace=os.getenv("OPIK_WORKSPACE", "default"),
        )
except ImportError:
    _OPIK_ENABLED = False
    opik = None  # type: ignore

_PROJECT = os.getenv("OPIK_PROJECT_NAME", "slo-recommendation-engine")


class PipelineTracer:
    """
    Lightweight wrapper around an Opik trace for one SLO pipeline run.

    Attributes
    ----------
    _name : str
        Name of the trace used for display in the Opik dashboard.
    _inputs : dict
        Input payload passed to the Opik trace on creation.
    _trace : opik.Trace or None
        Active Opik trace object, or ``None`` when Opik is disabled.
    _steps : list of dict
        Accumulated list of logged step dicts for final output metadata.
    _started_at : datetime
        UTC datetime when this tracer was instantiated.

    Notes
    -----
    When Opik is disabled or unavailable, all methods silently no-op so
    the calling pipeline is not affected. Exceptions from Opik API calls
    are caught and logged at DEBUG level.
    """

    def __init__(self, trace_name: str, inputs: dict[str, Any]):
        """
        Initialise the tracer with a name and input payload.

        Parameters
        ----------
        trace_name : str
            Human-readable name for the pipeline trace.
        inputs : dict
            Input data to attach to the trace for observability.

        Returns
        -------
        None

        Notes
        -----
        Does not start the Opik trace — call ``_start`` explicitly or use
        the ``trace_pipeline`` context manager which calls it automatically.
        """
        self._name = trace_name
        self._inputs = inputs
        self._trace = None
        self._steps: list[dict] = []
        self._started_at = datetime.now(UTC)

    def _start(self):
        """
        Start the Opik trace for this pipeline run.

        Returns
        -------
        None

        Notes
        -----
        Creates an ``opik.Trace`` via ``opik.Opik`` client with the ``slo-engine``
        tag. Silently no-ops when Opik is disabled. Exceptions during trace
        creation are caught and logged at DEBUG level to avoid disrupting the pipeline.
        """
        if _OPIK_ENABLED and opik:
            try:
                client = opik.Opik(project_name=_PROJECT)
                self._trace = client.trace(
                    name=self._name,
                    input=self._inputs,
                    tags=["slo-engine"],
                )
            except Exception as exc:
                logger.debug("Opik trace start failed: {}", exc)

    def log_step(self, step_name: str, output: dict[str, Any], metadata: dict | None = None):
        """
        Log a completed agent step as an Opik span.

        Parameters
        ----------
        step_name : str
            Name of the agent step (e.g. ``"dependency_agent"``).
        output : dict
            Output payload from the step to record.
        metadata : dict, optional
            Additional metadata key-value pairs to attach to the span.
            Defaults to an empty dict.

        Returns
        -------
        None

        Notes
        -----
        Always appends to ``_steps`` regardless of Opik availability, so the
        step count is accurate for ``_end``. Opik span creation and ``span.end()``
        are guarded by try/except to prevent tracing failures from propagating.
        """
        self._steps.append({"step": step_name, "output": output, "metadata": metadata or {}})
        if self._trace:
            try:
                span = self._trace.span(
                    name=step_name,
                    input=self._inputs,
                    output=output,
                    metadata=metadata or {},
                )
                span.end()
            except Exception as exc:
                logger.debug("Opik span log failed: {}", exc)

    def log_recommendation(
        self,
        service_name: str,
        recommendation: dict[str, Any],
        sources: list[str],
        decision: str,
    ):
        """
        Log the final recommendation outcome for the audit trail.

        Parameters
        ----------
        service_name : str
            Name of the service the recommendation applies to.
        recommendation : dict
            SLO recommendation payload with ``availability``, ``latency_p99_ms``,
            and ``confidence_score`` keys.
        sources : list of str
            Knowledge sources cited in the recommendation.
        decision : str
            Approval decision string (e.g. ``"auto_approved"``, ``"pending_review"``).

        Returns
        -------
        None

        Notes
        -----
        Always emits an INFO log via loguru regardless of Opik availability.
        When Opik is active, logs a feedback score of ``1.0`` for
        ``"auto_approved"`` decisions and ``0.5`` for all others, enabling
        dashboard filtering by outcome quality.
        """
        logger.info(
            "AUDIT | service={} avail={} conf={:.2f} sources={} decision={}",
            service_name,
            recommendation.get("availability"),
            recommendation.get("confidence_score", 0),
            sources,
            decision,
        )
        if self._trace:
            try:
                self._trace.log_feedback_score(
                    name="recommendation_decision",
                    value=1.0 if decision == "auto_approved" else 0.5,
                    reason=decision,
                )
            except Exception as exc:
                logger.debug("Opik feedback log failed: {}", exc)

    def _end(self, output: dict[str, Any]):
        """
        End the Opik trace with a final output payload.

        Parameters
        ----------
        output : dict
            Final output metadata to attach to the closing trace record.

        Returns
        -------
        None

        Notes
        -----
        Called automatically by the ``trace_pipeline`` context manager in its
        ``finally`` block. Silently no-ops when ``_trace`` is ``None``.
        """
        if self._trace:
            try:
                self._trace.end(output=output)
            except Exception as exc:
                logger.debug("Opik trace end failed: {}", exc)


@asynccontextmanager
async def trace_pipeline(name: str, inputs: dict[str, Any]):
    """
    Async context manager that wraps a pipeline run in an Opik trace.

    Parameters
    ----------
    name : str
        Human-readable trace name shown in the Opik dashboard.
    inputs : dict
        Input payload attached to the trace on creation.

    Yields
    ------
    PipelineTracer
        Active tracer instance for logging steps and the final recommendation.

    Notes
    -----
    Creates a ``PipelineTracer``, starts it, yields it to the caller, then
    always calls ``_end`` in a ``finally`` block to close the trace cleanly
    even if the pipeline raises an exception.
    """
    tracer = PipelineTracer(name, inputs)
    tracer._start()
    try:
        yield tracer
    finally:
        tracer._end({"steps_completed": len(tracer._steps)})


def log_recommendation_audit(
    service_name: str,
    recommendation: dict[str, Any],
    sources: list[str],
    decision: str,
) -> None:
    """
    Write a standalone audit log entry without an active pipeline trace.

    Parameters
    ----------
    service_name : str
        Name of the service the recommendation applies to.
    recommendation : dict
        SLO recommendation payload with ``availability``, ``latency_p99_ms``,
        and ``confidence_score`` keys.
    sources : list of str
        Knowledge sources cited in the recommendation.
    decision : str
        Approval decision string (e.g. ``"auto_approved"``, ``"pending_review"``).

    Returns
    -------
    None

    Notes
    -----
    Always emits a structured INFO log via loguru. When Opik is enabled,
    additionally creates a standalone trace named
    ``slo_recommendation_<service_name>`` tagged with ``"audit"`` and the
    decision string. Opik errors are caught and logged at DEBUG level.
    """
    logger.info(
        "AUDIT TRAIL | service={} availability={} latency_p99={} confidence={:.2f} "
        "sources={} decision={} timestamp={}",
        service_name,
        recommendation.get("availability"),
        recommendation.get("latency_p99_ms"),
        recommendation.get("confidence_score", 0),
        sources,
        decision,
        datetime.now(UTC).isoformat(),
    )
    if _OPIK_ENABLED and opik:
        try:
            client = opik.Opik(project_name=_PROJECT)
            trace = client.trace(
                name=f"slo_recommendation_{service_name}",
                input={"service_name": service_name, "sources": sources},
                output=recommendation,
                tags=["audit", decision],
                metadata={"decision": decision, "sources": sources},
            )
            trace.end()
        except Exception as exc:
            logger.debug("Opik audit trace failed: {}", exc)
