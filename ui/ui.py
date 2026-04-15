"""
SLO Recommendation Engine — Streamlit UI.

Notes
-----
Multi-page Streamlit application providing a complete interface to the SLO
recommendation pipeline. Pages:

  Dashboard
      Service health overview with PageRank bar chart.
  Ingest Graph
      Paste or use the sample e-commerce dependency graph JSON for ingestion.
  Agent Pipeline
      Full ADK pipeline with LLM reasoning and mid-pipeline NEEDS_INPUT
      handling. Supports session resume when the pipeline pauses to ask
      questions about tier assignment or cycle resolution.
  Fast Recommendations
      Computation-only pipeline (no LLM) with Bayesian + MILP — instant results.
  Error Budgets
      Real-time gauge charts per service with multi-window burn rate lines.
  Human Review
      Post-pipeline HITL queue for low-confidence recommendations.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE  = "http://localhost:8000/api/v1"
ADK_BASE  = "http://localhost:8000"
APP_NAME  = "slo_pipeline"
ADK_USER  = "ui-user"
PIPELINE_TIMEOUT = 360

st.set_page_config(
    page_title="SLO Recommendation Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

for _k, _v in {
    "session_id":        None,
    "pipeline_state":    "idle",
    "awaiting_questions": [],
    "pipeline_events":   [],
    "pipeline_result":   None,
    "last_service":      None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def api_get(path: str, base: str = API_BASE) -> dict | list | None:
    """
    Perform an HTTP GET request to the SLO engine API.

    Parameters
    ----------
    path : str
        API path to append to the base URL.
    base : str, optional
        Base URL. Defaults to ``API_BASE`` (``http://localhost:8000/api/v1``).

    Returns
    -------
    dict or list or None
        Parsed JSON response, or ``None`` if the request fails.

    Notes
    -----
    On any exception the error is displayed in the Streamlit UI via
    ``st.error`` and ``None`` is returned so callers can check for falsy
    results.
    """
    try:
        r = httpx.get(f"{base}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"GET {path} — {e}")
        return None


def api_post(path: str, body: dict, timeout: int = 120, base: str = API_BASE) -> dict | list | None:
    """
    Perform an HTTP POST request to the SLO engine API.

    Parameters
    ----------
    path : str
        API path to append to the base URL.
    body : dict
        JSON-serialisable request body.
    timeout : int, optional
        Request timeout in seconds. Defaults to 120.
    base : str, optional
        Base URL. Defaults to ``API_BASE``.

    Returns
    -------
    dict or list or None
        Parsed JSON response, or ``None`` if the request fails.

    Notes
    -----
    HTTP error responses are displayed with the status code and the first
    300 characters of the response body. Other exceptions are displayed as
    plain error messages.
    """
    try:
        r = httpx.post(f"{base}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"POST {path} — HTTP {e.response.status_code}: {e.response.text[:300]}")
        return None
    except Exception as e:
        st.error(f"POST {path} — {e}")
        return None


def adk_create_session(user_id: str = ADK_USER, initial_state: dict | None = None) -> str | None:
    """
    Create a new ADK session and return its ID.

    Parameters
    ----------
    user_id : str, optional
        ADK user identity. Defaults to ``ADK_USER``.
    initial_state : dict or None, optional
        Optional initial state to inject into the session.

    Returns
    -------
    str or None
        The new session ID string, or ``None`` if creation failed.

    Notes
    -----
    On failure, the API error is displayed via ``st.error`` and ``None`` is
    returned to let the caller abort cleanly.
    """
    body: dict = {}
    if initial_state:
        body["state"] = initial_state
    result = api_post(
        f"/apps/{APP_NAME}/users/{user_id}/sessions",
        body,
        base=ADK_BASE,
    )
    if result and "id" in result:
        return result["id"]
    st.error(f"Could not create ADK session: {result}")
    return None


def adk_run(session_id: str, message: str, user_id: str = ADK_USER) -> list[dict]:
    """
    Send a message to the ADK pipeline and return the list of events.

    Parameters
    ----------
    session_id : str
        Active ADK session ID.
    message : str
        User message to send to the pipeline.
    user_id : str, optional
        ADK user identity. Defaults to ``ADK_USER``.

    Returns
    -------
    list of dict
        List of ADK event dicts returned by the pipeline run. Returns an
        empty list if the response is not a list or the request fails.

    Notes
    -----
    Uses ``PIPELINE_TIMEOUT`` (360 s) as the request timeout to accommodate
    the full three-stage pipeline (~145 s + buffer).
    """
    result = api_post(
        "/run",
        {
            "app_name":   APP_NAME,
            "user_id":    user_id,
            "session_id": session_id,
            "new_message": {
                "role":  "user",
                "parts": [{"text": message}],
            },
        },
        timeout=PIPELINE_TIMEOUT,
        base=ADK_BASE,
    )
    if isinstance(result, list):
        return result
    return []


def adk_get_state(session_id: str, user_id: str = ADK_USER) -> dict:
    """
    Fetch the current session state dict from the ADK.

    Parameters
    ----------
    session_id : str
        Active ADK session ID.
    user_id : str, optional
        ADK user identity. Defaults to ``ADK_USER``.

    Returns
    -------
    dict
        Session state dict, or an empty dict if unavailable.

    Notes
    -----
    The state dict contains all intermediate pipeline outputs and the
    pending-questions keys used by ``_is_awaiting_input``.
    """
    result = api_get(
        f"/apps/{APP_NAME}/users/{user_id}/sessions/{session_id}",
        base=ADK_BASE,
    )
    if isinstance(result, dict):
        return result.get("state", {})
    return {}


def _extract_text_from_events(events: list[dict]) -> list[tuple[str, str]]:
    """
    Extract (author, text) pairs from a list of ADK events.

    Parameters
    ----------
    events : list of dict
        ADK event dicts as returned by ``adk_run``.

    Returns
    -------
    list of tuple of str
        List of ``(author, text)`` pairs for all non-empty text parts.

    Notes
    -----
    Each ADK event may contain multiple ``parts`` in its ``content`` dict.
    Only parts with a non-empty ``text`` field are included.
    """
    messages = []
    for event in events:
        author  = event.get("author", "agent")
        content = event.get("content") or {}
        if isinstance(content, dict):
            for part in content.get("parts", []):
                if isinstance(part, dict) and part.get("text"):
                    messages.append((author, part["text"]))
    return messages


def _is_awaiting_input(events: list[dict], state: dict) -> bool:
    """
    Detect whether the pipeline paused mid-execution awaiting human input.

    Parameters
    ----------
    events : list of dict
        ADK event dicts from the most recent pipeline run.
    state : dict
        Current ADK session state dict.

    Returns
    -------
    bool
        ``True`` if the pipeline has pending questions requiring user answers,
        ``False`` otherwise (including when the final report is already present).

    Notes
    -----
    Checks pending-question state keys (``dep_pending_questions``,
    ``metrics_pending_questions``, ``rec_pending_questions``). These are
    cleared by each agent after answers are processed, making them the
    reliable signal. Workflow-step keys are not used because ADK event
    state_deltas can leave stale ``AWAIT_INPUT`` values even after the agent
    has advanced to ``DONE``.
    """
    if state.get("rec_report_output") or state.get("final_slo"):
        return False

    for key in ("dep_pending_questions", "metrics_pending_questions", "rec_pending_questions"):
        raw = state.get(key)
        if not raw:
            continue
        try:
            qs = json.loads(raw) if isinstance(raw, str) else raw
            if qs:
                return True
        except Exception:
            pass
    return False


def _find_questions(state: dict) -> list[dict]:
    """
    Return the structured questions list when the pipeline pauses for input.

    Parameters
    ----------
    state : dict
        Current ADK session state dict.

    Returns
    -------
    list of dict
        List of question dicts, each with keys ``question_id``, ``question``,
        and optionally ``options``. Returns an empty list if no questions are
        pending.

    Notes
    -----
    Reads from the authoritative pending-questions state keys written by each
    agent. Each key holds a JSON list of question dicts. The first non-empty
    list found is returned.
    """
    pending_keys = (
        "dep_pending_questions",
        "metrics_pending_questions",
        "rec_pending_questions",
    )
    for key in pending_keys:
        raw = state.get(key)
        if not raw:
            continue
        try:
            questions = json.loads(raw) if isinstance(raw, str) else raw
            if questions:
                return questions
        except Exception:
            continue
    return []


def _extract_report(state: dict, events: list[dict] | None = None) -> dict | None:
    """
    Pull the final SLO report from the session state or event text.

    Parameters
    ----------
    state : dict
        Current ADK session state dict.
    events : list of dict or None, optional
        Pipeline events for fallback text parsing.

    Returns
    -------
    dict or None
        The final SLO report dict, or ``None`` if not found.

    Notes
    -----
    Tries ``rec_report_output`` and ``final_slo`` state keys first. If
    neither is present, falls back to scanning reversed pipeline events for
    ``slo_report_agent`` or ``recommendation_agent`` events whose text may
    contain raw JSON, fenced JSON, or a JSON object embedded in prose.
    """
    def _parse_jsonish(raw: Any) -> dict | None:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None

        text = raw.strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
                fenced = "\n".join(lines[1:-1]).strip()
                if fenced.lower().startswith("json"):
                    fenced = fenced[4:].strip()
                try:
                    parsed = json.loads(fenced)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                pass

        return None

    for key in ("rec_report_output", "final_slo"):
        raw = state.get(key)
        if not raw:
            continue
        parsed = _parse_jsonish(raw)
        if parsed:
            return parsed
    if events:
        for ev in reversed(events):
            if ev.get("author", "") in ("slo_report_agent", "recommendation_agent"):
                content = ev.get("content") or {}
                if isinstance(content, dict):
                    for part in content.get("parts", []):
                        text = part.get("text", "") if isinstance(part, dict) else ""
                        parsed = _parse_jsonish(text)
                        if parsed:
                            return parsed
    return None


SAMPLE_GRAPH = {
    "services": [
        {
            "service": "api-gateway",
            "tier": "critical",
            "p99_latency_ms": 250.0,
            "depends_on": [
                {"name": "auth-service",     "dep_type": "synchronous", "weight": 1.0},
                {"name": "checkout-service", "dep_type": "synchronous", "weight": 1.0},
                {"name": "product-service",  "dep_type": "synchronous", "weight": 0.8},
            ],
        },
        {
            "service": "auth-service",
            "tier": "high",
            "p99_latency_ms": 50.0,
            "depends_on": [{"name": "user-db", "dep_type": "datastore", "weight": 1.0}],
        },
        {
            "service": "checkout-service",
            "tier": "critical",
            "p99_latency_ms": 800.0,
            "depends_on": [
                {"name": "payment-service",   "dep_type": "synchronous",  "weight": 1.0},
                {"name": "inventory-service", "dep_type": "synchronous",  "weight": 0.9},
                {"name": "order-events",      "dep_type": "asynchronous", "weight": 0.7},
            ],
        },
        {
            "service": "payment-service",
            "tier": "critical",
            "p99_latency_ms": 600.0,
            "depends_on": [
                {"name": "external-payment-api", "dep_type": "external",  "weight": 1.0},
                {"name": "payment-db",           "dep_type": "datastore", "weight": 1.0},
            ],
        },
        {"service": "product-service",    "tier": "medium", "p99_latency_ms": 80.0,  "depends_on": []},
        {"service": "inventory-service",  "tier": "medium", "p99_latency_ms": 120.0, "depends_on": []},
        {"service": "user-db",            "tier": "high",   "p99_latency_ms": 5.0,   "depends_on": []},
        {"service": "payment-db",         "tier": "high",   "p99_latency_ms": 4.0,   "depends_on": []},
        {"service": "external-payment-api","tier": "low",   "p99_latency_ms": 400.0, "depends_on": []},
        {"service": "order-events",       "tier": "medium", "p99_latency_ms": 15.0,  "depends_on": []},
    ],
    "edges": [
        {"from": "api-gateway",      "to": "auth-service",         "dep_type": "synchronous"},
        {"from": "api-gateway",      "to": "checkout-service",     "dep_type": "synchronous"},
        {"from": "checkout-service", "to": "payment-service",      "dep_type": "synchronous"},
        {"from": "checkout-service", "to": "inventory-service",    "dep_type": "synchronous"},
        {"from": "checkout-service", "to": "order-events",         "dep_type": "asynchronous"},
        {"from": "payment-service",  "to": "external-payment-api", "dep_type": "external"},
        {"from": "payment-service",  "to": "payment-db",           "dep_type": "datastore"},
        {"from": "auth-service",     "to": "user-db",              "dep_type": "datastore"},
    ],
}

with st.sidebar:
    st.title("SLO Engine")
    st.caption("Powered by Agents")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Ingest Graph", "Agent Pipeline", "Fast Recommendations", "Impact Analysis", "Error Budgets", "Human Review"],
        index=0,
    )
    st.divider()
    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:12]}...`")
        st.caption(f"State: `{st.session_state.pipeline_state}`")
        if st.button("Reset Session", use_container_width=True):
            for k in ["session_id", "pipeline_state", "awaiting_questions",
                      "pipeline_events", "pipeline_result", "last_service"]:
                st.session_state[k] = None if k in ("session_id", "pipeline_result", "last_service") else (
                    [] if k in ("pipeline_events", "awaiting_questions") else "idle")
            st.rerun()
    st.caption(f"API: {API_BASE}")


if page == "Dashboard":
    st.title("SLO Recommendation Engine")
    st.subheader("Service Health Overview")

    services_data = api_get("/services") or []
    if services_data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Services Tracked", len(services_data))
        col2.metric("Active SLOs", "-")
        col3.metric("Pending Reviews", "-")
        col4.metric("Avg PageRank", f"{sum(s.get('pagerank_score', 0) for s in services_data) / len(services_data):.4f}")

        st.divider()
        df = pd.DataFrame([{
            "Service":     s.get("name", ""),
            "Tier":        s.get("tier", "").capitalize(),
            "PageRank":    round(s.get("pagerank_score", 0), 4),
            "Display Name": s.get("display_name", ""),
        } for s in services_data])
        st.dataframe(df, use_container_width=True)

        fig = px.bar(
            df, x="Service", y="PageRank",
            color="Tier",
            title="Service PageRank Scores (criticality)",
            color_discrete_map={"Critical": "red", "High": "orange", "Medium": "steelblue", "Low": "gray"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No services ingested yet. Go to **Ingest Graph** to load the sample graph.")

        st.subheader("Sample Graph Preview")
        preview = [{"Service": s["service"], "Tier": s["tier"], "p99 (ms)": s["p99_latency_ms"],
                    "Deps": len(s["depends_on"])} for s in SAMPLE_GRAPH["services"]]
        st.dataframe(pd.DataFrame(preview), use_container_width=True)


elif page == "Ingest Graph":
    st.title("Ingest Service Dependency Graph")
    st.info("Paste your service graph JSON or use the sample e-commerce graph. "
            "The **edges** list (flat `from`/`to` format) is merged with inline `depends_on`.")

    col1, col2 = st.columns([2, 1])
    with col1:
        graph_json = st.text_area(
            "Service Graph JSON",
            value=json.dumps(SAMPLE_GRAPH, indent=2),
            height=450,
        )
    with col2:
        st.subheader("Edge Preview")
        try:
            parsed = json.loads(graph_json)
            edges = []
            for svc in parsed.get("services", []):
                for dep in svc.get("depends_on", []):
                    edges.append({"from": svc["service"], "to": dep["name"],
                                  "type": dep.get("dep_type", "sync"), "src": "depends_on"})
            for e in parsed.get("edges", []):
                edges.append({"from": e.get("from", e.get("source", "")),
                              "to": e.get("to", e.get("target", "")),
                              "type": e.get("dep_type", "sync"), "src": "edges[]"})
            if edges:
                st.dataframe(pd.DataFrame(edges), use_container_width=True)
            st.caption(f"{len(parsed.get('services', []))} services · {len(edges)} total edges")
        except Exception:
            st.warning("Invalid JSON")

    if st.button("Ingest Graph", type="primary"):
        with st.spinner("Ingesting dependency graph..."):
            result = api_post("/services/dependencies", json.loads(graph_json))
        if result:
            st.success(
                f"Ingested **{result.get('services_ingested')} services**, "
                f"**{result.get('edges')} edges**"
            )
            if result.get("circular_dependencies"):
                st.warning(f"Circular dependencies: {result['circular_dependencies']}")
            if result.get("critical_path"):
                cp = result["critical_path"]
                lat = result.get("critical_path_latency_ms", 0)
                st.info(f"Critical path: **{' -> '.join(cp)}** ({lat:.0f} ms)")
            st.json(result)


elif page == "Agent Pipeline":
    st.title("ADK Agent Pipeline")
    st.caption(
        "Full pipeline: **Dependency -> Metrics -> RAG -> Recommendation -> HITL gate**. "
        "Uses native Gemini 3 Flash Preview for reasoning. May pause mid-pipeline to ask you a question "
        "(e.g. tier assignment for a new service, cycle resolution)."
    )

    services_resp = api_get("/services") or []
    service_names = [s["name"] for s in services_resp] if services_resp else []

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if service_names:
            focus = st.selectbox("Focus Service", service_names)
        else:
            focus = st.text_input("Focus Service (type name)", value="recommendation-engine")
    with col_b:
        window = st.slider("Window (days)", 7, 90, 30)

    st.divider()

    _graph_services = api_get("/services/graph") or []
    if _graph_services:
        st.caption(
            f"Graph payload: **{len(_graph_services)} services** loaded from DB "
            f"(ingested via Ingest Graph page). The planner will decide whether "
            f"questions are needed based on this data."
        )
        with st.expander("Preview graph payload"):
            st.json(_graph_services)
    else:
        st.warning(
            "No services in the database yet. Go to **Ingest Graph** first to load a graph, "
            "then come back here to run the pipeline."
        )

    st.divider()

    state = st.session_state.pipeline_state

    if state == "done" and st.session_state.last_service and st.session_state.last_service != focus:
        st.session_state.pipeline_state  = "idle"
        st.session_state.pipeline_result = None
        st.session_state.pipeline_events = []
        st.session_state.session_id      = None
        state = "idle"

    if state == "idle":
        if st.button("Run Agent Pipeline", type="primary", use_container_width=True):
            with st.spinner("Creating session..."):
                sid = adk_create_session(
                    initial_state={"graph_payload": _graph_services}
                )
            if not sid:
                st.stop()

            st.session_state.session_id   = sid
            st.session_state.last_service = focus
            st.session_state.pipeline_state = "running"
            st.rerun()

    elif state == "running":
        svc = st.session_state.last_service
        sid = st.session_state.session_id

        message = (
            f"Run a full SLO analysis for {svc}. "
            f"Window: {window} days. "
            "Run dependency analysis, metrics analysis, and generate SLO recommendation. "
            "Use the full pipeline."
        )

        st.info(f"Running pipeline for **{svc}** (session `{sid[:12]}...`). "
                f"This takes ~1-2 minutes. Please wait.")

        progress = st.progress(0, text="Sending to ADK pipeline...")

        with st.spinner(f"Pipeline running (~{PIPELINE_TIMEOUT//60} min max)..."):
            t0 = time.time()
            events = adk_run(sid, message)
            elapsed = time.time() - t0

        progress.progress(80, text="Fetching session state...")
        sess_state = adk_get_state(sid)
        progress.progress(100, text="Done.")

        st.session_state.pipeline_events = events

        if _is_awaiting_input(events, sess_state):
            st.session_state.awaiting_questions = _find_questions(sess_state)
            st.session_state.pipeline_state     = "awaiting_input"
        else:
            all_events = st.session_state.get("pipeline_events", [])
            st.session_state.pipeline_result = _extract_report(sess_state, all_events) or {}
            st.session_state.pipeline_state  = "done"

        st.rerun()

    elif state == "awaiting_input":
        sid       = st.session_state.session_id
        questions = st.session_state.get("awaiting_questions") or []

        st.warning("Pipeline paused — please answer all questions below before continuing.")
        st.divider()

        if not questions:
            st.info("No structured questions found. Please type your answer.")
            free = st.text_area("Your answer", height=120)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit & Resume", type="primary", use_container_width=True):
                    if free.strip():
                        st.session_state.pipeline_state    = "running_resume"
                        st.session_state.awaiting_questions = []
                        st.session_state["_resume_answer"] = free.strip()
                        st.rerun()
            with col2:
                if st.button("Cancel Pipeline", use_container_width=True):
                    st.session_state.pipeline_state = "idle"
                    st.rerun()
        else:
            st.subheader(f"Please answer {len(questions)} question(s) to continue:")

            with st.form("questions_form"):
                selections: dict[str, str] = {}

                for i, q in enumerate(questions, 1):
                    q_id   = q.get("question_id", f"q{i}")
                    q_text = q.get("question", "")
                    opts   = q.get("options", [])

                    st.markdown(f"**Q{i}** — `{q_id}`")
                    st.markdown(q_text)

                    if opts:
                        chosen = st.radio(
                            "Select an option",
                            options=opts,
                            index=None,
                            key=f"radio_{q_id}_{i}",
                            label_visibility="collapsed",
                        )
                        selections[q_id] = chosen or ""
                    else:
                        val = st.text_input("Your answer", key=f"text_{q_id}_{i}")
                        selections[q_id] = val.strip()

                    st.divider()

                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button(
                        "Submit All Answers & Resume",
                        type="primary",
                        use_container_width=True,
                    )
                with col2:
                    cancelled = st.form_submit_button("Cancel Pipeline", use_container_width=True)

            if submitted:
                missing = [q.get("question_id", f"q{i}") for i, q in enumerate(questions, 1)
                           if not selections.get(q.get("question_id", f"q{i}"))]
                if missing:
                    st.error(f"Please answer all questions before submitting. Unanswered: {', '.join(missing)}")
                else:
                    combined = "\n".join(f"{qid}: {ans}" for qid, ans in selections.items())
                    st.session_state.pipeline_state     = "running_resume"
                    st.session_state.awaiting_questions = []
                    st.session_state["_resume_answer"]  = combined
                    st.rerun()
            elif cancelled:
                st.session_state.pipeline_state = "idle"
                st.rerun()

        with st.expander("Show pipeline events so far"):
            for author, text in _extract_text_from_events(st.session_state.pipeline_events):
                st.markdown(f"**[{author}]** {text[:800]}")

    elif state == "running_resume":
        sid    = st.session_state.session_id
        answer = st.session_state.get("_resume_answer", "")
        svc    = st.session_state.last_service or "service"

        st.info(f"Resuming pipeline for **{svc}** with your answers. Please wait ~1-2 minutes.")

        c1, c2, c3 = st.columns(3)
        c1.info("**Stage 1** — Dependency Analysis")
        c2.info("**Stage 2** — Metrics Analysis")
        c3.info("**Stage 3** — SLO Recommendation")

        with st.spinner("Running all 3 stages..."):
            events = adk_run(sid, answer)

        sess_state = adk_get_state(sid)
        st.session_state.pipeline_events.extend(events)

        if _is_awaiting_input(events, sess_state):
            st.session_state.awaiting_questions = _find_questions(sess_state)
            st.session_state.pipeline_state     = "awaiting_input"
        else:
            all_events = st.session_state.get("pipeline_events", [])
            st.session_state.pipeline_result = _extract_report(sess_state, all_events) or {}
            st.session_state.pipeline_state  = "done"

        st.rerun()

    elif state == "done":
        result = st.session_state.pipeline_result or {}
        svc    = st.session_state.last_service or "unknown"

        col_hdr, col_btn = st.columns([4, 1])
        with col_hdr:
            st.success(f"Pipeline complete for **{svc}**")
        with col_btn:
            if st.button("New Run", use_container_width=True):
                st.session_state.pipeline_state  = "idle"
                st.session_state.pipeline_result = None
                st.session_state.pipeline_events = []
                st.session_state.session_id      = None
                st.rerun()
        st.divider()

        st.subheader("Final SLO Recommendation")

        avail  = result.get("recommended_availability")
        lat    = result.get("recommended_latency_p99_ms")
        conf   = result.get("confidence_score")
        feas   = result.get("is_feasible")
        review = result.get("requires_human_review")

        if avail or lat:
            c1, c2, c3, c4 = st.columns(4)
            if avail is not None:
                c1.metric("Availability SLO", f"{float(avail):.4%}")
            if lat is not None:
                c2.metric("Latency p99", f"{float(lat):.0f} ms")
            if conf is not None:
                c3.metric("Confidence", f"{float(conf):.1%}",
                          delta="auto-approved" if not review else "review required")
            if feas is not None:
                c4.metric("Feasible", "Yes" if feas else "No")
        else:
            st.warning("Could not extract SLO metrics from the report.")

        summary = result.get("summary", "")
        if summary:
            st.info(summary)

        review_reason = result.get("review_reason", "")
        if review and review_reason:
            st.warning(f"Review required: {review_reason}")

        reasoning = result.get("reasoning", "")
        if reasoning:
            with st.expander("Reasoning"):
                st.write(reasoning)

        sources = result.get("sources", result.get("data_sources", []))
        if sources:
            st.caption(f"Knowledge sources: {', '.join(sources)}")

        knowledge_context = result.get("knowledge_context", "")
        if knowledge_context:
            with st.expander("Knowledge Context"):
                st.write(knowledge_context)

        st.divider()

        st.subheader("Agent Pipeline Responses")

        _AGENT_LABELS = {
            "slo_router_decision_agent": "Router",
            "dependency_planner_agent":  "Dependency Planner",
            "dependency_agent":          "Dependency Agent",
            "dependency_report_agent":   "Dependency Report",
            "metrics_report_agent":      "Metrics Report",
            "slo_report_agent":          "SLO Report",
            "recommendation_agent":      "Recommendation",
            "slo_router":                "Pipeline Summary",
        }

        seen = set()
        for author, text in _extract_text_from_events(st.session_state.pipeline_events):
            if not text.strip():
                continue
            label = _AGENT_LABELS.get(author, author)
            key   = f"{author}:{text[:60]}"
            if key in seen:
                continue
            seen.add(key)

            parsed = _extract_report({}, [{"author": author, "content": {"parts": [{"text": text}]}}])
            if parsed:
                with st.expander(f"**{label}**", expanded=(author in ("slo_report_agent", "recommendation_agent", "slo_router"))):
                    st.json(parsed)
                continue

            with st.expander(f"**{label}**", expanded=(author in ("recommendation_agent", "slo_router"))):
                st.markdown(text[:2000])

        pareto = result.get("pareto_optimal_slos", {})
        if pareto and len(pareto) > 1:
            st.divider()
            st.subheader("Portfolio SLOs (MILP Pareto)")
            pareto_df = pd.DataFrame([
                {"Service": k, "Optimal SLO": v} for k, v in pareto.items()
            ])
            fig = px.bar(pareto_df, x="Service", y="Optimal SLO",
                         title="MILP-Optimal SLO Targets",
                         range_y=[0.98, 1.001])
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full result JSON"):
            st.json(result)

        st.divider()
        if st.button("Run Another", use_container_width=True):
            st.session_state.pipeline_state  = "idle"
            st.session_state.pipeline_result = None
            st.session_state.pipeline_events = []
            st.session_state.session_id      = None
            st.rerun()

    elif state == "error":
        st.error("Pipeline encountered an error. Check the API server logs.")
        if st.button("Reset", use_container_width=True):
            st.session_state.pipeline_state  = "idle"
            st.session_state.pipeline_result = None
            st.session_state.pipeline_events = []
            st.session_state.session_id      = None
            st.rerun()

    elif state not in ("idle", "running", "awaiting_input", "running_resume", "done", "error"):
        st.warning(f"Unknown pipeline state: `{state}`. Resetting.")
        st.session_state.pipeline_state = "idle"
        st.rerun()


elif page == "Fast Recommendations":
    st.title("Fast SLO Recommendations")
    st.caption(
        "Runs Bayesian math + MILP optimization directly — **no LLM**, instant results. "
        "For LLM reasoning and NEEDS_INPUT handling use the **Agent Pipeline** page."
    )

    focus = st.selectbox(
        "Focus Service (optional)",
        ["All Services", "api-gateway", "checkout-service", "auth-service",
         "payment-service", "inventory-service"],
    )
    window = st.slider("Observation Window (days)", 7, 90, 30)

    if st.button("Generate", type="primary"):
        payload = {
            "services": SAMPLE_GRAPH["services"],
            "focus_service": None if focus == "All Services" else focus,
            "window_days": window,
        }
        with st.spinner("Running computation pipeline..."):
            result = api_post("/recommendations/bulk", payload, timeout=120)

        if result:
            services_analysed = result.get("services_analysed", [])
            st.success(f"Recommendations for {len(services_analysed)} services")

            graph_sum = result.get("graph_summary", {})
            if graph_sum.get("critical_path"):
                cp  = graph_sum["critical_path"]
                lat = graph_sum.get("critical_path_latency_ms", 0)
                st.info(f"Critical path: **{' -> '.join(cp)}** ({lat:.0f} ms)")
            if graph_sum.get("circular_deps"):
                st.warning(f"Circular deps detected: {graph_sum['circular_deps']}")

            recs = result.get("recommendations", {})
            if recs:
                rows = []
                for svc, rec in recs.items():
                    if isinstance(rec, dict):
                        rows.append({
                            "Service":         svc,
                            "Availability SLO": f"{rec.get('recommended_availability', 0):.4f}",
                            "p99 Latency (ms)": rec.get("recommended_latency_p99_ms", "-"),
                            "Confidence":       f"{rec.get('confidence_score', 0):.1%}",
                            "Review Required":  "Yes" if rec.get("requires_human_review") else "No",
                            "Sources":          ", ".join(rec.get("sources", rec.get("data_sources", []))),
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                conf_vals = {s: r.get("confidence_score", 0) for s, r in recs.items()
                             if isinstance(r, dict)}
                fig = px.bar(
                    x=list(conf_vals.keys()), y=list(conf_vals.values()),
                    title="Confidence Scores",
                    labels={"x": "Service", "y": "Confidence"},
                    color=list(conf_vals.values()), color_continuous_scale="RdYlGn",
                    range_y=[0, 1],
                )
                fig.add_hline(y=0.75, line_dash="dash",
                              annotation_text="HITL threshold (0.75)", line_color="orange")
                st.plotly_chart(fig, use_container_width=True)

            portfolio = result.get("portfolio_optimization", {})
            optimal   = portfolio.get("optimal_slos", {}) if isinstance(portfolio, dict) else {}
            if optimal:
                st.subheader("MILP Portfolio Optimization")
                opt_df = pd.DataFrame([{"Service": k, "Optimal SLO": v}
                                       for k, v in optimal.items()])
                fig2 = px.bar(opt_df, x="Service", y="Optimal SLO",
                              title="MILP-Optimal SLO Targets",
                              range_y=[0.98, 1.001])
                st.plotly_chart(fig2, use_container_width=True)


elif page == "Impact Analysis":
    st.title("Impact Analysis")
    st.caption(
        "Tests how a proposed service SLO change affects feasibility and upstream services. "
        "This page calls the backend `POST /api/v1/slos/impact-analysis` endpoint directly."
    )

    live_services = api_get("/services") or []
    svc_names = [s["name"] for s in live_services] if live_services else [
        "api-gateway", "checkout-service", "auth-service", "payment-service"
    ]

    c1, c2, c3 = st.columns(3)
    with c1:
        service_name = st.selectbox("Service", svc_names, index=0)
    with c2:
        proposed_availability = st.slider(
            "Proposed Availability",
            min_value=0.9000,
            max_value=0.9999,
            value=0.9950,
            step=0.0001,
            format="%.4f",
        )
    with c3:
        proposed_latency = st.number_input(
            "Proposed p99 Latency (ms)",
            min_value=10.0,
            max_value=10000.0,
            value=500.0,
            step=10.0,
        )

    if st.button("Run Impact Analysis", type="primary"):
        payload = {
            "service_name": service_name,
            "proposed_availability": proposed_availability,
            "proposed_latency_p99_ms": proposed_latency,
        }
        with st.spinner("Computing impact analysis..."):
            result = api_post("/slos/impact-analysis", payload, timeout=60)

        if result and isinstance(result, dict):
            st.success(f"Impact analysis complete for **{service_name}**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Feasible", "Yes" if result.get("feasible") else "No")
            m2.metric("Feasibility Score", f"{float(result.get('feasibility_score', 0.0)):.1%}")
            ceiling = result.get("availability_ceiling")
            m3.metric("Availability Ceiling", f"{float(ceiling):.4f}" if ceiling is not None else "-")
            hist = result.get("historical_availability")
            m4.metric("Historical Availability", f"{float(hist):.4f}" if hist is not None else "-")

            blocking = result.get("blocking_reason", [])
            if blocking:
                st.warning("Blocking reasons: " + "; ".join(str(x) for x in blocking))

            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("Dependency Context")
                st.write("Sync dependencies:", result.get("sync_dependencies", []))
                st.write("Upstream affected:", result.get("upstream_affected", []))

                dep_avails = result.get("dep_availabilities", {})
                if dep_avails:
                    dep_df = pd.DataFrame([
                        {"Dependency": k, "Historical Availability": v}
                        for k, v in dep_avails.items()
                    ])
                    st.dataframe(dep_df, use_container_width=True)

            with c_right:
                st.subheader("Cascade Impact")
                cascade = result.get("cascade", {})
                if cascade:
                    cascade_df = pd.DataFrame([
                        {
                            "Upstream Service": upstream,
                            "Current Availability": details.get("current_availability"),
                            "New Ceiling": details.get("new_availability_ceiling"),
                            "Impact": details.get("impact"),
                        }
                        for upstream, details in cascade.items()
                    ])
                    st.dataframe(cascade_df, use_container_width=True)

                    fig = px.bar(
                        cascade_df,
                        x="Upstream Service",
                        y="New Ceiling",
                        color="Impact",
                        title="Upstream Availability Ceilings After Proposed Change",
                        range_y=[0.90, 1.0],
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No upstream cascade impact detected for this service.")

            with st.expander("Full impact-analysis JSON"):
                st.json(result)


elif page == "Error Budgets":
    st.title("Error Budget Status")

    live_services = api_get("/services") or []
    svc_names = [s["name"] for s in live_services] if live_services else [
        "api-gateway", "checkout-service", "auth-service", "payment-service"
    ]

    selected = st.multiselect("Services to display", svc_names, default=svc_names[:4])
    slo_target = st.slider("SLO Target", 0.990, 0.9999, 0.999, step=0.0001, format="%.4f")

    if not selected:
        st.info("Select at least one service.")
        st.stop()

    st.divider()
    cols = st.columns(min(len(selected), 4))

    for col, svc in zip(cols, selected):
        rec_resp = api_get(f"/services/{svc}/slo-recommendations?slo_target={slo_target:.4f}")
        if rec_resp and isinstance(rec_resp, dict):
            m = rec_resp.get("metrics_summary", {})
            b = rec_resp.get("budget_summary", {})
            burn_1h = m.get("burn_rate_1h", 1.0)
            burn_6h = m.get("burn_rate_6h", 1.0)
            burn_frac = float(b.get("burn_fraction", 0.0))
            status = str(b.get("status", "healthy"))
            days_to_exhaustion = b.get("days_to_exhaustion")
            prob_exhaust = b.get("prob_exhaust_in_window")
        else:
            import random; random.seed(hash(svc) % 100)
            burn_1h = round(random.uniform(0.3, 8.0), 2)
            burn_6h = round(random.uniform(0.3, 6.0), 2)
            burn_frac = min(burn_1h / 14.4, 1.0)
            status = "critical" if burn_frac > 0.80 else "warning" if burn_frac > 0.50 else "healthy"
            days_to_exhaustion = None
            prob_exhaust = None
        with col:
            st.subheader(svc[:20])
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(burn_frac * 100, 1),
                title={"text": "Budget Used %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "darkred" if status == "critical"
                             else "orange" if status == "warning" else "green"},
                    "steps": [
                        {"range": [0, 50],  "color": "lightgreen"},
                        {"range": [50, 80], "color": "lightyellow"},
                        {"range": [80, 100],"color": "lightcoral"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4},
                                  "thickness": 0.75, "value": 90},
                },
                domain={"x": [0, 1], "y": [0, 1]},
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{svc}")
            st.caption(f"Burn 1h: **{burn_1h:.2f}x** · 6h: **{burn_6h:.2f}x**")
            if days_to_exhaustion is not None and prob_exhaust is not None:
                st.caption(
                    f"Status: **{status}** · Days to exhaustion: **{days_to_exhaustion}** "
                    f"· P(exhaust): **{float(prob_exhaust):.1%}**"
                )

    st.divider()
    st.subheader("Multi-Window Burn Rate — SRE Alert Thresholds")
    burn_data_rows = []
    for svc in selected:
        rec_resp = api_get(f"/services/{svc}/slo-recommendations?slo_target={slo_target:.4f}")
        if rec_resp and isinstance(rec_resp, dict):
            m = rec_resp.get("metrics_summary", {})
            for window_label, val_key in [("1h", "burn_rate_1h"), ("6h", "burn_rate_6h")]:
                burn_data_rows.append({"Window": window_label, "Service": svc,
                                       "Burn Rate": m.get(val_key, 1.0)})
        else:
            import random; random.seed(hash(svc) % 100)
            for window_label in ["1h", "6h"]:
                burn_data_rows.append({"Window": window_label, "Service": svc,
                                       "Burn Rate": round(random.uniform(0.5, 7.0), 2)})

    if burn_data_rows:
        fig_burn = px.line(
            pd.DataFrame(burn_data_rows), x="Window", y="Burn Rate", color="Service",
            title="Error Budget Burn Rate by Window",
            markers=True,
        )
        fig_burn.add_hline(y=1.0,  line_dash="dot",  annotation_text="Normal (1x)",          line_color="green")
        fig_burn.add_hline(y=6.0,  line_dash="dash", annotation_text="Slow-burn alert (6x)", line_color="orange")
        fig_burn.add_hline(y=14.4, line_dash="dash", annotation_text="Fast-burn alert (14.4x)", line_color="red")
        st.plotly_chart(fig_burn, use_container_width=True)


elif page == "Human Review":
    st.title("Human-in-the-Loop Review")
    st.caption(
        "Post-pipeline review queue for **low-confidence** recommendations (confidence < 0.75 "
        "or drift detected). This is separate from mid-pipeline NEEDS_INPUT questions "
        "(handled in the Agent Pipeline page)."
    )

    pending_raw = api_get("/reviews/pending")
    if isinstance(pending_raw, list):
        pending = {"count": len(pending_raw), "pending_reviews": pending_raw}
    elif isinstance(pending_raw, dict):
        pending = pending_raw
    else:
        pending = {"count": 0, "pending_reviews": []}

    if pending.get("count", 0) > 0:
        st.metric("Pending Reviews", pending["count"])
        st.divider()
        for review in pending.get("pending_reviews", []):
            rid  = review.get("recommendation_id", "unknown")
            svc  = review.get("service_name", "unknown")
            conf = review.get("confidence_score", 0)
            with st.expander(
                f"{'🔴' if conf < 0.5 else '🟡'} {svc} — Confidence: {conf:.1%} "
                f"| {review.get('time_remaining_seconds', '?')}s remaining"
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Recommended Availability", f"{review.get('recommended_availability', 0):.4f}")
                c2.metric("Recommended p99 (ms)",     f"{review.get('recommended_latency_p99_ms', 0):.0f}")
                c3.metric("Confidence",               f"{conf:.1%}")

                reason = review.get("review_reason", "Low confidence")
                st.caption(f"Review reason: {reason}")

                reviewer = st.text_input("Your name / email", key=f"rev_{rid}")
                decision = st.radio("Decision", ["approve", "reject", "modify"], key=f"dec_{rid}",
                                    horizontal=True)
                comment  = st.text_area("Comment (optional)", key=f"cmt_{rid}")

                modified_avail = modified_lat = None
                if decision == "modify":
                    c4, c5 = st.columns(2)
                    modified_avail = c4.number_input(
                        "Modified Availability", 0.9, 0.9999,
                        value=float(review.get("recommended_availability", 0.999)),
                        format="%.4f", key=f"ma_{rid}",
                    )
                    modified_lat = c5.number_input(
                        "Modified p99 Latency (ms)", 10.0, 10000.0,
                        value=float(review.get("recommended_latency_p99_ms", 200)),
                        key=f"ml_{rid}",
                    )

                if st.button("Submit Decision", key=f"sub_{rid}", type="primary"):
                    payload = {
                        "recommendation_id":      rid,
                        "decision":               decision,
                        "reviewer":               reviewer or "anonymous",
                        "comment":                comment,
                        "modified_availability":  modified_avail,
                        "modified_latency_p99_ms": modified_lat,
                    }
                    res = api_post(f"/reviews/{rid}/decision", payload)
                    if res:
                        st.success(f"Decision submitted: **{decision}**")
                        time.sleep(0.5)
                        st.rerun()
    else:
        st.success("No pending reviews. All recommendations are auto-approved or already reviewed.")
        st.info("Low-confidence or drifting recommendations from the Agent Pipeline will appear here.")
