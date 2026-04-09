"""
Interactive end-to-end SLO pipeline runner script.

Notes
-----
Starts the SLO pipeline, prints each question when it pauses for human input,
reads answers from the terminal, and continues until the pipeline completes
or the maximum number of turns is reached.

Usage:
    python scripts/interactive_run.py
"""
import json
import sys
import time
import requests
from pathlib import Path

BASE     = "http://localhost:8000"
APP_NAME = "slo_pipeline"
USER_ID  = "interactive-user"
ROOT     = Path(__file__).resolve().parents[1]


def get(path, **kw):
    """
    Perform an HTTP GET request against the local API server.

    Parameters
    ----------
    path : str
        URL path relative to ``BASE``.
    **kw : dict
        Additional keyword arguments forwarded to ``requests.get``.

    Returns
    -------
    dict
        Parsed JSON response body.

    Notes
    -----
    Raises ``requests.HTTPError`` if the response status code indicates failure.
    Default timeout is 30 seconds.
    """
    r = requests.get(f"{BASE}{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json()

def post(path, body=None, **kw):
    """
    Perform an HTTP POST request against the local API server.

    Parameters
    ----------
    path : str
        URL path relative to ``BASE``.
    body : dict or None
        JSON-serialisable request body. Defaults to ``None``.
    **kw : dict
        Additional keyword arguments forwarded to ``requests.post``.

    Returns
    -------
    dict
        Parsed JSON response body.

    Notes
    -----
    Raises ``requests.HTTPError`` if the response status code indicates failure.
    Default timeout is 300 seconds to accommodate long-running pipeline calls.
    """
    r = requests.post(f"{BASE}{path}", json=body, timeout=300, **kw)
    r.raise_for_status()
    return r.json()

def get_state(session_id):
    """
    Retrieve the current ADK session state for a given session ID.

    Parameters
    ----------
    session_id : str
        ADK session identifier.

    Returns
    -------
    dict
        Session state dictionary, or an empty dict if no state is present.

    Notes
    -----
    Calls the ADK sessions endpoint and extracts the ``state`` field.
    """
    return get(f"/apps/{APP_NAME}/users/{USER_ID}/sessions/{session_id}").get("state", {})

def send_message(session_id, text):
    """
    Send a user message to the ADK pipeline for a given session.

    Parameters
    ----------
    session_id : str
        ADK session identifier.
    text : str
        Message text to send as the user turn.

    Returns
    -------
    dict
        ADK run response dictionary.

    Notes
    -----
    Posts to the ADK ``/run`` endpoint with the user message wrapped in the
    expected ADK message format.
    """
    return post("/run", {
        "app_name": APP_NAME,
        "user_id":  USER_ID,
        "session_id": session_id,
        "new_message": {"role": "user", "parts": [{"text": text}]},
    })


GRAPH = {
    "services": [
        {"service": "order-orchestrator", "tier": "critical", "depends_on": [
            {"name": "inventory-checker", "dep_type": "synchronous", "weight": 1.0},
            {"name": "payment-processor", "dep_type": "synchronous", "weight": 1.0},
            {"name": "fraud-service",     "dep_type": "synchronous", "weight": 1.0},
        ]},
        {"service": "payment-processor", "tier": "high", "depends_on": [
            {"name": "stripe-gateway",  "dep_type": "external",     "weight": 1.0},
            {"name": "payment-db",      "dep_type": "synchronous",  "weight": 1.0},
            {"name": "paypal-gateway",  "dep_type": "external",     "weight": 1.0},
        ]},
        {"service": "fraud-service", "tier": "medium", "depends_on": [
            {"name": "order-orchestrator", "dep_type": "synchronous", "weight": 1.0},
        ]},
        {"service": "inventory-checker", "tier": "high", "depends_on": [
            {"name": "warehouse-db",  "dep_type": "synchronous", "weight": 1.0},
            {"name": "supplier-api",  "dep_type": "external",    "weight": 1.0},
        ]},
        {"service": "stripe-gateway",  "tier": "medium", "depends_on": []},
        {"service": "paypal-gateway",  "tier": "medium", "depends_on": []},
        {"service": "payment-db",      "tier": "high",   "depends_on": []},
        {"service": "warehouse-db",    "tier": "medium", "depends_on": []},
        {"service": "supplier-api",    "tier": "medium", "depends_on": []},
    ],
    "edges": [],
}

print("\n=== SLO Interactive Pipeline Runner ===\n")
print("Step 1: Ingesting dependency graph...")
r = post("/api/v1/services/dependencies", GRAPH)
print(f"  Ingested: {r['services_ingested']} services, {r['edges']} edges")
print(f"  Circular deps: {r['circular_dependencies']}")

print("\nStep 2: Fetching graph from DB...")
graph_payload = get("/api/v1/services/graph")
print(f"  Graph: {len(graph_payload)} services")

print("\nStep 3: Creating session...")
DEP_SLOS = {
    "payment-processor":  {"recommended_availability": 0.95,  "dep_type": "synchronous"},
    "inventory-checker":  {"recommended_availability": 0.995, "dep_type": "synchronous"},
    "fraud-service":      {"recommended_availability": 0.99,  "dep_type": "asynchronous"},
}
sess = post(f"/apps/{APP_NAME}/users/{USER_ID}/sessions", {"state": {
    "graph_payload": graph_payload,
    "rec_dep_slos":  DEP_SLOS,
}})
session_id = sess["id"]
print(f"  Session: {session_id}")
print(f"  Pre-seeded dep_slos: {list(DEP_SLOS.keys())}")

# Use "order-orchestrator" (in graph → no AWAIT_INPUT pause).
# To test AWAIT_INPUT, change to a service NOT in the graph, e.g. "new-analytics-service".
TARGET_SERVICE = "order-orchestrator"

print(f"\nStep 4: Starting pipeline for {TARGET_SERVICE}...")
print("  (This will take ~30s for the planner to run...)\n")
send_message(session_id, f"Run full SLO pipeline for {TARGET_SERVICE}")


for turn in range(1, 30):
    time.sleep(5)
    state = get_state(session_id)

    workflow = {k: v for k, v in state.items() if "workflow_step" in k}
    awaiting = any("AWAIT_INPUT" in str(v) for v in workflow.values())
    pending  = {k: v for k, v in state.items() if "pending_questions" in k and v}

    final    = state.get("final_slo") or state.get("rec_report_output")

    if final:
        print("\n" + "="*60)
        print("PIPELINE COMPLETE — FINAL SLO:")
        print("="*60)
        try:
            print(json.dumps(json.loads(final), indent=2))
        except Exception:
            print(final)
        print("\nFull agent trace in:", ROOT / "logs")
        sys.exit(0)

    if not awaiting and not pending:
        print(f"  [turn {turn}] Pipeline still running... workflow={workflow}")
        continue

    print("\n" + "="*60)
    print("PIPELINE PAUSED — HUMAN INPUT REQUIRED")
    print("="*60)

    all_answers = []

    for pk, raw in pending.items():
        questions = json.loads(raw) if isinstance(raw, str) else raw
        print(f"\nQuestions from [{pk}]:\n")
        for q in questions:
            qid  = q.get("question_id", "?")
            text = q.get("question", "")
            opts = q.get("options", [])
            print(f"  [{qid}]")
            print(f"  {text}")
            for j, opt in enumerate(opts, 1):
                print(f"    {j}) {opt}")
            answer = input(f"\n  Your answer for [{qid}]: ").strip()
            all_answers.append(f"{qid}: {answer}")
            print()

    if not all_answers:
        print("  No questions found, waiting...")
        continue

    combined = "\n".join(all_answers)
    print("\nSubmitting answers:\n" + combined)
    print("\nWaiting for pipeline to continue...\n")
    send_message(session_id, combined)

print("\nPipeline did not complete within expected turns.")
