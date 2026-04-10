# SLO Recommendation Engine

An AI-assisted system that analyses service dependencies and service metrics, retrieves grounding context from an MCP-backed knowledge base, and recommends SLOs with a human-review gate.

Built with Google ADK agents, native Gemini via Google ADK, a stdio Knowledge MCP server, ChromaDB + sentence-transformers + adaptive MMR, deterministic Python computation/optimization tools, and FastAPI + Streamlit.

## System Architecture

```text
+----------------------------------------------------------------------------------+
|                           DEVELOPER PLATFORM / UI                                |
|  Streamlit UI | Backstage / Port / Cortex / custom portal | webhooks / polling   |
+----------------------------------------------+-----------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------+
|                               API LAYER (FastAPI)                                |
|  /api/v1/services/dependencies  /api/v1/recommendations/bulk                     |
|  /api/v1/reviews/*              /api/v1/catalog/ingest                           |
|  /api/v1/webhooks/*             /.well-known/agent.json                          |
|  ADK endpoints: /apps/.../sessions, /run                                         |
+----------------------------------------------+-----------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------+
|                           ADK APP: slo_pipeline                                  |
|                                SLORouterAgent                                    |
|                 route -> dependency -> metrics -> recommendation -> gate          |
+---------------------------+-------------------------+----------------------------+
                            |                         |
                            v                         v
                 +-----------------------+   +-----------------------+
                 | DependencyOrchestrator|   |  MetricsOrchestrator |
                 | - PLAN / AWAIT_INPUT  |   | - QUERY / AWAIT_INPUT|
                 | - INGEST              |   | - ANALYZE / ANOMALY  |
                 | - CYCLES              |   | - BUDGET / REPORT    |
                 | - REPORT              |   +-----------------------+
                 +-----------+-----------+
                             \                        
                              \                       
                               v
                 +------------------------------------+
                 | RecommendationOrchestratorAgent    |
                 | - GENERATE                         |
                 | - FEASIBILITY                      |
                 | - OPTIMIZE                         |
                 | - REPORT                           |
                 +-----------+---------------+--------+
                             |               |
                             |               +--> deterministic Python computation tools
                             |
                             v
                 +------------------------------------+
                 | KnowledgeMCPClient                 |
                 | persistent stdio MCP client        |
                 +----------------+-------------------+
                                  |
                                  v
                 +------------------------------------+
                 | knowledge_mcp_server               |
                 | MCP tool boundary                  |
                 +----------------+-------------------+
                                  |
                                  v
                 +------------------------------------+
                 | KnowledgeStore.retrieve()          |
                 | Chroma ANN + adaptive MMR          |
                 +------------------------------------+
                                               |
                                               v
                                 inline confidence gate + review queue
```

## Agent Workflow

```text
User / UI
-> SLORouterAgent
   -> RouterDecisionAgent chooses route when the pipeline is not resuming
-> DependencyOrchestratorAgent
   -> DependencyPlannerAgent
      -> PLAN
   -> AWAIT_INPUT when assumptions are missing
   -> direct Python tools
      -> ingest_service_dependencies(...)
      -> analyse_dependency_graph(...)
      -> detect_circular_dependencies(...)
   -> DependencyReportAgent
-> MetricsOrchestratorAgent
   -> direct Python tools
      -> query_service_metrics(...)
      -> detect_metric_anomaly(...)
      -> compute_error_budget_status(...)
   -> AWAIT_INPUT when assumptions are missing
   -> MetricsReportAgent
-> RecommendationOrchestratorAgent
   -> GENERATE
      -> deterministic knowledge query build
      -> KnowledgeMCPClient
      -> knowledge_mcp_server
      -> KnowledgeStore.retrieve()
      -> Chroma ANN + adaptive MMR
      -> generate_slo_recommendation(...)
   -> FEASIBILITY
      -> check_slo_feasibility(...)
   -> OPTIMIZE
      -> run_milp_optimization(...)
   -> REPORT
      -> SLOReportAgent
-> inline confidence gate
-> auto-approve or queue human review
```

## RAG - Knowledge Retrieval

RAG happens through the Knowledge MCP server.

Actual implementation path:

```text
RecommendationOrchestratorAgent
  -> reads service + tier + drift + anomaly context
  -> builds the query deterministically in Python
  -> calls knowledge_client.retrieve_knowledge(...)
  -> stdio MCP call to knowledge_mcp_server
  -> knowledge_mcp_server calls knowledge_store.retrieve(...)
  -> KnowledgeStore runs:
       1. Chroma ANN candidate retrieval
       2. adaptive MMR reranking
  -> returns:
       - results
       - source_ids
       - context_summary
       - total_returned
       - kb_size
  -> orchestrator passes knowledge_context + knowledge_sources into generation
  -> final report includes sources + knowledge_context
```

What this means:
- The LLM does not call the knowledge tool directly.
- The recommendation stage still uses the Knowledge MCP server as an MCP tool boundary.
- MMR is still active.
- Retrieved knowledge is injected before final report synthesis.

### MCP Deployment Model

Current runtime model:
- `knowledge_mcp_server` is not exposed as a separate HTTP or gRPC service
- it is spawned by `KnowledgeMCPClient` as a local stdio subprocess
- the subprocess runs in the same deployment unit as the API worker
- ChromaDB remains the shared retrieval backend behind that subprocess

Production implication:
- deploy the MCP server code inside the same API image
- let each API worker or pod spawn and own its own MCP subprocess
- do not create a separate ingress or Kubernetes service for the MCP server
- scale API replicas normally; MCP subprocess count scales with API worker count

Operational path:

```text
API worker
-> KnowledgeMCPClient
-> stdio-spawned knowledge_mcp_server subprocess
-> KnowledgeStore.retrieve()
-> ChromaDB
```

Knowledge sources:
- `data/knowledge_base/*.json`
- `docs/*.md`

## Mid-Pipeline Human Input (NEEDS_INPUT / AWAIT_INPUT)

The pipeline can pause mid-run when critical assumptions are missing.

Typical examples:
- missing external SLA assumptions
- unresolved sync cycle handling
- missing latency baseline assumptions
- missing tier assignment

Current flow:

```text
stage runs
-> planner/tool discovers missing assumption
-> pending questions are written to session state
-> pipeline returns control
-> UI reads pending question keys
-> user answers
-> /run is called again on the same ADK session
-> orchestrator resumes from the saved workflow step
```

This is how the UI and ADK session resumability work together in the current codebase.

## Agent Architecture

### Router

`SLORouterAgent` in `src/slo_engine/agents/agent.py` is the top-level orchestrator.

Responsibilities:
- choose the route using `RouterDecisionAgent` when not resuming an in-flight session
- run `dependency -> metrics -> recommendation` for `full_pipeline`
- run the final inline gate

Actual sub-agent call:
- `RouterDecisionAgent`

### Dependency Agent

Implemented by `DependencyOrchestratorAgent`.

Workflow:
- `DependencyPlannerAgent`
- `AWAIT_INPUT`
- direct tools:
  `ingest_service_dependencies(...)`, `analyse_dependency_graph(...)`, `detect_circular_dependencies(...)`
- `DependencyReportAgent`
- `AWAIT_INPUT` when needed
- `DONE`

Main algorithms:
- PageRank
- Tarjan SCC
- critical-path / latency analysis
- blast-radius / centrality analysis

Actual sub-agent calls:
- `DependencyPlannerAgent`
- `DependencyReportAgent`

Direct deterministic calls:
- `ingest_service_dependencies(...)`
- `analyse_dependency_graph(...)`
- `detect_circular_dependencies(...)`

Main outputs:
- `dep_plan_output`
- `dep_analysis_output`
- `dep_cycles_output`
- `dep_report_output`

### Metrics Agent

Implemented by `MetricsOrchestratorAgent`.

Workflow:
- direct tools:
  `query_service_metrics(...)`, `detect_metric_anomaly(...)`, `compute_error_budget_status(...)`
- `AWAIT_INPUT` when needed
- `MetricsReportAgent`
- `DONE`

Main methods:
- Beta-Binomial posterior
- Kalman smoothing
- KL-divergence drift detection
- anomaly detection
- burn-rate / budget status analysis

Actual sub-agent call:
- `MetricsReportAgent`

Direct deterministic calls:
- `query_service_metrics(...)`
- `detect_metric_anomaly(...)`
- `compute_error_budget_status(...)`

Main outputs:
- `metrics_query_output`
- `metrics_anomaly_output`
- `metrics_budget_output`
- `metrics_report_output`

### Recommendation Agent

Implemented by `RecommendationOrchestratorAgent`.

Workflow:
- `GENERATE`
- `FEASIBILITY`
- `OPTIMIZE`
- `SLOReportAgent`
- `DONE`

Real `GENERATE` step:

```text
load dependency + metrics outputs
-> construct deterministic knowledge query
-> call KnowledgeMCPClient
-> retrieve knowledge from Knowledge MCP server
-> call generate_slo_recommendation(...) tool
-> store deterministic numeric recommendation result
```

Then:
- `FEASIBILITY` checks caps and constraints
- `OPTIMIZE` runs portfolio MILP
- `REPORT` delegates to `SLOReportAgent` over already-computed outputs

Important distinction:
- retrieval and numeric recommendation logic are deterministic
- the final report is synthesized by the LLM from deterministic upstream results

Actual sub-agent call:
- `SLOReportAgent`

Direct deterministic calls:
- `KnowledgeMCPClient.retrieve_knowledge(...)`
- `generate_slo_recommendation(...)`
- `check_slo_feasibility(...)`
- `run_milp_optimization(...)`

### Confidence Gate

The gate is inline in `SLORouterAgent`, not its own agent.

Current implementation:
- if `requires_human_review` is true, queue review
- if drift is detected, queue review
- otherwise auto-approve

Outputs:
- `final_slo`
- optional webhook push
- review-store entry when human review is required
- audit log / Opik trace

## Post-Pipeline HITL Review

Review endpoints are exposed at:
- `GET /api/v1/reviews/pending`
- `GET /api/v1/reviews/{recommendation_id}`
- `POST /api/v1/reviews/{recommendation_id}/decision`
- `GET /api/v1/reviews/feedback/summary`

Review decisions support:
- approve
- reject
- modify

The Streamlit Human Review page uses these endpoints directly.

## Impact Analysis

`POST /api/v1/slos/impact-analysis` evaluates a proposed service SLO change and computes cascade impact on upstream services.

It checks:
- feasibility against history
- dependency ceilings
- upstream propagation effects

This is separate from the ADK multi-agent pipeline and runs through direct tool-layer logic.

## Future Enhancements

The current system is working end-to-end, but there are a few clear next improvements already suggested by the codebase and deployment model.

- Unify dependency impact analysis behind one shared engine. Today, `POST /api/v1/slos/impact-analysis` computes cascade effects inline in the API route, while `compute_dependency_impact(...)` exists in the dependency tool layer but is not yet wired into the exposed path. A good next step is to consolidate both into one canonical dependency-impact subsystem so feasibility, upstream/downstream propagation, blast radius, and critical-path effects all come from one model.
- Expand impact analysis into a fuller service-contract view. Right now impact is mostly expressed as feasibility plus upstream availability tightening, which is useful but incomplete. The next version should combine feasibility impact, propagated availability ceilings, latency propagation, critical-path contribution, and structural risk in a single response so engineers can reason about contract changes more clearly.
- Strengthen final report contracts and validation. Planner-style agents already return structured outputs where appropriate, but dict-heavy report stages still rely on prompt-constrained JSON plus local parsing. Refactoring those report payloads into stricter explicit response shapes would improve robustness, reduce parser cleanup, and make downstream UI/API handling more reliable.
- Pre-enrich graphs before runtime. More catalog synchronization and metadata hydration would reduce `AWAIT_INPUT` pauses for missing latency, missing external SLA assumptions, and missing tier data. This would make the system feel less interactive for mature services and more automatic for repeated runs.
- Promote retrieval into a standalone service if scale demands it. The current worker-local MCP subprocess model is simple and correct for moderate scale, which is why it is the right default today. If retrieval traffic or corpus size grows materially, the knowledge boundary can move into a separately scaled retrieval tier while keeping the same logical MCP interface.
- Strengthen ChromaDB production operations. ChromaDB is currently a shared internal service with persistent storage and a single-replica deployment model, which is fine for the current scale target. Future production work could add clearer backup and reindex procedures, deeper health checks, performance tuning, and possibly a higher-scale retrieval backend if corpus size or QPS grows significantly.
- Add shadow-mode recommendation comparison. A practical next step is to run alternative recommendation policies, retrieval strategies, or prompts in shadow mode without changing the active production output. That would let the team compare acceptance rate, review overrides, and incident correlation before promoting changes.
- Learn from human review outcomes. Review decisions already capture where operators agree, modify, or reject recommendations, which makes them a valuable feedback signal. Those outcomes can be used to improve prompts, fallback heuristics, confidence thresholds, and default assumptions without turning the system into an opaque self-learning loop.
- Add stronger write-back controls for automated adoption. If the platform later wants recommendations to be auto-applied, the system should gain a policy engine, explicit approval rules, rollback controls, and tighter authorization boundaries. That work belongs after recommendation quality and governance are stable, not before.
- Improve multi-tenant and multi-region operating modes. The current architecture is a good fit for a moderate shared deployment, but a broader platform rollout would need clearer partitioning by tenant and region. That would include stronger isolation for graph state, recommendation history, retrieval corpora, and freshness guarantees across environments.

## Computation

The codebase keeps the core computation outside the LLM.

Dependency stage:
- PageRank
- SCC / cycle detection
- critical-path latency analysis
- blast radius / centrality

Metrics stage:
- Beta-Binomial posterior
- Kalman smoothing
- drift detection
- anomaly detection
- error budget and burn-rate analysis

Recommendation stage:
- series / parallel reliability reasoning
- Monte Carlo estimation
- feasibility checks
- CLT / Hoeffding style latency estimation
- MILP portfolio optimization

## Observability - Comet Opik

LLM calls and final recommendations are traced and logged through the observability layer.

Current stack:
- Opik for LLM traces and recommendation audit events
- loguru for structured server logs
- FastAPI request logging with trace information

Recommendation audit records include:
- service
- availability
- latency
- confidence
- sources
- decision

## Platform Integration

Platform-facing endpoints:
- `POST /api/v1/catalog/ingest`
- `POST /api/v1/webhooks/register`
- `GET /api/v1/webhooks`
- `GET /api/v1/slo/{service_name}`
- `GET /.well-known/agent.json`

Supported catalog normalization paths:
- Backstage
- Port
- Cortex
- generic format

## Project Structure

The repo is organized by runtime boundary. The tree below shows the important files and what each one is responsible for.

```text
slo_recommendation_engine/
├── api/
│   ├── main.py                        # FastAPI app, middleware wiring, ADK app exposure
│   ├── middleware/
│   │   └── rate_limit.py             # API rate limiting and request throttling
│   └── routes/
│       ├── services.py               # Service catalog, graph ingest, graph read APIs
│       ├── recommendations.py        # Direct compute recommendation APIs and impact analysis
│       ├── slos.py                   # Persisted SLO CRUD-style endpoints
│       ├── reviews.py                # Human review queue and review decision APIs
│       ├── integrations.py           # Catalog ingest, webhook registration, platform integration endpoints
│       └── health.py                 # Health and readiness style endpoints
├── src/
│   └── slo_engine/
│       ├── agents/
│       │   ├── agent.py              # Root router, full pipeline orchestration, inline confidence gate
│       │   ├── base.py               # Shared ADK agent abstraction, model resolution, output handling
│       │   ├── llm_manager.py        # Native Gemini default path and alternative provider registration
│       │   ├── prompt.py             # Shared top-level prompt text
│       │   ├── dependency_agent/
│       │   │   ├── agent.py          # Dependency workflow: planner, ingest, cycles, report
│       │   │   ├── prompt.py         # Dependency planner/report prompts
│       │   │   ├── schema.py         # Dependency workflow state and output schemas
│       │   │   └── tools/
│       │   │       ├── tools.py      # Graph ingest, PageRank, cycle detection, critical path, blast radius
│       │   │       └── schema.py     # Tool input/output schemas for dependency math
│       │   ├── metrics_agent/
│       │   │   ├── agent.py          # Metrics workflow: query, anomaly, budget, report
│       │   │   ├── prompt.py         # Metrics prompts and report instructions
│       │   │   ├── schema.py         # Metrics workflow state and report schemas
│       │   │   └── tools/
│       │   │       ├── tools.py      # Bayesian metrics, anomaly logic, error budget computation
│       │   │       └── schema.py     # Tool input/output schemas for metrics math
│       │   ├── recommendation_agent/
│       │   │   ├── agent.py          # Recommendation workflow: generate, feasibility, optimize, report
│       │   │   ├── prompt.py         # Final report synthesis prompts
│       │   │   ├── schema.py         # Recommendation workflow and final report schemas
│       │   │   └── tools/
│       │   │       ├── tools.py      # SLO generation, feasibility, optimization, knowledge retrieval helpers
│       │   │       └── schema.py     # Tool input/output schemas for recommendation math
│       │   └── slo_pipeline/
│       │       └── agent.py          # ADK application entry point exported as slo_pipeline
│       ├── config/
│       │   ├── settings.py           # Pydantic settings and runtime config resolution
│       │   ├── config.py             # Config helpers / structured config models
│       │   ├── dev.yaml              # Development defaults
│       │   └── prod.yaml             # Production defaults
│       ├── core/
│       │   └── agent_registry.py     # Shared registry for agent instances
│       ├── db/
│       │   ├── database.py           # SQLAlchemy engine and session setup
│       │   └── models.py             # ORM models for services, SLOs, reviews, integrations
│       ├── integrations/
│       │   ├── catalog_adapter.py    # Backstage / Port / Cortex / generic catalog normalization
│       │   ├── metrics_adapter.py    # Metrics-source adapter abstraction and mock adapter
│       │   └── webhook_sink.py       # Outbound webhook delivery
│       ├── mcp/
│       │   ├── client.py             # Shared MCP client used by recommendation stage
│       │   └── knowledge_mcp_server.py # Stdio MCP tool server for knowledge retrieval
│       ├── rag/
│       │   └── knowledge_store.py    # Chroma-backed retrieval and adaptive MMR reranking
│       ├── observability/
│       │   └── opik_tracer.py        # Opik tracing and recommendation audit logging
│       ├── utils/
│       │   └── pii_scrubber.py       # PII scrubbing helpers for logs / outputs
│       └── review_store.py           # Review queue persistence and review decision helpers
├── ui/
│   └── ui.py                         # Streamlit app for graph ingest, pipeline runs, error budgets, review UI
├── scripts/
│   └── e2e_run.py                    # Working end-to-end sample flow and best ingest example
├── tests/                            # API, agent, RAG, and property-style tests
├── data/
│   └── knowledge_base/               # JSON knowledge corpus used by the retrieval layer
├── docs/                             # Markdown knowledge docs and reference guidance
|
├── docker/
│   └── docker-compose.yml            # Local multi-service stack: API, UI, DB, Redis, monitoring
├── k8s/
│   ├── api/                          # API deployment, service, ingress
│   ├── ui/                           # UI deployment, service, ingress
│   ├── db/                           # PostgreSQL and Redis manifests
│   └── chromadb/                     # Chroma deployment, service, PVC
├── .github/
│   └── workflows/ci.yml              # CI pipeline for lint, typing, and tests
├── README.md                         # Project overview, setup, architecture, API guide
├── DESIGN_NOTES.md                   # Deeper architecture and design rationale
├── Makefile                          # Local dev, test, MCP, and deployment shortcuts
├── settings.toml                     # Alternate local config source / examples
└── pyproject.toml                    # Python packaging, Ruff, Black, and MyPy configuration
```

## API Contract

Main REST routes:
- `POST /api/v1/services/dependencies`
- `GET /api/v1/services`
- `GET /api/v1/services/graph`
- `GET /api/v1/services/{service_name}`
- `GET /api/v1/services/{service_id}/slo-recommendations`
- `POST /api/v1/recommendations/bulk`
- `POST /api/v1/slos/impact-analysis`
- `POST /api/v1/services/{service_id}/slos`
- `GET /api/v1/services/{service_id}/slos`
- `GET /api/v1/reviews/pending`
- `GET /api/v1/reviews/{recommendation_id}`
- `POST /api/v1/reviews/{recommendation_id}/decision`
- `GET /api/v1/reviews/feedback/summary`
- `POST /api/v1/catalog/ingest`
- `POST /api/v1/webhooks/register`
- `GET /api/v1/webhooks`
- `GET /api/v1/slo/{service_name}`
- `GET /.well-known/agent.json`

ADK app endpoints:
- `POST /apps/slo_pipeline/users/{user_id}/sessions`
- `POST /run`

### Error Response Format

The FastAPI layer uses **RFC 7807 Problem Details** for validation and unhandled error responses. In practice, this means API errors are returned as `application/problem+json` with fields such as:
- `type`
- `title`
- `status`
- `detail`
- `instance`
- `trace_id`

Why this is used:
- it gives the internal developer platform a standard machine-readable error contract instead of ad hoc JSON error shapes
- it makes API failures easier to debug in UI, platform plugins, and automation because every error includes the request instance and trace context
- it aligns cleanly with the W3C trace propagation already used in `api/main.py`, so a failed request can be correlated with backend logs and Opik traces

## Setup and Run

### LLM Backend

Default:

```env
GEMINI_API_KEY=your_key
LLM_MODEL=gemini-3-flash-preview
```

Gemini uses the native Google ADK model-name format.

Alternative providers can still be selected via `LLM_MODEL`:
- `openai/...`
- `huggingface/...`
- `ollama/...`

### Quick Start

```bash
git clone https://github.com/mitrasan22/slo_recommendation_engine
cd slo_recommendation_engine
copy .env.example .env
pip install -r requirements.txt
pip install .
```

Then set:

```env
GEMINI_API_KEY=your_key
LLM_MODEL=gemini-3-flash-preview
```

### Prerequisites

- Python environment for local development
- Gemini API key for the default native backend
- PostgreSQL / Redis / ChromaDB for full local or containerized execution

### Local Development

```bash
cd docker
docker-compose up -d postgres redis
python -m compileall src ui\ui.py
python api/main.py
streamlit run ui/ui.py
```

### Configuration

Relevant files:
- `.env.example`
- `.env`
- `settings.toml`
- `src/slo_engine/config/dev.yaml`
- `src/slo_engine/config/prod.yaml`

## Design Decisions and Trade-offs

- Knowledge retrieval is deterministic and MCP-based, because that preserves the knowledge-server boundary while avoiding unstable LLM tool-calling + JSON-output combinations.
- Numeric recommendation logic lives in tools, not in the LLM.
- Final report synthesis still uses the LLM, but only after deterministic retrieval and computation are complete.
- Native Gemini is the default backend; LiteLLM remains for non-Gemini alternatives.
- The knowledge MCP server is a stdio subprocess per worker, which keeps the architecture simple and local to the API process.

## Assumptions and Design Boundaries

### Key Unknowns

- how complete and trustworthy the service graph is at ingest time
- how complete latency, SLA, and topology metadata are for external dependencies and datastores
- whether recommendations are advisory only or will eventually be auto-applied to platform systems
- expected production scale for concurrent `/run` sessions, API workers, and retrieval traffic
- whether the Knowledge MCP boundary must remain co-located or later become an independently scalable service

### Current Assumptions

- the ingested service graph is the primary source of truth for dependency structure
- metrics adapters provide enough signal to estimate availability, drift, anomaly state, and budget burn
- missing assumptions can be surfaced interactively and answered by the user during pipeline execution
- the knowledge corpus is local to the platform domain and indexed into ChromaDB
- human review remains part of the operational safety model for low-confidence or drifted outputs
- the Knowledge MCP server can remain a per-worker subprocess without becoming a bottleneck

### How Different Assumptions Would Change the Design

- If service graphs become incomplete or stale:
  the system would need stronger catalog synchronization, graph validation, and possibly pre-ingest reconciliation jobs before dependency analysis.
- If retrieval throughput becomes much higher:
  the `knowledge_mcp_server` should move from a co-located subprocess model to a dedicated retrieval service tier with its own scaling and health model.
- If the platform wants auto-application of SLOs:
  the review gate would need policy enforcement, stronger validation, and write-back controls before recommendations could be pushed automatically.
- If external SLA and latency metadata become consistently available:
  the `NEEDS_INPUT` pause path becomes less central and more of the dependency analysis can run unattended.
- If formal schema guarantees become mandatory on every LLM output:
  report schemas would need to be reshaped into Gemini-safe JSON-schema forms instead of relying on prompt-constrained JSON plus local parsing.

## FAQ

### Why are the computation-heavy steps implemented as Python tools instead of LLM reasoning?

Because the codebase treats numeric recommendation logic as deterministic infrastructure. This keeps outputs testable, reproducible, and cheap.

### How does the engine minimize LLM token usage?

By keeping:
- graph computation in Python
- metrics computation in Python
- knowledge retrieval in MCP + Python
- only routing/planning/report synthesis on the LLM side

### Why MILP?

Because portfolio-level SLO allocation is a discrete optimization problem, not just a ranking problem.

### Why MMR?

Because the recommendation stage needs diverse grounding context, not just the nearest near-duplicate chunks. The current implementation retrieves candidates from Chroma and then reranks them with adaptive MMR.

### Why Bayesian Beta-Binomial for availability?

Because the engine needs smoothed probabilistic availability estimates rather than only point estimates.

### Why a confidence gate?

Because the system is designed to auto-approve only safe cases and route drifted or low-confidence outputs to human review.

### Why only a Knowledge MCP server?

Because the knowledge boundary benefits from being an independent retrieval tool surface. Metrics and dependency computation already live cleanly as direct deterministic Python tools in this codebase.

### How does RAG Happen?

Yes. The recommendation orchestrator calls the shared MCP client, which calls the Knowledge MCP server, which calls `KnowledgeStore.retrieve(...)`, which performs Chroma retrieval and MMR reranking.

## Production Architecture

```text
External users / platform integrations
              |
              v
        Ingress layer
   +----------------------+
   | api-ingress          |
   | ui-ingress           |
   +----------+-----------+
              |
      +-------+-------+
      |               |
      v               v
  slo-api         slo-ui
  Deployment      Deployment
  - FastAPI       - Streamlit
  - ADK app       - no business logic
  - native Gemini - calls API + ADK endpoints
  - per-worker
    Knowledge MCP subprocess
      |
      +--> PostgreSQL / TimescaleDB
      +--> Redis
      +--> ChromaDB
      +--> Opik
      +--> webhook targets
```

The production system is best understood as two planes:
- stateless serving plane
- stateful data and retrieval plane

Current defaults:
- Docker Compose API model: `gemini-3-flash-preview`
- Kubernetes API model: `gemini-3-flash-preview`
- dev config: `gemini-3-flash-preview`
- prod config: `gemini-3-flash-preview`

### Serving Plane

The serving plane consists of:
- API deployment:
  FastAPI, Google ADK app, native Gemini calls, and the per-worker Knowledge MCP subprocess
- UI deployment:
  Streamlit frontend, which only drives API and ADK flows

The API deployment is the control plane of the system. It owns:
- REST endpoints
- ADK `/run` sessions
- orchestration across dependency, metrics, and recommendation stages
- review gating
- webhook emission

### Data and Retrieval Plane

The stateful backing plane consists of:
- PostgreSQL / TimescaleDB
- Redis
- ChromaDB

These components are reflected directly in:
- `k8s/db/postgres-deployment.yaml`
- `k8s/db/redis-deployment.yaml`
- `k8s/chromadb/chromadb-deployment.yaml`

ChromaDB is not just a local library dependency here. In Kubernetes it is deployed as its own internal retrieval backend with:
- a dedicated `Deployment`
- a dedicated internal `ClusterIP` service
- a dedicated persistent volume claim for the vector store

### Knowledge MCP in Production

This is the key architecture nuance:
- the Knowledge MCP server is not deployed as its own network service
- it is launched as a stdio subprocess by the API worker through the shared MCP client
- each API worker keeps its own persistent MCP subprocess

So the production grounding path is:

```text
request
-> slo-api worker
-> RecommendationOrchestratorAgent
-> KnowledgeMCPClient
-> knowledge_mcp_server subprocess
-> KnowledgeStore.retrieve(...)
-> Chroma candidate retrieval
-> adaptive MMR reranking
-> grounded recommendation synthesis
```

### ChromaDB in Production

ChromaDB is the production retrieval backend behind the MCP layer.

Current deployment shape:
- `k8s/chromadb/chromadb-deployment.yaml`
- `k8s/chromadb/chromadb-service.yaml`
- `k8s/chromadb/chromadb-pvc.yaml`

Current production characteristics from the manifests:
- single replica deployment
- `ClusterIP` internal service, not internet-exposed
- persistent storage through a PVC
- `Recreate` rollout strategy because the current volume is `ReadWriteOnce`

This means the current scaling model is:
- API replicas can scale horizontally
- each API worker keeps its own MCP subprocess
- all retrieval subprocesses talk to one shared ChromaDB service
- ChromaDB itself is currently scaled vertically / operationally, not horizontally

So yes, ChromaDB is part of the production architecture and part of retrieval scaling, but in the current repo it is a shared internal retrieval backend rather than a multi-replica distributed tier.

### Horizontal Scaling

The API and UI can scale independently.

Important scaling behavior:
- each API worker creates and maintains its own Knowledge MCP subprocess
- scaling API replicas also scales MCP subprocess count
- PostgreSQL, Redis, and ChromaDB remain shared dependencies
- the knowledge server is lightweight and local to the worker rather than a separately scaled service tier

### Observability - Full Stack

Observability should be thought of at three layers:
- application/request logs from FastAPI and the agents
- LLM and recommendation traces through Opik
- infrastructure monitoring around API, DB, Redis, and Chroma

### Security Hardening

Security-relevant surfaces already present in the repo:
- JWT secret configuration
- webhook secret support
- rate limiting middleware
- separation of API, UI, DB, Redis, and Chroma components in deployment manifests

### Database - Production Setup

Current stateful components:
- PostgreSQL / Timescale for services, dependencies, metrics-derived records, reviews, audit
- Redis for rate limiting and transient queue/cache behaviors
- ChromaDB for vector retrieval
- SQLite only for local ADK session persistence in development

### LLM Reliability - Circuit Breaker and Fallback

The current codebase supports native Gemini as the default path and LiteLLM-based alternatives for non-Gemini models. Reliability handling should be thought of in two layers:
- provider/client retries
- higher-level circuit-breaker / fallback policy

When the LLM path is unhealthy, deterministic computation and retrieval still exist, but final outputs should remain human-gated.

### CI/CD and Deployment

Repo deployment assets exist in:
- `docker/`
- `k8s/`
- `.github/`

The current architecture expects:
- API image deployment
- UI image deployment
- shared access to PostgreSQL / Redis / ChromaDB
- API-hosted Knowledge MCP subprocesses rather than a separate knowledge-service deployment

### Backup and Recovery Summary

Operationally important data stores:
- PostgreSQL / Timescale
- Redis where used for transient state
- ChromaDB index / knowledge corpus
- local feedback log / audit surfaces
