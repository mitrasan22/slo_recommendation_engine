# SLO Recommendation Engine - Design Notes

---

## 1. Architectural Diagram

```text
+----------------------------------------------------------------------------------+
|                       DEVELOPER PLATFORM / OPERATIONS UI                         |
|  Streamlit UI | Backstage / Port / Cortex / custom portal | webhooks / polling   |
+----------------------------------------------+-----------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------+
|                               API LAYER (FastAPI)                                |
|  service graph ingest | direct tool APIs | reviews | integrations | ADK /run     |
+----------------------------------------------+-----------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------+
|                                ADK APP: slo_pipeline                             |
|                                  SLORouterAgent                                  |
|                  route -> dependency -> metrics -> recommendation -> gate         |
+-------------------------+--------------------------+-----------------------------+
                          |                          |
                          v                          v
               +-----------------------+   +-----------------------+
               | DependencyOrchestrator|   |  MetricsOrchestrator |
               | - PLAN                |   | - QUERY              |
               | - AWAIT_INPUT         |   | - AWAIT_INPUT        |
               | - INGEST              |   | - ANALYZE            |
               | - CYCLES              |   | - ANOMALY            |
               | - REPORT              |   | - BUDGET / REPORT    |
               +-----------+-----------+   +-----------------------+
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
                           |               +--> deterministic Python solver / optimization tools
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
                              inline confidence gate + review queue + webhook push
```

---

## 2. Agent Workflow

```text
User / UI
-> SLORouterAgent
   -> RouterDecisionAgent when the session is not resuming
-> DependencyOrchestratorAgent
   -> DependencyPlannerAgent
      -> PLAN
   -> AWAIT_INPUT if graph assumptions are missing
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
   -> AWAIT_INPUT if metric assumptions are missing
   -> MetricsReportAgent
-> RecommendationOrchestratorAgent
   -> GENERATE
      -> deterministic query build
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
-> webhook push / human review queue
```

---

## 3. Architecture Overview

### Key System Components

#### Data Ingestion and Processing

The system accepts service graphs through `POST /api/v1/services/dependencies`. Each graph can declare dependencies in two forms:
- inline `depends_on` per service
- top-level `edges`

Both are merged before graph analysis. The resulting graph is also exposed in agent-ready form through `GET /api/v1/services/graph`, which is what the UI uses before starting an ADK session.

Platform integration is handled separately through `POST /api/v1/catalog/ingest`, where platform-specific adapters normalize Backstage, Port, Cortex, or generic inputs into the internal service graph format.

Operational inputs also include:
- service-level metrics from the configured metrics adapter
- developer-provided clarification answers during `AWAIT_INPUT`
- local knowledge documents and JSON knowledge entries for grounding

#### Dependency Modeling and Graph Construction

Dependencies are modeled as a weighted directed graph:
- node = service
- edge = dependency from caller to callee
- `dep_type` captures synchronous / asynchronous / external behavior
- `weight` represents dependency strength

The dependency stage computes:
- PageRank
- cycle detection / SCC analysis
- critical-path style latency structure
- blast-radius / centrality information

#### Metrics Analysis

The metrics stage produces service-health summaries from the configured metrics adapter. It combines:
- Beta-Binomial availability estimation
- Kalman smoothing
- KL-divergence drift detection
- anomaly detection
- burn-rate / budget state analysis

#### Recommendation Engine

The recommendation stage combines:
- dependency context
- metrics context
- deterministic knowledge retrieval
- feasibility checks
- portfolio optimization
- final report synthesis

At runtime the orchestrator directly calls deterministic tools for generation,
feasibility, and optimization, then delegates only the final report step to
`SLOReportAgent`.

#### Knowledge / RAG Layer

The knowledge layer is an MCP-backed retrieval system. In the current implementation:

`RecommendationOrchestratorAgent`
-> deterministic query construction
-> `KnowledgeMCPClient.retrieve_knowledge(...)`
-> `knowledge_mcp_server`
-> `KnowledgeStore.retrieve(...)`
-> Chroma ANN retrieval
-> adaptive MMR reranking

This means RAG is happening through the Knowledge MCP server boundary, not inside the LLM.

#### MCP Deployment Model

The MCP server is deployed as a co-located runtime component, not as a
standalone network service:
- `KnowledgeMCPClient` spawns `knowledge_mcp_server` as a stdio subprocess
- the subprocess lives inside the same API runtime boundary as the worker that uses it
- retrieval state remains externalized in ChromaDB, while the MCP boundary remains local

This is an important production detail because it changes the deployment model:
- there is no dedicated MCP ingress, load balancer, or service discovery layer
- each API pod or worker owns its own MCP subprocess
- scaling the API horizontally also scales MCP subprocess count
- the MCP server should therefore be packaged into the same deployable image as the API

Production retrieval path:

```text
API worker
-> KnowledgeMCPClient
-> stdio knowledge_mcp_server subprocess
-> KnowledgeStore.retrieve()
-> ChromaDB
```

#### Actual Orchestrator-to-Sub-Agent Calls

The real call graph is narrower than a generic staged-agent diagram might imply:
- `SLORouterAgent` calls `RouterDecisionAgent` only when it is not resuming an in-progress session.
- `DependencyOrchestratorAgent` calls `DependencyPlannerAgent` and `DependencyReportAgent`; ingest, graph analysis, and cycle detection are direct Python tool calls.
- `MetricsOrchestratorAgent` delegates only the report step to `MetricsReportAgent`; query, anomaly detection, and budget evaluation are direct Python tool calls.
- `RecommendationOrchestratorAgent` delegates only the report step to `SLOReportAgent`; knowledge retrieval and all recommendation solver are deterministic direct calls.

#### Evaluation, Feedback, and Governance

Recommendations can be auto-approved or sent to human review. Review decisions are exposed through the review endpoints and stored in the review layer for audit and feedback analysis.

#### Observability and Auditing

The system records:
- API logs
- agent-step logs
- recommendation audit events
- Opik traces for LLM-related flows

---

### Data Flow

```text
graph ingest
-> dependency analysis
-> metrics analysis
-> knowledge retrieval through MCP
-> recommendation generation
-> feasibility check
-> portfolio optimization
-> final report synthesis
-> confidence gate
-> auto-approve or human review
-> webhook / polling / audit outputs
```

### API Layer Design

The FastAPI layer serves three kinds of workloads:
- direct REST endpoints for graph, recommendation, and review operations
- platform integration endpoints
- ADK app endpoints for the multi-agent pipeline

Important routes in the current repo:
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

### Integration with Developer Platform

The system is platform-agnostic. Integration happens through:
- catalog ingest
- webhook registration
- SLO polling
- agent-card discovery

In a Backstage-like environment this maps naturally to:
- service catalog sync jobs that publish topology and ownership metadata
- plugin or portal UI panels that call the REST API and ADK session endpoints
- webhook or polling-based recommendation delivery back into service views
- platform auth controls in front of the API rather than inside every agent

### Deployment Architecture

Local and production both keep the same logical architecture:
- API process hosting FastAPI + ADK
- UI process hosting Streamlit
- PostgreSQL / Timescale
- Redis
- ChromaDB

The important current implementation detail is that the Knowledge MCP server is **not** an independently networked service. It runs as a stdio subprocess managed by the API worker via the shared MCP client helper.

---

## 4. Dependency Modeling

### Graph Schema

The graph payload shape is:

```json
{
  "services": [
    {
      "service": "order-orchestrator",
      "tier": "critical",
      "p99_latency_ms": 250.0,
      "depends_on": [
        { "name": "payment-processor", "dep_type": "synchronous", "weight": 1.0 }
      ]
    }
  ],
  "edges": []
}
```

This is the same shape used by the UI and by the e2e runner.

### How Dependency Structure Influences SLO Recommendations

Dependency shape affects:
- achievable availability ceilings
- critical-path latency pressure
- cycle handling requirements
- blast radius and review severity
- propagation of upstream error and latency constraints into downstream SLO targets

The recommendation engine uses dependency structure to:
- cap achievable availability based on upstream chains and external dependencies
- account for high fan-in or high blast-radius services more conservatively
- flag cycle-heavy or graph-invalid cases for review instead of pretending the structure is linear

### Handling Circular, Missing, or Partial Dependency Information

The dependency stage can pause for:
- cycle resolution assumptions
- missing external SLAs
- missing latency assumptions
- missing tier information

This is handled through the `AWAIT_INPUT` state and session resume.

---

## 5. SLO Recommendation Approach

### Inputs Considered

The recommendation stage reads:
- dependency report output
- metrics query / metrics report output
- knowledge MCP output
- dependency SLO assumptions where relevant
- historical availability and smoothed performance estimates
- burn-rate and error-budget status
- service criticality inferred from graph position and tier
- incident-like drift/anomaly signals
- resource/latency proxies when direct telemetry is incomplete

### Methodology - Hybrid

The system is hybrid by design:
- LLMs for routing, clarification, and synthesis
- deterministic Python for retrieval control, solver, feasibility, and optimization

More concretely:
- rule-based logic is used in workflow gating, pause/resume behavior, and review routing
- statistical methods are used in Bayesian availability estimation, drift detection, and feasibility reasoning
- optimization is used in portfolio-level SLO allocation
- LLM assistance is used where explanation, routing, or clarification quality matters more than exact arithmetic

### Agent Architecture

#### Router

`SLORouterAgent` decides the route and executes the full pipeline sequence.

#### Dependency Stage

`DependencyOrchestratorAgent` workflow:
- `PLAN`
- `AWAIT_INPUT`
- `INGEST`
- `CYCLES`
- `REPORT`

#### Metrics Stage

`MetricsOrchestratorAgent` workflow:
- `QUERY`
- `AWAIT_INPUT`
- `ANALYZE`
- `ANOMALY`
- `BUDGET`
- `REPORT`

#### Recommendation Stage

`RecommendationOrchestratorAgent` workflow:
- `GENERATE`
- `FEASIBILITY`
- `OPTIMIZE`
- `REPORT`

Actual `GENERATE` path:

```text
load state
-> build deterministic knowledge query
-> call KnowledgeMCPClient
-> knowledge_mcp_server runs retrieve_knowledge
-> KnowledgeStore.retrieve performs Chroma + MMR
-> use source_ids + context_summary in generation input
-> call generate_slo_recommendation(...) tool
```

Important nuance:
- not every LLM agent is schema-enforced in the same way
- planner-style agents use structured response JSON where compatible
- dict-heavy final report synthesizers emit prompt-constrained JSON and are parsed locally by orchestrator code

### Propagation of Upstream/Downstream Constraints

Dependency shape constrains what can be recommended upstream. This is why the system runs graph analysis before recommendation synthesis.

In practice:
- upstream external dependencies constrain achievable availability
- synchronous chains contribute latency and reliability pressure
- circular structures invalidate naive series reasoning and trigger alternate handling or review
- downstream criticality influences how conservative the recommendation and gate should be

### Balancing Feasibility vs Ambition

The system separates:
- what is mathematically / operationally feasible
- what is desirable as a target

This is why feasibility is a distinct stage before the final report.

This gives a practical balance between:
- ambitious platform goals
- what historical performance can actually sustain
- what dependency structure makes possible
- what should still be escalated to humans instead of auto-approved

---

## 6. Knowledge & Reasoning Layer

### Knowledge Sources

Knowledge comes from:
- `data/knowledge_base/*.json`
- `docs/*.md`

The MCP server does not own inline knowledge. It delegates retrieval to the shared knowledge store.

### Retrieval Strategy

Current retrieval flow:
- deterministic query construction in recommendation orchestrator
- MCP call through shared `KnowledgeMCPClient`
- `knowledge_mcp_server` tool dispatch
- `KnowledgeStore.retrieve(...)`
- Chroma ANN candidate retrieval
- adaptive MMR reranking

### Grounding

The recommendation stage receives:
- `source_ids`
- `context_summary`
- raw `results`

The final report includes:
- `sources`
- `knowledge_context`

Grounding is validated by:
- deterministic retrieval before synthesis
- explicit source propagation into final output
- keeping math-heavy outputs outside the LLM path
- routing low-confidence or structurally risky cases to review

### Guardrails Against Hallucinations

Three important safeguards:
1. numeric outputs come from deterministic tools, not the LLM
2. knowledge retrieval happens before final synthesis and is explicit in state
3. the final confidence gate can still require human review

---

## 7. Evaluation & Safety

### Recommendation Quality

Recommendation quality is shaped by:
- dependency context quality
- metrics quality
- retrieved knowledge quality
- feasibility constraints
- review gating

Quality can be evaluated through:
- backtesting against historical service performance
- correlation with later incident/error-budget outcomes
- consistency checks across dependent services
- comparison between proposed targets and historical feasible envelopes

### Safety & Guardrails

Safety comes from:
- deterministic numeric logic
- explicit review path
- source-cited final reports
- ability to pause for missing assumptions

Additional protections in the current design:
- input validation on API payloads
- PII/sensitive-data scrubbing utilities in the codebase
- local parsing and validation of LLM-produced JSON
- rate limiting at the API middleware layer
- safe degradation when retrieval or synthesis confidence is poor

---

## 8. Explainability

### What Every Recommendation Discloses

The intended final report shape includes:
- recommended availability
- recommended latency
- confidence
- feasibility
- review reason when needed
- sources
- knowledge context
- reasoning summary

### Audit Trail

The system logs recommendation audit events and review transitions for traceability.

### Feedback Mechanism

The review endpoints create an explicit feedback loop around low-confidence or drifted recommendations.

---

## 9. Continuous Improvement

### Feedback Loop

Review decisions can be used to understand where recommendations are consistently modified or rejected.

These signals can feed:
- prompt refinement
- heuristic tuning
- better dependency assumption defaults
- future offline evaluation datasets

### A/B Testing and Shadow Mode

This is not deeply implemented as a separate subsystem in the current codebase, but the architecture supports comparing recommendation behavior through logged outputs and review outcomes.

A practical next step would be:
- run candidate recommendation policies in shadow mode
- compare acceptance rate, incident correlation, and review overrides
- promote only when the new policy improves both quality and operator trust

### Model Drift Detection and Retraining

The metrics stage already detects drift at runtime for service behavior; that drift signal is part of the recommendation and gating path.

For longer-term evolution:
- retrieval content versions should be tracked
- prompt and heuristic versions should be logged with outputs
- any learned ranking or scoring component should be retrained only from reviewed or backtested data

---

## 10. Production Architecture & Scalability

### Production Deployment Diagram

```text
Ingress / clients
    |
    v
API deployment
  - FastAPI
  - ADK app
  - native Gemini client
  - per-worker Knowledge MCP subprocess
    |
    +--> PostgreSQL / Timescale
    +--> Redis
    +--> ChromaDB
    +--> Opik
    +--> webhook targets

UI deployment
  - Streamlit
  - API / ADK client
```

In this production shape, ChromaDB is a first-class backend service rather than only an embedded implementation detail. The repo contains dedicated manifests for:
- `k8s/chromadb/chromadb-deployment.yaml`
- `k8s/chromadb/chromadb-service.yaml`
- `k8s/chromadb/chromadb-pvc.yaml`

That means the deployed retrieval path is best understood as:

```text
API worker
-> KnowledgeMCPClient
-> worker-local knowledge_mcp_server subprocess
-> KnowledgeStore.retrieve()
-> shared ChromaDB service
-> persistent vector index on PVC
```

### Horizontal Scaling

The API and UI can scale independently. The main current caveat is:
- each API worker owns its own persistent Knowledge MCP subprocess

The retrieval layer has a different scaling model:
- the MCP server scales with API workers because it is co-located
- ChromaDB is currently a shared internal backend service
- the Kubernetes manifests show a single-replica ChromaDB deployment with persistent storage
- this is a valid production pattern for moderate scale, but it is vertical/shared scaling rather than a distributed retrieval cluster

Operationally, this means:
- API throughput can increase without duplicating the vector index
- retrieval contention is centralized at the ChromaDB tier
- future high-scale evolution would likely split ChromaDB into a more explicitly managed retrieval platform or replace it with a horizontally scalable vector backend

The design is intended to scale to hundreds or thousands of services by:
- keeping core graph and metrics computation deterministic
- separating interactive `/run` flows from direct REST endpoints
- using shared stateful backends for persistence and retrieval
- allowing the retrieval tier to be split out later if needed

### Observability - Full Stack

Current observability surfaces:
- request / application logs
- Opik traces
- audit logs
- infrastructure monitoring around API, DB, Redis, and Chroma

### Security Hardening

Current security-related surfaces in repo:
- JWT secret config
- webhook secrets
- rate limiting
- deployment separation of components

### Database - Production Setup

Current persistent / stateful stores:
- PostgreSQL / Timescale
- Redis
- ChromaDB
- local SQLite for development ADK session persistence

Data freshness assumptions:
- service graph freshness depends on ingest cadence or catalog sync cadence
- metrics freshness depends on the adapter and the query window
- knowledge freshness depends on re-indexing cadence into ChromaDB

### LLM Reliability - Circuit Breaker + Fallback

The codebase now defaults Gemini to the native ADK path. Alternative providers still use LiteLLM-compatible backends.

Operationally this means:
- provider/client retries depend on backend
- deterministic computation and retrieval remain available even if final synthesis is degraded
- human review is the safe fallback path when LLM behavior is not trustworthy

### Operational Resilience

Important resilience behaviors:
- missing assumptions -> pause and resume
- review-required recommendations -> queue instead of auto-applying
- knowledge retrieval failure -> recommendation can still proceed, but with less grounding

### API Performance Targets

Not hard-coded as formal SLOs in docs, but the architecture is designed around:
- relatively fast direct REST endpoints
- slower multi-agent `/run` pipeline requests
- resumable sessions for long-running interactive flows

Cost considerations:
- deterministic Python computation keeps token spend lower than LLM-only approaches
- co-located MCP subprocesses reduce deployment complexity at moderate scale
- final synthesis is the main LLM-heavy step, which is intentional for cost control

---

## 11. Assumptions & Trade-offs

### Assumption-Driven Baseline Architecture

The current architecture is built around five operating assumptions:

1. The service graph is good enough to act as the primary topology input.
2. Metrics adapters provide enough signal for availability, drift, and budget analysis.
3. Missing business or operational context can be recovered interactively during the run.
4. Knowledge retrieval traffic is moderate enough for a worker-local MCP subprocess model.
5. Recommendations are advisory and can be gated by human review when confidence is low.

Under those assumptions, the chosen architecture is:

```text
platform graph + metrics
-> API / ADK app
-> dependency + metrics orchestrators
-> recommendation orchestrator
-> worker-local Knowledge MCP subprocess
-> Chroma-backed retrieval
-> final report
-> confidence gate + review queue
```

This baseline favors:
- low operational complexity
- deterministic math and retrieval
- resumable interactive clarification
- safe rollout through human review

### Key Unknowns That Could Change the Architecture

The most important open variables are:
- how authoritative the graph is compared with live platform state
- how often SLA, latency, datastore, and infrastructure metadata are missing
- how much concurrent recommendation traffic the platform must sustain
- whether recommendations stay advisory or become an automated write path
- whether the MCP subprocess model remains operationally sufficient at scale

### Architectural Variants If Assumptions Change

#### If the Service Graph Is Not Trustworthy

Current assumption:
- graph ingest is sufficiently accurate for dependency analysis

Architecture consequence:
- dependency analysis can run directly from ingested graph state

If this assumption breaks, the architecture should change to:

```text
catalog systems / deployment metadata / service discovery
-> graph reconciliation layer
-> validated topology store
-> dependency orchestrator
```

That adds:
- topology reconciliation jobs
- graph quality scoring
- conflict handling between declared and observed dependencies

#### If Missing Context Becomes Rare

Current assumption:
- some missing inputs are expected, so `AWAIT_INPUT` is a normal part of the workflow

Architecture consequence:
- planners can pause the pipeline and ask the engineer for assumptions

If this assumption improves, the architecture should change to:
- reduce planner-centric clarification paths
- move more runs to unattended execution
- pre-enrich graph and metrics inputs from platform metadata before the pipeline starts

#### If Retrieval Load or Corpus Size Grows Materially

Current assumption:
- a per-worker `knowledge_mcp_server` subprocess is operationally acceptable

Architecture consequence:
- MCP stays co-located with the API worker
- no separate network service is needed for knowledge retrieval

If this assumption breaks, the architecture should change to:

```text
API workers
-> retrieval gateway / standalone MCP service
-> shared retrieval workers
-> Chroma or another vector backend
```

That would improve:
- independent scaling of retrieval
- centralized health management
- better isolation between API throughput and retrieval throughput

#### If Recommendations Become Auto-Applied

Current assumption:
- recommendations are advisory outputs with a confidence gate and human review

Architecture consequence:
- the system can optimize for explanation quality and safe review routing

If this assumption changes, the architecture should add:

```text
recommendation output
-> policy engine
-> approval / exception workflow
-> platform write-back adapters
-> audit and rollback controls
```

That would require:
- stronger validation
- explicit authorization boundaries
- write safety, rollback, and change tracking

#### If Strict Structured Output Guarantees Become Mandatory

Current assumption:
- prompt-constrained JSON plus local parsing is acceptable for report synthesis

Architecture consequence:
- report agents can stay flexible even with Gemini schema limitations

If this assumption changes, the architecture should change to:
- refactor report schemas into Gemini-safe explicit shapes
- move more validation into typed post-processing
- possibly split synthesis into smaller schema-safe stages

### Why the Current Assumptions Are Reasonable

For the current repo and working demo, these assumptions are reasonable because:
- the system already supports interactive pause/resume well
- deterministic math and retrieval are implemented and tested
- the current deployment model is simpler to run locally and in a single API tier
- the review gate provides a safe operational fallback when uncertainty is high

### Explicit Assumptions Required by the Design

#### Available Data Sources

- metrics are available at service granularity with enough retention to evaluate recent windows such as 30-day behavior
- availability, latency, and burn-related indicators are derivable from the configured metrics source
- dependency metadata is available either from ingest payloads or platform catalog synchronization
- knowledge sources are curated internal documents, templates, and guidelines rather than open-ended web search

#### Scale Characteristics

- the baseline design assumes hundreds to low-thousands of services rather than hyperscale fleet telemetry
- recommendation runs are relatively sparse compared with raw metrics ingestion volume
- the initial deployment can tolerate worker-local MCP subprocesses and shared ChromaDB access
- multi-region concerns exist conceptually, but the current implementation is closer to single-region or modest multi-region operation
- ChromaDB is assumed to be a shared internal retrieval backend with moderate corpus size and moderate concurrent query load, not yet a globally distributed retrieval plane

#### Developer Platform Capabilities

- the internal developer platform can expose service metadata through APIs or exportable catalog payloads
- the platform can authenticate users before they reach this API
- the platform can embed a UI or plugin that calls REST and ADK session endpoints
- asynchronous delivery via webhook or polling is acceptable for completed recommendations

#### Team Structure and Operational Maturity

- service owners are available to answer clarification questions when automation lacks context
- some human review capacity exists for flagged recommendations
- the platform/SRE team can maintain the knowledge corpus and retrieval indexing pipeline
- the organization values explainability and controlled rollout over fully autonomous action-taking

### Trade-offs & Design Decisions

- deterministic MCP retrieval was chosen over LLM tool-calling for the knowledge step
- native Gemini is the default backend for stability on the current path
- final report synthesis remains LLM-based because explanation quality matters
- the knowledge MCP server is local-to-worker rather than a separately managed network service
- the hybrid method was chosen because it gives a better accuracy/latency/cost balance than either pure rules or pure LLM reasoning
- engineering effort is invested first in deterministic graph/math/retrieval correctness and safe review loops, and later in higher-scale retrieval/service decomposition

### Out of Scope

- full autonomous action-taking on platform systems
- automatic alert-rule deployment as a first-class subsystem
- independent horizontal scaling of the knowledge server as its own service tier

---

## 12. FAQ

### Why are the mathematical algorithms implemented as Python tool functions instead of the LLM?

Because the system treats numeric recommendation logic as deterministic infrastructure.

### How does the engine minimize LLM token usage?

By keeping graph analysis, metrics analysis, knowledge retrieval control, feasibility, and optimization outside the LLM path.

### Why MILP instead of a simple score formula?

Because portfolio allocation is a constrained optimization problem.

### Why MMR with adaptive lambda?

Because the recommendation stage needs diverse, non-duplicate grounding context from the knowledge base.

### Why Bayesian availability estimation?

Because the system needs smoothed probabilistic estimates rather than only raw point values.

### Why a confidence gate instead of manually reviewing everything?

Because the design supports auto-approval for safe cases and human review for drifted or flagged ones.

### How does RAG happen?

Yes. The recommendation orchestrator calls the shared MCP client, which calls the Knowledge MCP server, which calls `KnowledgeStore.retrieve(...)`, which performs Chroma retrieval and MMR reranking.

---

## 13. API Contract Summary

Core routes:
- `POST /api/v1/services/dependencies`
- `GET /api/v1/services`
- `GET /api/v1/services/graph`
- `GET /api/v1/services/{service_name}`
- `GET /api/v1/services/{service_id}/slo-recommendations`
- `POST /api/v1/recommendations/bulk`
- `POST /api/v1/slos/impact-analysis`
- `GET /api/v1/reviews/pending`
- `POST /api/v1/reviews/{recommendation_id}/decision`
- `POST /api/v1/catalog/ingest`
- `POST /api/v1/webhooks/register`
- `GET /.well-known/agent.json`
- ADK: `POST /apps/.../sessions`, `POST /run`

---

## 14. Illustrative Example - e-Commerce System

The working end-to-end example in this repo is the graph in `scripts/e2e_run.py`, centered around:
- `order-orchestrator`
- `payment-processor`
- `fraud-service`
- `inventory-checker`
- `stripe-gateway`
- `paypal-gateway`
- `supplier-api`

That example is the best current reference for:
- graph ingest shape
- pipeline behavior
- MCP-backed knowledge retrieval
- review-required recommendation flow
