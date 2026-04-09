# SLO Recommendation Engine — Makefile
.PHONY: help install dev build up down test lint fmt pull-model k8s-deploy k8s-teardown clean

DOCKER_COMPOSE = docker compose -f docker/docker-compose.yml
K8S_NAMESPACE   = slo-engine
IMAGE_TAG       = latest
OLLAMA_MODEL    = mistral

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ────────────────────────────────────────────────────────────────
install: ## Install Python dependencies
	pip install -r requirements.txt

dev: ## Run API in dev mode (hot reload)
	PYTHONPATH=src uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dev-ui: ## Run Streamlit UI in dev mode
	streamlit run ui/ui.py --server.port 8501

# ── Ollama ─────────────────────────────────────────────────────────────────────
pull-model: ## Pull Ollama model (default: mistral)
	ollama pull $(OLLAMA_MODEL)
	@echo "Model $(OLLAMA_MODEL) ready"

# ── Docker ─────────────────────────────────────────────────────────────────────
build: ## Build all Docker images
	$(DOCKER_COMPOSE) build --no-cache

up: ## Start all services (detached)
	$(DOCKER_COMPOSE) up -d
	@echo "Services starting..."
	@echo "  API:        http://localhost:8000/docs"
	@echo "  UI:         http://localhost:8501"
	@echo "  Prometheus: http://localhost:9090"

down: ## Stop all services
	$(DOCKER_COMPOSE) down

logs: ## Follow API logs
	$(DOCKER_COMPOSE) logs -f api

# ── Testing ────────────────────────────────────────────────────────────────────
test: ## Run test suite
	PYTHONPATH=src pytest tests/ -v --cov=src/slo_engine --cov-report=term-missing

test-math: ## Run math engine tests only
	PYTHONPATH=src pytest tests/test_math_engine.py -v

test-api: ## Run API integration tests
	PYTHONPATH=src pytest tests/test_api.py -v

# ── Code quality ───────────────────────────────────────────────────────────────
lint: ## Lint with ruff
	ruff check src/ api/ ui/
	mypy src/slo_engine --ignore-missing-imports

fmt: ## Format with black + ruff
	black src/ api/ ui/
	ruff check --fix src/ api/ ui/

# ── Kubernetes ─────────────────────────────────────────────────────────────────
k8s-namespace: ## Create K8s namespace
	kubectl create namespace $(K8S_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -

k8s-secrets: ## Apply K8s secrets (set env vars first)
	kubectl create secret generic slo-engine-secrets \
	  --namespace=$(K8S_NAMESPACE) \
	  --from-literal=postgres-dsn="$(POSTGRES_DSN)" \
	  --from-literal=postgres-user="$(POSTGRES_USER)" \
	  --from-literal=postgres-password="$(POSTGRES_PASSWORD)" \
	  --from-literal=redis-dsn="$(REDIS_DSN)" \
	  --from-literal=jwt-secret="$(JWT_SECRET)" \
	  --dry-run=client -o yaml | kubectl apply -f -

k8s-deploy: k8s-namespace ## Deploy all K8s manifests
	kubectl apply -f k8s/db/
	kubectl apply -f k8s/api/
	kubectl apply -f k8s/ui/
	@echo "Deployed to namespace: $(K8S_NAMESPACE)"

k8s-status: ## Check K8s deployment status
	kubectl get all -n $(K8S_NAMESPACE)

k8s-teardown: ## Remove all K8s resources
	kubectl delete namespace $(K8S_NAMESPACE)

# ── MCP Servers ────────────────────────────────────────────────────────────────
mcp-metrics: ## Start metrics MCP server
	PYTHONPATH=src python -m slo_engine.mcp.metrics_mcp_server

mcp-dependency: ## Start dependency MCP server
	PYTHONPATH=src python -m slo_engine.mcp.dependency_mcp_server

mcp-knowledge: ## Start knowledge MCP server
	PYTHONPATH=src python -m slo_engine.mcp.knowledge_mcp_server

# ── DB Migrations ──────────────────────────────────────────────────────────────
migrate: ## Run Alembic migrations
	PYTHONPATH=src alembic upgrade head

migrate-new: ## Create new migration
	PYTHONPATH=src alembic revision --autogenerate -m "$(MSG)"

# ── Demo ───────────────────────────────────────────────────────────────────────
demo: ## Run demo: ingest sample graph + get recommendations
	@echo "Ingesting sample e-commerce graph..."
	curl -s -X POST http://localhost:8000/api/v1/services/dependencies \
	  -H "Content-Type: application/json" \
	  -d @tests/fixtures/sample_ecommerce_graph.json | python -m json.tool
	@echo "\nGetting recommendations for checkout-service..."
	curl -s http://localhost:8000/api/v1/services/checkout-service/slo-recommendations | python -m json.tool

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist/ build/
