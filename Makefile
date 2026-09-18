# TheraGraph developer commands.
#
#   make setup      install everything
#   make demo       build the demo corpus, train models, run the agent
#   make check      everything CI runs
#
# Targets run against the local virtualenv, not Docker, so they are fast to iterate on.

SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := uv pip install --python $(VENV)/bin/python
BACKEND := backend
FRONTEND := frontend

export PYTHONPATH := $(CURDIR)/$(BACKEND)/src

.DEFAULT_GOAL := help
.PHONY: help setup setup-backend setup-frontend lint format typecheck test test-fast \
        test-backend test-frontend coverage check migrate migration bootstrap seed \
        seed-smoke train evals agent api worker mcp frontend-dev build docker-up \
        docker-down docker-logs docker-rebuild openapi clean demo

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------- setup
setup: setup-backend setup-frontend ## Install backend and frontend dependencies

setup-backend: ## Create the virtualenv and install the backend
	@command -v uv >/dev/null || (echo "install uv: https://docs.astral.sh/uv/" && exit 1)
	uv venv --python 3.11 $(VENV)
	cd $(BACKEND) && $(PIP) -e ".[dev,xgb]"

setup-frontend: ## Install frontend dependencies
	cd $(FRONTEND) && npm install --no-audit --no-fund

# ------------------------------------------------------------------- quality
lint: ## Lint backend and frontend
	$(VENV)/bin/ruff check $(BACKEND)/src $(BACKEND)/tests
	$(VENV)/bin/ruff format --check $(BACKEND)/src $(BACKEND)/tests
	cd $(FRONTEND) && npm run lint

format: ## Auto-format everything
	$(VENV)/bin/ruff check --fix $(BACKEND)/src $(BACKEND)/tests
	$(VENV)/bin/ruff format $(BACKEND)/src $(BACKEND)/tests
	cd $(FRONTEND) && npm run format

typecheck: ## Type-check backend and frontend
	cd $(BACKEND) && ../$(VENV)/bin/mypy
	cd $(FRONTEND) && npm run typecheck

test: test-backend test-frontend ## Run every test

test-backend: ## Run the backend test suite
	cd $(BACKEND) && ../$(VENV)/bin/python -m pytest

test-fast: ## Backend tests, skipping the slow training ones
	cd $(BACKEND) && ../$(VENV)/bin/python -m pytest -m "not slow"

test-frontend: ## Run the frontend test suite
	cd $(FRONTEND) && npm run test

coverage: ## Backend coverage report
	cd $(BACKEND) && ../$(VENV)/bin/python -m pytest --cov=theragraph --cov-report=term-missing --cov-report=html

check: lint typecheck test ## Everything CI runs
	@echo "✓ all checks passed"

# ------------------------------------------------------------------- database
migrate: ## Apply migrations
	cd $(BACKEND) && ../$(VENV)/bin/alembic upgrade head

migration: ## Create a migration: make migration m="add x"
	cd $(BACKEND) && ../$(VENV)/bin/alembic revision --autogenerate -m "$(m)"

bootstrap: ## Create the schema from the models (throwaway databases only)
	$(VENV)/bin/theragraph db bootstrap
	$(VENV)/bin/theragraph db sources

# ----------------------------------------------------------------------- data
# `replay` is explicit rather than `auto` on the offline targets: auto falls back to the
# network when a fixture is missing, which would make "this works offline" true until
# the day it quietly isn't.
seed: ## Build the full demo corpus from public sources (needs network)
	$(VENV)/bin/theragraph ingest seed --plan demo

seed-smoke: ## Build a small corpus from committed fixtures (no network)
	$(VENV)/bin/theragraph ingest seed --plan smoke --mode replay

train: ## Train and register models for every endpoint (no network)
	$(VENV)/bin/theragraph ingest moleculenet esol --mode replay
	$(VENV)/bin/theragraph ingest moleculenet bbbp --mode replay
	$(VENV)/bin/theragraph ingest moleculenet lipophilicity --mode replay
	$(VENV)/bin/theragraph ingest moleculenet clintox --mode replay
	$(VENV)/bin/theragraph ml train --endpoint solubility --dataset esol
	$(VENV)/bin/theragraph ml train --endpoint bbb_permeability --dataset bbbp
	$(VENV)/bin/theragraph ml train --endpoint lipophilicity --dataset lipophilicity
	$(VENV)/bin/theragraph ml train --endpoint clinical_toxicity --dataset clintox

evals: ## Run the agent evaluation suite and check the CI thresholds
	$(VENV)/bin/theragraph agent eval

# -------------------------------------------------------------------- running
api: ## Run the API with auto-reload
	$(VENV)/bin/theragraph serve --reload

worker: ## Run a background job worker
	$(VENV)/bin/theragraph worker

mcp: ## Run the MCP server on stdio
	$(VENV)/bin/theragraph mcp

frontend-dev: ## Run the frontend dev server
	cd $(FRONTEND) && npm run dev

agent: ## Ask the agent a question: make agent q="..."
	$(VENV)/bin/theragraph agent research "$(q)"

openapi: ## Write docs/openapi.json
	$(VENV)/bin/theragraph openapi

build: ## Build the frontend bundle
	cd $(FRONTEND) && npm run build

# --------------------------------------------------------------------- docker
docker-up: ## Start the full stack
	docker compose up -d --build
	@echo "frontend  http://localhost:5173"
	@echo "api docs  http://localhost:8000/docs"

docker-down: ## Stop the stack (keeps volumes)
	docker compose down

docker-rebuild: ## Rebuild images from scratch
	docker compose build --no-cache

docker-logs: ## Tail the api and worker logs
	docker compose logs -f api worker

# --------------------------------------------------------------------- demo
demo: bootstrap seed-smoke train evals ## Full offline-capable demo from scratch
	@echo ""
	@echo "Demo corpus built and models trained. Next:"
	@echo "  make api           # then open http://localhost:8000/docs"
	@echo "  make frontend-dev  # then open http://localhost:5173"

clean: ## Remove caches and build output
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.mypy_cache $(BACKEND)/.ruff_cache
	rm -rf $(BACKEND)/htmlcov $(BACKEND)/.coverage $(FRONTEND)/dist
