SHELL := /bin/bash
.DEFAULT_GOAL := help

BACKEND_PORT := 5586
DOCKER_IMAGE := ghcr.io/cubicalbatch/vesta:latest

.PHONY: help install install-py install-frontend \
	dev dev-backend dev-frontend \
	build build-spa \
	docker-build docker-up docker-down docker-logs \
	bench benchmark \
	test lint format format-check typecheck check \
	clean clean-frontend

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## ── Setup ─────────────────────────────────────────────────────────────────

install: install-py install-frontend ## Install backend (uv) + frontend (npm) deps

install-py: ## Sync Python deps via uv
	uv sync

install-frontend: ## Install SPA deps via npm
	cd frontend && npm install

## ── Dev servers ──────────────────────────────────────────────────────────

dev: ## Run backend (reload) + SPA dev server (HMR) together; backend gated ready first
	@./dev.sh

dev-backend: ## Backend only: uvicorn --reload on 0.0.0.0:5586
	./start.sh

dev-frontend: ## SPA only: vite dev server on 0.0.0.0, live-reloading, proxied to the backend
	cd frontend && VESTA_API_PROXY_TARGET=http://127.0.0.1:$(BACKEND_PORT) npm run dev -- --host 0.0.0.0

## ── Build ────────────────────────────────────────────────────────────────

build: build-spa ## Alias for build-spa

build-spa: ## Build the SPA into src/vesta/static/app (served by FastAPI in prod)
	cd frontend && npm run build

## ── Docker ───────────────────────────────────────────────────────────────

docker-build: ## Build the production image (tag: $(DOCKER_IMAGE))
	docker build -t $(DOCKER_IMAGE) .

docker-up: ## Start production via docker compose (localhost:5129, ./data mounted)
	docker compose up --build

docker-down: ## Stop the production compose stack
	docker compose down

docker-logs: ## Tail production compose logs
	docker compose logs -f

## ── Benchmark ────────────────────────────────────────────────────────────

bench: benchmark ## Alias for benchmark

benchmark: ## Run the unified benchmark (default systems, core slice, smoke limit)
	uv run vesta bench run --limit 5

## ── Quality gates ────────────────────────────────────────────────────────

test: ## Run the Python test suite
	uv run pytest

lint: ## Lint Python source with ruff
	uv run ruff check .

format: ## Auto-format Python source with ruff
	uv run ruff format .

format-check: ## Check Python formatting without writing changes
	uv run ruff format --check .

typecheck: ## Strict mypy over src/vesta
	uv run mypy --strict src/vesta

check: lint format-check typecheck test ## Run the full gate: lint + format-check + typecheck + test

## ── Cleanup ──────────────────────────────────────────────────────────────

clean: ## Remove Python/tooling caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name '__pycache__' -not -path './.venv/*' -not -path './frontend/node_modules/*' -exec rm -rf {} +
	rm -rf src/vesta/static/app

clean-frontend: ## Remove frontend build output and node_modules
	rm -rf frontend/node_modules frontend/.svelte-kit src/vesta/static/app
