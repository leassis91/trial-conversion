.PHONY: help install fetch

help: ## Show available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

fetch: ## Fetch data from the database into data/01_raw
	uv run python -m scripts.fetch_data