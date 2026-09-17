.PHONY: all install fetch train

all: install fetch train

install: ## Install dependencies
	uv sync

fetch: ## Fetch data from the database into data/01_raw
	uv run python -m scripts.fetch_data

train: ## Train data with latest model
	uv run python -m scripts.train