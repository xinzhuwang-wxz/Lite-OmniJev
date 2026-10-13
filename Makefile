.PHONY: setup test lint typecheck build check dev smoke queue
setup:
	uv sync --frozen --group dev
test:
	uv run pytest
lint:
	uv run ruff check .
	uv run ruff format --check .
typecheck:
	uv run mypy data train eval serve scripts
build:
	uv build
check: lint typecheck test build
smoke:
	uv run python -m scripts.smoke --output reports/cpu-smoke.json
queue:
	uv run python -m scripts.loop list
dev:
	uv run uvicorn serve.gateway.app:app --host 127.0.0.1 --port 8000
.PHONY: training-check
training-check:
	uv run --extra training pytest tests/test_training.py
