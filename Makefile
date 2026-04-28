.PHONY: install lint format test benchmark smoke setup-biomni eval1-smoke eval1-full notebooks clean pre-commit check

install:
	uv sync --all-extras
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

lint:
	uv run ruff check .
	uv run pyright

format:
	uv run ruff format .

test:
	uv run pytest tests/

smoke:
	uv run python pipelines/run_benchmark.py --config configs/quick_smoke.yaml

benchmark:
	uv run python pipelines/run_benchmark.py --config configs/full_benchmark.yaml

setup-biomni:
	uv sync --extra biomni
	@echo "Biomni installed. First agent.go() call will download ~11 GB of tools into \$$BIOMNI_DATA_PATH (default: ./data)"

eval1-smoke:
	uv run python pipelines/run_eval1.py --config configs/biomni_eval1_smoke.yaml

eval1-full:
	uv run python pipelines/run_eval1.py --config configs/biomni_eval1_full.yaml

notebooks:
	uv sync --all-extras
	uv run jupyter lab notebooks/

clean:
	@echo "Removing notebook artifacts..."
	rm -rf notebooks/results notebooks/data notebooks/.ipynb_checkpoints
	@echo "Removing framework runtime outputs..."
	rm -rf robin_output/
	@echo "Removing tool caches..."
	rm -rf .mypy_cache .ruff_cache .pytest_cache .pyright
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@echo "Done."

pre-commit:
	uv run pre-commit run --all-files

check: lint test pre-commit
