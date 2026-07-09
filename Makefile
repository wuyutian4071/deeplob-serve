.PHONY: help install lint format-check format typecheck test cov check \
        cpp-configure cpp-build cpp-test cpp-check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Python -----------------------------------------------------------------

install:  ## Sync all Python dependencies (incl. dev group) into a local venv
	uv sync --all-groups

lint:  ## Run ruff lint checks
	uv run ruff check src tests

format-check:  ## Check Python formatting without modifying files (what CI actually runs)
	uv run ruff format --check src tests

format:  ## Auto-format Python with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:  ## Run mypy in strict mode
	uv run mypy

test:  ## Run the Python test suite
	uv run pytest

cov:  ## Run Python tests with coverage report
	uv run pytest --cov=deeplob --cov-report=term-missing

check: lint format-check typecheck test  ## Run the full Python CI gate locally (must match ci.yml)

# --- C++ ----------------------------------------------------------------------

cpp-configure:  ## Configure the C++ build (Debug, ASan+UBSan)
	cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_UBSAN=ON

cpp-build: cpp-configure  ## Build the C++ side
	cmake --build cpp/build -j

cpp-test: cpp-build  ## Run the C++ test suite
	ctest --test-dir cpp/build --output-on-failure

cpp-check: cpp-test  ## Run the full C++ CI gate locally (must match ci.yml)

# --- Housekeeping -------------------------------------------------------------

clean:  ## Remove caches and generated artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml
	rm -rf data checkpoints mlruns
	rm -rf cpp/build cpp/build-release
	find . -type d -name __pycache__ -exec rm -rf {} +
