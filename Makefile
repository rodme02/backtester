.PHONY: install test lint format notebooks clean all

PYTHON ?= python
VENV ?= .venv
PIP = $(VENV)/bin/pip
PYBIN = $(VENV)/bin/python

install:
	$(PYBIN) -m pip install --upgrade pip
	$(PIP) install -e ".[dev,notebooks,ml,llm]"
	$(VENV)/bin/pre-commit install || true

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/ruff check . --fix
	$(VENV)/bin/ruff format .

notebooks:
	@for nb in notebooks/*.ipynb; do \
	  echo "Executing $$nb"; \
	  $(VENV)/bin/jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true

all: lint test
