# Automated Swing Trader - Makefile

.PHONY: help setup test test-cov lint format check clean

help:
	@echo "AST Development Commands"
	@echo ""
	@echo "  make setup      - Set up development environment"
	@echo "  make test       - Run all tests"
	@echo "  make test-cov   - Run tests with coverage"
	@echo "  make lint       - Run linters (ruff, mypy)"
	@echo "  make format     - Format code (black, isort)"
	@echo "  make check      - Run all quality checks"
	@echo "  make clean      - Clean temporary files"

setup:
	@echo "Setting up development environment..."
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@test -f config/config.yaml || cp config/config.example.yaml config/config.yaml
	@test -f config/secrets.env || cp config/secrets.env.example config/secrets.env
	@echo "Done. Edit config/secrets.env with your API keys."

test:
	@pytest --tb=short -q

test-cov:
	@pytest --cov=. --cov-report=html --cov-report=term

lint:
	@ruff check .
	@mypy . --ignore-missing-imports

format:
	@black .
	@isort .

check: lint test

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@rm -f .coverage
