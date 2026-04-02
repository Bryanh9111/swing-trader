# Contributing to AST

Thanks for your interest in contributing! This document covers the basics.

## Development Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp config/config.example.yaml config/config.yaml
cp config/secrets.env.example config/secrets.env
# Edit config/secrets.env with your POLYGON_API_KEY
```

## Running Tests

```bash
pytest --tb=short -q
```

## Code Standards

- Python 3.11+, type hints on all function signatures
- `msgspec.Struct` (frozen=True, kw_only=True) for data contracts
- `Result[T]` return type at all module boundaries (no exceptions for business logic)
- `ruff` for linting, `black` for formatting (both configured in `pyproject.toml`)
- Tests required for all new business logic (target 90%+ coverage)

## Adding a New Pattern Detector

The easiest way to extend the framework:

1. Create `scanner/patterns/your_pattern.py`
2. Implement `TrendPatternDetector` protocol (see `scanner/patterns/interface.py`)
3. Register in `scanner/patterns/__init__.py`
4. Add to `scanner/trend_pattern_router.py`
5. Write tests in `scanner/tests/`

See `scanner/patterns/ma_crossover.py` for a complete example.

## Adding a New Module

Follow the standard module structure:

```
module_name/
  __init__.py
  interface.py      # Input/Output schemas (msgspec.Struct)
  core.py           # Business logic
  config.py         # Configuration schema
  tests/
    test_core.py
```

Key rules:
- Define interfaces first, implementation second
- Return `Result[T]` from all public functions
- Implement degradation behavior for all failure modes
- Emit events for significant state transitions

## Pull Request Process

1. Create a feature branch: `feature/your-feature`
2. Write tests first (TDD encouraged)
3. Ensure all tests pass: `pytest --tb=short -q`
4. Run linter: `ruff check .`
5. Submit PR with clear description of changes

## Architecture Principles

- **Safety first**: When in doubt, do nothing. Capital preservation > opportunity
- **Fail-closed**: Missing data = no trade
- **Deterministic replay**: All snapshots journaled with schema + system version
- **Idempotent operations**: All write-side actions are retry-safe
- **Plugin-based**: Modules interact through stable contracts

## Disclaimer

This software is for educational and research purposes. Trading involves risk of loss. Contributors are not responsible for any financial losses incurred through use of this software.
