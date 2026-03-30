# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tcga-db is a Python 3.13+ DuckDB database integrating TCGA tumour and GTEx normal tissue data from the UCSC Toil RNA-seq Recompute. Uses `uv` for dependency management and `hatchling` as build backend. Architecture follows patterns from the sibling `depmap-db` project.

## Commands

```bash
# Install / sync
uv sync --all-extras --dev

# Run the app
uv run tcga-db

# Lint & format
uv run ruff check .
uv run ruff format .

# Type checking (uses ty, not mypy)
uv run ty check src tests

# Tests
uv run pytest tests/ -v          # all tests
uv run pytest tests/test_foo.py  # single file
uv run pytest -k "test_name"     # single test

# Pre-commit (runs ruff, ty, codespell, commitizen on push)
uv run pre-commit run --all-files

# Version bump (conventional commits required)
cz bump
```

## Architecture

- **`src/tcga_db/`** — main package
  - `config.py` — Pydantic settings (`AppSettings` with `DatabaseSettings`, `XenaSettings`), env var loading, and logging setup. Settings use `TCGA_` prefix with `__` nested delimiter (e.g. `TCGA_DATABASE__MEMORY=true`).
  - `database/connection.py` — `DatabaseManager` singleton for DuckDB (supports file-based and in-memory). Use `get_db_manager()` for the global instance.
  - `database/schema.py` — Declarative schema (9 tables for Phase 1). `TableSchema` dataclass with dependency tracking, topological creation order. `create_tables()` / `drop_all_tables()` for lifecycle.
  - `__init__.py` — calls `set_env_vars()` on import, exports `__version__`

### Phase 1 Database Schema (Expression + Survival)

System: `schema_version`, `data_imports`
Metadata: `genes`, `patients`, `samples`, `survival`, `tissue_mapping`
Core data: `expression_counts_wide`, `expression_tpm_wide` (samples × ~20K protein-coding genes, columns added dynamically during ETL)

Key relationships: `samples.patient_id → patients.patient_id`, `survival.patient_id → patients.patient_id`, expression tables FK to `samples.sample_id`.

## Key Conventions

- **Type checker is `ty`**, not `mypy`. Config in `[tool.ty.*]` sections of pyproject.toml. Rules are set to `all = "error"`.
- **Commits must follow Conventional Commits** — enforced by commitizen pre-push hook. Use `cz commit` for interactive commit.
- **CI** runs lint → format check → ty → pytest. Release workflow auto-bumps version on fix/feat/refactor/perf commits to main.
- **Logging**: use `from tcga_db.config import get_logger; logger = get_logger(__name__)` — never use `print()` or configure logging manually.
- **Settings**: use `from tcga_db.config import get_settings; settings = get_settings()` for database/xena configuration.
- **Line length**: 79 chars (ruff config).
- **Tests**: Set `TCGA_DATABASE__MEMORY=true` for in-memory DuckDB. Call `reset_db_manager()` + `reload_settings()` in fixtures for isolation.
