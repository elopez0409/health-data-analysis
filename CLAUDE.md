# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (with dev extras)
pip install -e ".[dev]"

# Start DB only (for local dev without full Docker)
docker compose up db -d

# Run migrations
alembic upgrade head

# Start the API server
uvicorn app.main:app --reload

# Run all tests
pytest

# Run a single test file or test
pytest tests/test_catapult.py
pytest tests/test_catapult.py::TestCatapultNormalizers::test_normalize_activity

# CLI commands
python -m app verify-all          # check all provider connections
python -m app pull strava         # pull latest data for a provider

# Full stack via Docker
docker compose up -d
docker compose exec app alembic upgrade head
docker compose exec app python -m app pull fitbit
```

## Architecture

This is a multi-provider health/fitness data ingestion backend. Data flows through two tiers:

**Raw Tier → Unified Tier**

1. `app/providers/<name>/client.py` — Each provider implements the `ProviderClient` ABC (`app/providers/base.py`). The `pull()` method returns `list[RawRecord]` (intermediate dataclass with `external_id`, `payload` dict, `payload_hash`). These are written to typed raw tables (e.g. `raw_strava_activities`, `raw_fitbit_sleep`) as JSONB blobs.

2. `app/normalizers/registry.py` — `normalize(raw_record)` dispatches by type to the correct `normalize_*` function in `app/providers/<name>/normalizer.py`. Returns a unified model instance (e.g. `UnifiedActivity`, `UnifiedSleep`) or `None`.

3. `app/models/raw.py` — All raw tables share `RawRecordMixin` columns: `user_id`, `provider`, `external_id`, `fetched_at`, `payload` (JSONB), `payload_hash`. Deduplication is enforced by `UniqueConstraint(provider, external_id, user_id)`.

4. `app/models/unified.py` — Typed unified tables. Every row carries `source` (provider name), `source_record_id` (FK to raw), `confidence`, `ingested_at`.

**Provider registration:** Clients are registered with `@ProviderRegistry.register(Provider.NAME)` (see `app/providers/registry.py`). Garmin is feature-flagged via `GARMIN_ENABLED=true` in `.env`. Catapult uses a static team API token (`CATAPULT_API_TOKEN`), not per-user OAuth — `get_authorize_url()` and `exchange_code()` raise `NotImplementedError`.

**Scheduling:** `app/jobs/scheduler.py` uses APScheduler. Per-provider pull jobs live in `app/jobs/<name>_pull.py`.

**Settings:** All config is in `app/config.py` via `pydantic-settings`. Reads from `.env`. Provider credentials follow `{PROVIDER}_CLIENT_ID` / `{PROVIDER}_CLIENT_SECRET` convention.

## Adding a New Provider

1. Create `app/providers/<name>/` with `auth.py`, `client.py`, `models.py`, `normalizer.py`
2. Implement `ProviderClient` and register with `@ProviderRegistry.register(Provider.NAME)`
3. Add `Provider.NAME` to the `Provider` enum in `app/schemas/common.py`
4. Add raw table(s) to `app/models/raw.py`, update `app/models/__init__.py`, create a migration under `migrations/versions/`
5. Add normalizer dispatch to `app/normalizers/registry.py`
6. Document field mappings in `MAPPING.md`
7. Add fixture JSON files under `tests/fixtures/` and tests in `tests/test_<name>.py`

## Testing Patterns

Tests use `respx` for mocking `httpx` HTTP calls and `pytest-asyncio` (configured `asyncio_mode = "auto"`). Fixture JSON files live in `tests/fixtures/`. Normalizer tests construct raw model instances directly (no DB needed).

## Data / ML Extras

- `app/explore.py` — Polars helpers for querying unified tables directly from notebooks
- `notebooks/` — Jupyter notebooks for coverage/agreement/signal analysis
- `hr_selection/` — Separate ML subpackage for personal HR model training; entry point `hr-train` CLI
- Research datasets (PPG-DaLiA, BigIdeasLab STEP) are git-ignored; download with `scripts/fetch_*.py`
