# Health Data Unification Backend

Multi-provider health/fitness data ingestion and normalization. Pulls data from wearable APIs into a unified PostgreSQL/TimescaleDB store with full source provenance.

## Quick Start

```bash
# 1. Copy env and fill in your API credentials
cp .env.example .env

# 2. Start Postgres (TimescaleDB) + app
docker compose up -d

# 3. Run migrations (happens automatically on docker compose up, or manually):
docker compose exec app alembic upgrade head

# 4. Connect providers via OAuth (open in browser):
#    http://localhost:8000/auth/strava/start
#    http://localhost:8000/auth/fitbit/start
#    http://localhost:8000/auth/oura/start
#    http://localhost:8000/auth/withings/start
#    http://localhost:8000/auth/whoop/start

# 5. Verify connections
docker compose exec app python -m app verify-all

# 6. Pull data
docker compose exec app python -m app pull strava
docker compose exec app python -m app pull fitbit
```

## Local Development (without Docker)

```bash
# Install dependencies
pip install -e ".[dev]"

# Start TimescaleDB (requires local install or use docker for just the DB):
docker compose up db -d

# Run migrations
alembic upgrade head

# Start the app
uvicorn app.main:app --reload

# Run tests
pytest
```

## OAuth Provider Registration

| Provider | Dev Portal | Notes |
|----------|-----------|-------|
| Strava | https://www.strava.com/settings/api | Set callback to `http://localhost:8000/auth/strava/callback` |
| Fitbit | https://dev.fitbit.com/apps | Personal app type for intraday HR access |
| Oura | https://cloud.ouraring.com/v2/docs | Create application in Oura developer portal |
| Withings | https://developer.withings.com/ | Register app, set callback URL |
| WHOOP | https://developer.whoop.com/ | Beta API, requires developer access |
| Garmin | https://developerportal.garmin.com/ | Requires manual program approval (weeks). Feature-flagged. |

## .env Configuration

See `.env.example` for all configuration options. Key settings:

- `DATABASE_URL` — Postgres connection string (async: `postgresql+asyncpg://...`)
- `{PROVIDER}_CLIENT_ID` / `{PROVIDER}_CLIENT_SECRET` — OAuth credentials per provider
- `GARMIN_ENABLED=true` — Enable Garmin (disabled by default pending approval)
- `DEFAULT_USER_ID` — UUID for the single-user prototype

## Architecture

```
Raw Tier (jsonb)          →   Normalizers   →   Unified Tier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
raw_strava_activities     →   Strava norm   →   unified_activities
raw_fitbit_sleep          →   Fitbit norm   →   unified_sleep
raw_oura_daily_sleep      →   Oura norm     →   unified_sleep
raw_fitbit_heart_rate     →   Fitbit norm   →   unified_heart_rate (hypertable)
raw_*_activity            →   * norm        →   unified_daily_metrics
raw_withings_weight       →   Withings norm →   unified_body_metrics
```

Every unified row carries:
- `source` — which provider it came from
- `source_record_id` — FK back to the raw table for full provenance
- `confidence` — 1.0 for direct measurements, lower for estimates
- `ingested_at` — when we processed it

## CLI Commands

```bash
python -m app verify-all          # Check all provider connections
python -m app pull strava         # Pull latest Strava data
python -m app pull fitbit         # Pull latest Fitbit data
```

## Exploration

The `app/explore.py` module provides helpers for querying unified data:

```python
from app.explore import load_from_postgres

# In notebooks, use direct Postgres queries via Polars:
df = load_from_postgres("SELECT * FROM unified_activities WHERE source = 'strava'")
```

See `notebooks/` for starter analyses:
- `01_coverage.ipynb` — Data inventory across providers
- `02_cross_provider_agreement.ipynb` — Agreement between overlapping metrics
- `03_signal_relationships.ipynb` — Sleep/HRV/recovery vs activity correlations

## Datasets

Raw research datasets are **not** committed to git (they're git-ignored). Download
them locally with the helper scripts:

```bash
# Open-access wearable HR datasets (PPG-DaLiA, GalaxyPPG)
python scripts/fetch_hr_datasets.py

# BigIdeasLab_STEP smartwatch HR (PhysioNet, credentialed access)
export PHYSIONET_USERNAME=yourname      # password is prompted (hidden) if unset
python scripts/fetch_bigideas_step.py
```

`fetch_bigideas_step.py` places files under
`physionet.org/files/bigideaslab-step-hr-smartwatch/1.0/`, where
`hr_selection/config.py` expects them, and verifies SHA-256 checksums.

> **Restricted data:** BigIdeasLab_STEP is published under the PhysioNet
> Restricted Health Data License. You must hold a credentialed PhysioNet account
> and sign the project's Data Use Agreement. **Never commit or share this data** —
> the `physionet.org/` directory is git-ignored on purpose.

## Adding a New Provider

1. Create `app/providers/<name>/` with `auth.py`, `client.py`, `models.py`, `normalizer.py`
2. Implement `ProviderClient` protocol and register with `@ProviderRegistry.register()`
3. Add raw table(s) to `app/models/raw.py` and create a migration
4. Write normalizer(s) and document in `MAPPING.md`
5. Add fixture-based tests in `tests/`
6. Add OAuth credentials to `.env.example`

## Project Structure

```
app/
├── main.py              # FastAPI app
├── config.py            # Settings from .env
├── db.py                # Async SQLAlchemy engine
├── cli.py               # Typer CLI (verify-all, pull)
├── explore.py           # Polars query helpers
├── rate_limit.py        # Token-bucket per provider
├── models/              # SQLAlchemy models (raw + unified)
├── providers/           # Per-provider clients + normalizers
├── jobs/                # Pull jobs + APScheduler
├── routes/              # FastAPI routers (OAuth, ingest, webhooks)
└── webhooks/            # Webhook receivers (Garmin)
```
