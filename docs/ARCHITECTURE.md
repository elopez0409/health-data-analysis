# System Architecture

Multi-provider health/fitness data ingestion and normalization backend, plus an offline ML pipeline for HR source selection.

## High-Level Overview

```
Wearable APIs / Webhooks
        │
        ▼
  Provider Clients  (OAuth, rate-limited HTTP)
        │
        ▼
   Raw Tier (JSONB)  ──►  Normalizers  ──►  Unified Tier (Postgres/TimescaleDB)
        │                                              │
        │                                              ▼
        │                                    app/explore.py (Polars)
        │                                              │
        └──────────────────────────────────────────────┼──► notebooks/
                                                       └──► hr_selection/ (ML)
```

## Backend (`app/`)

### Entry Points

| Component | Path | Role |
|-----------|------|------|
| FastAPI app | `app/main.py` | HTTP server, lifespan hooks, route registration |
| CLI | `app/cli.py` | `python -m app verify-all`, `python -m app pull <provider>` |
| Scheduler | `app/jobs/scheduler.py` | APScheduler: Strava every 15 min, Fitbit every 30 min |

### Configuration

- `app/config.py` — settings from `.env` (database URL, OAuth credentials, `DEFAULT_USER_ID`, feature flags like `GARMIN_ENABLED`)
- `app/db.py` — async SQLAlchemy engine and session factory

### Data Flow: Raw → Unified

1. **Pull jobs** (`app/jobs/`) fetch data from provider APIs and upsert into raw tables.
2. **Raw tier** (`app/models/raw.py`) stores full API payloads as JSONB with provenance (`user_id`, `provider`, `external_id`, `payload_hash`).
3. **Normalizers** (`app/providers/*/normalizer.py`) map raw records to unified models.
4. **Normalizer dispatch** (`app/normalizers/registry.py`) routes each raw model class to its normalizer.
5. **Unified tier** (`app/models/unified.py`) stores normalized, cross-provider metrics with `source`, `source_record_id`, and `confidence`.

Every unified row links back to its raw record via `source_record_id` for full provenance.

### Providers

Each provider lives in `app/providers/<name>/`:

| File | Purpose |
|------|---------|
| `auth.py` | OAuth flow helpers |
| `client.py` | API client implementing `ProviderClient` protocol |
| `models.py` | Provider-specific Pydantic models |
| `normalizer.py` | Raw → unified mapping |
| `MAPPING.md` | Field-level mapping documentation |

Providers register via `@ProviderRegistry.register()` in `app/providers/registry.py`. Garmin is feature-flagged (`GARMIN_ENABLED`).

Supported providers: Strava, Fitbit, Oura, Withings, WHOOP, Garmin.

### HTTP Routes

| Router | Path prefix | Purpose |
|--------|-------------|---------|
| `app/routes/oauth_callback.py` | `/auth/{provider}/...` | OAuth start + callback |
| `app/routes/ingest.py` | `/ingest/...` | Manual ingestion triggers |
| `app/webhooks/garmin.py` | `/webhooks/garmin` | Garmin push notifications |

### Supporting Infrastructure

- **Token store** (`app/providers/token_store.py`) — OAuth tokens in `provider_tokens` table
- **Rate limiting** (`app/rate_limit.py`) — per-provider token-bucket rate limiters
- **Schemas** (`app/schemas/`) — Pydantic request/response models (`common.py`, `ingest.py`)
- **Logging** (`app/logging.py`) — structured logging setup

### Data Exploration

`app/explore.py` loads unified tables into Polars DataFrames:

- `load_unified(table, user_id, since, until)` — async, filtered by user and date range
- `load_unified_sync(...)` — notebook-friendly sync wrapper
- `load_from_postgres(query)` — direct SQL via connectorx (fastest for ad-hoc queries)

## ML Pipeline (`hr_selection/`)

Offline package for **HR source selection**: given multiple wearable HR sources in a time window, predict which source is closest to the reference (ECG/Polar).

See [ML_SYSTEM.md](ML_SYSTEM.md) for full pipeline documentation.

Relationship to the backend:
- The backend ingests and unifies wearable data from real providers.
- `hr_selection` trains on public research datasets (BigIdeasLab, GalaxyPPG, PPG-DaLiA) with synthetic or real data.
- The **personal living model** (`hr_selection/personal/`, `app/jobs/personal_hr.py`) runs nightly on `unified_heart_rate`, learning per-user device offsets and detecting anomalies. See [ML_SYSTEM.md](ML_SYSTEM.md).

## Database

PostgreSQL with TimescaleDB extension. See [SCHEMA.md](SCHEMA.md) for table reference.

Key design:
- **Raw tier** — one table per provider resource, JSONB payloads
- **Unified tier** — five normalized tables (activities, sleep, heart_rate, daily_metrics, body_metrics)
- **`unified_heart_rate`** is a TimescaleDB hypertable partitioned on `recorded_at` (7-day chunks)

## Project Layout

```
health-data-analysis/
├── app/                    # FastAPI backend
│   ├── main.py
│   ├── cli.py
│   ├── explore.py
│   ├── config.py
│   ├── db.py
│   ├── models/             # SQLAlchemy (raw + unified + tokens)
│   ├── providers/          # Per-provider clients + normalizers
│   ├── normalizers/        # Dispatch registry
│   ├── jobs/               # Pull jobs + scheduler
│   ├── routes/             # OAuth + ingest
│   ├── schemas/            # Pydantic models
│   └── webhooks/           # Garmin webhooks
├── hr_selection/           # ML: HR source selection
├── migrations/             # Alembic (001_baseline)
├── docs/                   # Architecture, schema, ML docs
├── notebooks/              # Exploratory analysis
├── scripts/                # Utility scripts
├── tests/                  # pytest suite
└── data/                   # ML datasets and artifacts
```

## Adding a New Provider

1. Create `app/providers/<name>/` with `auth.py`, `client.py`, `models.py`, `normalizer.py`
2. Implement `ProviderClient` protocol and register with `@ProviderRegistry.register()`
3. Add raw table(s) to `app/models/raw.py` and create an Alembic migration
4. Add normalizer dispatch in `app/normalizers/registry.py`
5. Write normalizer(s) and document in `MAPPING.md`
6. Add fixture-based tests in `tests/`
7. Add OAuth credentials to `.env.example`
8. Update `docs/SCHEMA.md` and `docs/ARCHITECTURE.md`
