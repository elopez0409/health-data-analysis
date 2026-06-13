# Database Schema

PostgreSQL with TimescaleDB. Baseline migration: `migrations/versions/001_baseline.py`.

## Extensions

- `timescaledb` — hypertable support for time-series data

## Common Patterns

### Raw Tier (`RawRecordMixin`)

All raw tables share these columns:

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | Primary key, `gen_random_uuid()` |
| `user_id` | UUID | Owner |
| `provider` | VARCHAR(50) | Source provider name |
| `external_id` | VARCHAR | Provider's record ID |
| `fetched_at` | TIMESTAMPTZ | When the record was fetched, default `now()` |
| `payload` | JSONB | Full API response |
| `payload_hash` | VARCHAR(64) | Content hash for deduplication |

Unique constraint: `(provider, external_id, user_id)`.

### Unified Tier (`UnifiedMixin`)

All unified tables share these columns:

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | Primary key |
| `user_id` | UUID | Owner |
| `source` | VARCHAR(50) | Provider name |
| `source_record_id` | UUID | FK to raw table row |
| `ingested_at` | TIMESTAMPTZ | When normalized, default `now()` |
| `confidence` | FLOAT | 1.0 for direct measurements, lower for estimates |

Unique constraint: `(source, source_record_id)`.

---

## Auth & Ingestion

### `provider_tokens`

OAuth credentials per user/provider.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | Primary key |
| `user_id` | UUID | |
| `provider` | VARCHAR(50) | |
| `access_token` | VARCHAR | |
| `refresh_token` | VARCHAR | Nullable |
| `token_type` | VARCHAR(50) | Default `Bearer` |
| `expires_at` | TIMESTAMPTZ | Nullable |
| `scopes` | VARCHAR | Nullable |
| `extra` | JSONB | Provider-specific metadata |
| `updated_at` | TIMESTAMPTZ | Default `now()` |

Unique constraint: `(user_id, provider)`.

### `ingestion_cursors`

Tracks last-fetched position per user/provider/resource for incremental pulls.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | Primary key |
| `user_id` | UUID | |
| `provider` | VARCHAR(50) | |
| `resource` | VARCHAR(100) | e.g. `activities`, `sleep` |
| `last_value` | VARCHAR | Cursor value (timestamp or ID) |
| `updated_at` | TIMESTAMPTZ | Default `now()` |

Unique constraint: `(user_id, provider, resource)`.

---

## Raw Tier Tables

| Table | Provider | Resource |
|-------|----------|----------|
| `raw_strava_activities` | Strava | Activities |
| `raw_fitbit_sleep` | Fitbit | Sleep |
| `raw_fitbit_heart_rate` | Fitbit | Intraday heart rate |
| `raw_fitbit_activity` | Fitbit | Daily activity |
| `raw_oura_daily_sleep` | Oura | Daily sleep |
| `raw_oura_daily_readiness` | Oura | Daily readiness |
| `raw_oura_daily_activity` | Oura | Daily activity |
| `raw_withings_weight` | Withings | Weight |
| `raw_withings_sleep` | Withings | Sleep |
| `raw_withings_blood_pressure` | Withings | Blood pressure |
| `raw_whoop_recovery` | WHOOP | Recovery |
| `raw_whoop_sleep` | WHOOP | Sleep |
| `raw_whoop_workout` | WHOOP | Workouts |
| `raw_garmin_activity` | Garmin | Activities |
| `raw_garmin_sleep` | Garmin | Sleep |

All use the raw tier column pattern above. SQLAlchemy models in `app/models/raw.py`.

---

## Unified Tier Tables

### `unified_activities`

| Column | Type | Notes |
|--------|------|-------|
| *(mixin columns)* | | |
| `activity_type` | VARCHAR(100) | e.g. `Run`, `Ride` |
| `started_at` | TIMESTAMPTZ | |
| `ended_at` | TIMESTAMPTZ | Nullable |
| `duration_seconds` | FLOAT | Nullable |
| `distance_meters` | FLOAT | Nullable |
| `calories` | FLOAT | Nullable |
| `avg_heart_rate_bpm` | FLOAT | Nullable |
| `max_heart_rate_bpm` | FLOAT | Nullable |
| `elevation_gain_meters` | FLOAT | Nullable |
| `title` | VARCHAR | Nullable |

### `unified_sleep`

| Column | Type | Notes |
|--------|------|-------|
| *(mixin columns)* | | |
| `sleep_date` | DATE | |
| `bedtime` | TIMESTAMPTZ | Nullable |
| `wake_time` | TIMESTAMPTZ | Nullable |
| `total_seconds` | FLOAT | Nullable |
| `deep_seconds` | FLOAT | Nullable |
| `light_seconds` | FLOAT | Nullable |
| `rem_seconds` | FLOAT | Nullable |
| `awake_seconds` | FLOAT | Nullable |
| `sleep_score` | FLOAT | Nullable |

### `unified_heart_rate` (TimescaleDB hypertable)

Partitioned on `recorded_at` with 7-day chunks.

| Column | Type | Notes |
|--------|------|-------|
| *(mixin columns)* | | |
| `recorded_at` | TIMESTAMPTZ | Hypertable partition key |
| `bpm` | INTEGER | |
| `context` | VARCHAR(50) | Nullable, e.g. `resting`, `active` |

### `unified_daily_metrics`

| Column | Type | Notes |
|--------|------|-------|
| *(mixin columns)* | | |
| `metric_date` | DATE | |
| `steps` | INTEGER | Nullable |
| `calories_total` | FLOAT | Nullable |
| `calories_active` | FLOAT | Nullable |
| `hrv_rmssd` | FLOAT | Nullable |
| `resting_heart_rate` | FLOAT | Nullable |
| `readiness_score` | FLOAT | Nullable |
| `strain_score` | FLOAT | Nullable |
| `recovery_score` | FLOAT | Nullable |

### `unified_body_metrics`

| Column | Type | Notes |
|--------|------|-------|
| *(mixin columns)* | | |
| `measured_at` | TIMESTAMPTZ | |
| `weight_kg` | FLOAT | Nullable |
| `body_fat_pct` | FLOAT | Nullable |
| `muscle_mass_kg` | FLOAT | Nullable |
| `systolic_bp` | FLOAT | Nullable |
| `diastolic_bp` | FLOAT | Nullable |

SQLAlchemy models in `app/models/unified.py`. Pydantic schemas in `app/schemas/`.

---

## Personal HR Model

Per-user state for the living HR model (migration `002_personal_hr`).

### `personal_hr_state`

One row per `(user_id, source)`. Stores learned device offsets and habitual baseline.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | Primary key |
| `user_id` | UUID | Owner |
| `source` | VARCHAR(50) | Provider name (e.g. `fitbit`, `garmin`) |
| `trusted_source` | VARCHAR(50) | Nullable; per-user anchor device |
| `offset_mean` | FLOAT | Mean offset vs trusted device (bpm) |
| `offset_var` | FLOAT | Running variance of offset |
| `n_samples` | INT | Number of offset observations |
| `ci_low` / `ci_high` | FLOAT | Confidence interval on offset mean |
| `ewma_offset` | FLOAT | EWMA track for re-baselining after shifts |
| `last_updated` | TIMESTAMPTZ | Last nightly update |
| `baseline_profile` | JSONB | Resting HR by context + circadian curve |

Unique constraint: `(user_id, source)`.

### `hr_anomalies`

Detected sudden shifts in device offset (e.g. firmware update).

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | Primary key |
| `user_id` | UUID | Owner |
| `source` | VARCHAR(50) | Device that shifted |
| `detected_at` | TIMESTAMPTZ | When anomaly was detected |
| `shift_bpm` | FLOAT | Magnitude of shift |
| `prev_offset` | FLOAT | Prior offset mean |
| `new_offset` | FLOAT | Recent batch mean |
| `severity` | VARCHAR(20) | `warning` or `critical` |
| `resolved` | BOOLEAN | Whether shift was acknowledged/resolved |
| `details` | JSONB | Full anomaly check result |

SQLAlchemy models in `app/models/personal.py`.

---

## Entity Relationships

```
provider_tokens ──► user_id
ingestion_cursors ──► user_id + provider + resource

raw_* tables ──► (normalize) ──► unified_* tables
                                    source_record_id ──► raw_*.id

unified_heart_rate ──► (nightly personal_hr job) ──► personal_hr_state
                                                      hr_anomalies
```
