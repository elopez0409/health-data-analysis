# Oura → Unified Field Mapping

## Data Source

- **API**: Oura Ring v2 REST API
- **Base URL**: `https://api.ouraring.com/v2/usercollection`
- **Auth**: OAuth 2.0

## Daily Sleep → `unified_sleep`

| Oura Field             | Unified Field    | Unit      | Conversion |
|------------------------|------------------|-----------|------------|
| `day`                  | `sleep_date`     | date      | ISO parse  |
| `score`                | `sleep_score`    | 0–100     | cast float |
| `total_sleep_duration` | `total_seconds`  | seconds   | none       |
| `deep_sleep_duration`  | `deep_seconds`   | seconds   | none       |
| `light_sleep_duration` | `light_seconds`  | seconds   | none       |
| `rem_sleep_duration`   | `rem_seconds`    | seconds   | none       |
| `awake_time`           | `awake_seconds`  | seconds   | none       |
| `bedtime_start`        | `bedtime`        | datetime  | tz-aware   |
| `bedtime_end`          | `wake_time`      | datetime  | tz-aware   |

All durations are natively in **seconds** from the Oura API — no unit conversion required.

## Daily Readiness → `unified_daily_metrics`

| Oura Field              | Unified Field     | Unit    | Conversion            |
|-------------------------|-------------------|---------|-----------------------|
| `day`                   | `metric_date`     | date    | ISO parse             |
| `score`                 | `readiness_score` | 0–100   | cast float            |
| `hrv_balance.value`     | `hrv_rmssd`       | ms      | extract from dict     |
| `temperature_deviation` | *(not mapped)*    | °C dev  | available in raw JSON |

Temperature deviation is stored in **Celsius deviation** from personal baseline.
Currently retained in raw payload only; no unified column exists yet.

## Daily Activity → `unified_daily_metrics`

| Oura Field                  | Unified Field    | Unit     | Conversion |
|-----------------------------|------------------|----------|------------|
| `day`                       | `metric_date`    | date     | ISO parse  |
| `steps`                     | `steps`          | count    | none       |
| `total_calories`            | `calories_total` | kcal     | cast float |
| `active_calories`           | `calories_active`| kcal     | cast float |
| `equivalent_walking_distance` | *(not mapped)* | meters   | raw only   |
| `score`                     | *(not mapped)*   | 0–100    | raw only   |

## Notes

- Readiness and activity both map to `unified_daily_metrics` with separate `source_record_id` values.
- Fields not yet mapped to unified columns are preserved in the raw JSONB `payload`.
- `confidence` is set to `1.0` for all Oura records (official device data).
