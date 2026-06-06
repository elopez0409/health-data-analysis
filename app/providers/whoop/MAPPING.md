# WHOOP → Unified Mapping

## Source: WHOOP Developer API (`https://api.prod.whoop.com/developer`)

### Recovery (`GET /v1/recovery`) → `UnifiedDailyMetrics`

| WHOOP Field                  | Unified Field         | Transformation                          |
|------------------------------|-----------------------|-----------------------------------------|
| `cycle_id`                   | (external_id on raw)  | Cast to string                          |
| `created_at`                 | `metric_date`         | Extract date portion                    |
| `score.recovery_score`       | `recovery_score`      | Direct (0–100 percentage)               |
| `score.hrv_rmssd_milli`      | `hrv_rmssd`           | ÷ 1000 (WHOOP milli → standard ms)     |
| `score.resting_heart_rate`   | `resting_heart_rate`  | Direct (bpm)                            |
| `score.skin_temp_celsius`    | —                     | Not mapped (no unified field)           |
| `score.spo2_percentage`      | —                     | Not mapped (no unified field)           |

### Sleep (`GET /v1/activity/sleep`) → `UnifiedSleep`

| WHOOP Field                                        | Unified Field    | Transformation                    |
|----------------------------------------------------|------------------|-----------------------------------|
| `id`                                               | (external_id)    | Cast to string                    |
| `start`                                            | `bedtime`        | Direct (ISO 8601 timestamp)       |
| `end`                                              | `wake_time`      | Direct (ISO 8601 timestamp)       |
| `start`                                            | `sleep_date`     | Extract date portion              |
| `score.stage_summary.total_light_sleep_time_milli` | `light_seconds`  | ÷ 1000 (milliseconds → seconds)  |
| `score.stage_summary.total_slow_wave_sleep_time_milli` | `deep_seconds`   | ÷ 1000 (milliseconds → seconds)  |
| `score.stage_summary.total_rem_sleep_time_milli`   | `rem_seconds`    | ÷ 1000 (milliseconds → seconds)  |
| `score.stage_summary.total_awake_time_milli`       | `awake_seconds`  | ÷ 1000 (milliseconds → seconds)  |
| (sum of light + deep + rem)                        | `total_seconds`  | Sum of sleep stages ÷ 1000       |
| `score.sleep_performance_percentage`               | `sleep_score`    | Direct (0–100 percentage)         |

### Workout (`GET /v1/activity/workout`) → `UnifiedActivity`

| WHOOP Field                    | Unified Field            | Transformation                             |
|--------------------------------|--------------------------|--------------------------------------------|
| `id`                           | (external_id on raw)     | Cast to string                             |
| `sport_id`                     | `activity_type`          | Prefixed as `whoop_sport_{id}`             |
| `start`                        | `started_at`             | Direct (ISO 8601 timestamp)                |
| `end`                          | `ended_at`               | Direct (ISO 8601 timestamp)                |
| (end - start)                  | `duration_seconds`       | Computed from timestamps                   |
| `score.kilojoule`              | `calories`               | × 0.239006 (kJ → kcal)                    |
| `score.strain`                 | —                        | Not directly mapped (WHOOP-specific 0–21)  |
| `score.average_heart_rate`     | `avg_heart_rate_bpm`     | Direct (bpm)                               |
| `score.max_heart_rate`         | `max_heart_rate_bpm`     | Direct (bpm)                               |
| `score.distance_meter`         | `distance_meters`        | Direct (already meters)                    |
| `score.altitude_gain_meter`    | `elevation_gain_meters`  | Direct (already meters)                    |

## Confidence

- All sensor-derived fields: **1.0**

## Notes

- WHOOP uses **milliseconds** for all duration fields in sleep stage summaries. We divide by 1000 to convert to seconds for the unified schema.
- `hrv_rmssd_milli` is HRV in a milli-unit representation. Divide by 1000 to get the standard RMSSD value in milliseconds used across other providers.
- WHOOP `strain` is a proprietary 0–21 scale; it does not map directly to any unified field but could be stored in `strain_score` on `UnifiedDailyMetrics` if desired.
- WHOOP pagination uses a `nextToken` query parameter (not page-based). The response includes a `next_token` field; when absent, all data has been fetched.
- `sport_id` is a numeric identifier for activity type (e.g., 0 = Running, 1 = Cycling). We prefix with `whoop_sport_` for the unified `activity_type`.
