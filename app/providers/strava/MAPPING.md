# Strava → Unified Activities Mapping

## Source: Strava Activities API (`GET /api/v3/athlete/activities`)

| Strava Field             | Unified Field            | Transformation                      |
|--------------------------|--------------------------|-------------------------------------|
| `id`                     | (external_id on raw)     | Cast to string                      |
| `name`                   | `title`                  | Direct                              |
| `sport_type` / `type`    | `activity_type`          | sport_type preferred, fallback type |
| `start_date`             | `started_at`             | ISO8601 → datetime (UTC)            |
| (computed)               | `ended_at`               | started_at + elapsed_time           |
| `elapsed_time`           | `duration_seconds`       | Seconds (no conversion)             |
| `distance`               | `distance_meters`        | Already in meters                   |
| `total_elevation_gain`   | `elevation_gain_meters`  | Already in meters                   |
| `average_heartrate`      | `avg_heart_rate_bpm`     | bpm (no conversion)                 |
| `max_heartrate`          | `max_heart_rate_bpm`     | bpm (no conversion)                 |
| `calories` / `kilojoules`| `calories`               | calories preferred; kJ used 1:1 for rides |

## Confidence

- All fields directly from sensor: **1.0**
- Calories from kilojoules estimate: **0.9**

## Notes

- Strava reports distance in meters for all activity types.
- `elapsed_time` includes rest/pause; `moving_time` excludes it. We store elapsed_time as the canonical duration.
- Heart rate is only present if the activity was recorded with an HR sensor.
- The `sport_type` field is more specific than `type` (e.g., "TrailRun" vs "Run").
