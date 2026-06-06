# Garmin → Unified Mapping

## Source: Garmin Health API Webhook Push

Garmin uses SI units natively — no unit conversions are required.

### Activities (`activities` payload)

| Garmin Field                          | Unified Field          | Transformation                      |
|---------------------------------------|------------------------|--------------------------------------|
| `activityId`                          | (external_id on raw)   | Cast to string                       |
| `activityType`                        | `activity_type`        | Direct                               |
| `startTimeInSeconds`                  | `started_at`           | Epoch seconds → datetime (UTC)       |
| (computed)                            | `ended_at`             | started_at + durationInSeconds       |
| `durationInSeconds`                   | `duration_seconds`     | Seconds (no conversion)              |
| `distanceInMeters`                    | `distance_meters`      | Meters (no conversion)               |
| `activeKilocalories`                  | `calories`             | kcal (no conversion)                 |
| `averageHeartRateInBeatsPerMinute`    | `avg_heart_rate_bpm`   | bpm (no conversion)                  |
| `maxHeartRateInBeatsPerMinute`        | `max_heart_rate_bpm`   | bpm (no conversion)                  |
| —                                     | `elevation_gain_meters`| Not provided by Garmin push          |
| —                                     | `title`                | Not provided by Garmin push          |

### Sleep (`sleeps` payload)

| Garmin Field                    | Unified Field    | Transformation                 |
|---------------------------------|------------------|--------------------------------|
| `startTimeInSeconds`            | `bedtime`        | Epoch seconds → datetime (UTC) |
| (computed)                      | `wake_time`      | bedtime + durationInSeconds    |
| `startTimeInSeconds`            | `sleep_date`     | Epoch seconds → date (UTC)     |
| `durationInSeconds`             | `total_seconds`  | Seconds (no conversion)        |
| `deepSleepDurationInSeconds`    | `deep_seconds`   | Seconds (no conversion)        |
| `lightSleepDurationInSeconds`   | `light_seconds`  | Seconds (no conversion)        |
| `remSleepDurationInSeconds`     | `rem_seconds`    | Seconds (no conversion)        |
| `awakeDurationInSeconds`        | `awake_seconds`  | Seconds (no conversion)        |
| —                               | `sleep_score`    | Not provided by Garmin push    |

## Confidence

- All fields come directly from Garmin device sensors: **1.0**

## Notes

- Garmin delivers data via webhook push — there is no polling API.
- All timestamps are Unix epoch seconds (UTC).
- Distance is in meters, energy in kilocalories, heart rate in bpm.
- OAuth 1.0a is used for authorization (consumer key/secret, no token refresh).
- The feature flag `GARMIN_ENABLED` must be set to `true` to enable the provider.
