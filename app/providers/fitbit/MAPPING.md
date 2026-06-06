# Fitbit → Unified Mapping

## Sleep: Fitbit Sleep Logs API → `unified_sleep`

| Fitbit Field                    | Unified Field      | Transformation                          |
|---------------------------------|--------------------|-----------------------------------------|
| `logId`                         | (external_id)      | Cast to string                          |
| `dateOfSleep`                   | `sleep_date`       | ISO date string → date                  |
| `startTime`                     | `bedtime`          | ISO datetime → UTC datetime             |
| `endTime`                       | `wake_time`        | ISO datetime → UTC datetime             |
| `duration`                      | `total_seconds`    | Milliseconds → seconds (÷1000)          |
| `levels.summary.deep.minutes`   | `deep_seconds`     | Minutes → seconds (×60)                 |
| `levels.summary.light.minutes`  | `light_seconds`    | Minutes → seconds (×60)                 |
| `levels.summary.rem.minutes`    | `rem_seconds`      | Minutes → seconds (×60)                 |
| `levels.summary.wake.minutes`   | `awake_seconds`    | Minutes → seconds (×60)                 |
| `efficiency`                    | `sleep_score`      | Direct (0-100 scale)                    |

## Daily Activity: Fitbit Activity Summary → `unified_daily_metrics`

| Fitbit Field                  | Unified Field         | Transformation        |
|-------------------------------|-----------------------|-----------------------|
| (date from URL)               | `metric_date`         | ISO date              |
| `summary.steps`               | `steps`               | Direct                |
| `summary.caloriesOut`         | `calories_total`      | Direct (kcal)         |
| `summary.activityCalories`    | `calories_active`     | Direct (kcal)         |
| `summary.restingHeartRate`    | `resting_heart_rate`  | Direct (bpm)          |

## Heart Rate Intraday → `unified_heart_rate`

| Fitbit Field         | Unified Field   | Transformation                   |
|----------------------|-----------------|----------------------------------|
| `date` + `time`      | `recorded_at`   | Combined → UTC datetime          |
| `value`              | `bpm`           | Direct (beats per minute)        |

## Confidence

- Sleep stages from device sensor: **1.0**
- Activity calories (Fitbit-estimated): **0.9**
- Steps from accelerometer: **1.0**
- Resting HR from optical sensor: **1.0**

## Notes

- Fitbit sleep duration is in milliseconds; we convert to seconds for SI consistency.
- Sleep stage minutes are rounded by Fitbit; sub-minute resolution not available.
- Intraday HR requires special Fitbit API access (personal apps have it by default).
- Activity summary is per-day; no individual workout breakdown in this endpoint.
