# Withings → Unified Mapping

## Weight: Measure - Getmeas (meastype 1, 6, 76) → `unified_body_metrics`

| Withings Field      | Type Code | Unified Field    | Transformation                          |
|---------------------|-----------|------------------|-----------------------------------------|
| `grpid`             | —         | (external_id)    | Cast to string                          |
| `date`              | —         | `measured_at`    | Epoch → UTC datetime                    |
| measure type 1      | 1         | `weight_kg`      | `value * 10^unit` (already kg)          |
| measure type 6      | 6         | `body_fat_pct`   | `value * 10^unit` (fat ratio %)         |
| measure type 76     | 76        | `muscle_mass_kg` | `value * 10^unit` (already kg)          |

## Blood Pressure: Measure - Getmeas (meastype 9, 10, 11) → `unified_body_metrics`

| Withings Field      | Type Code | Unified Field    | Transformation                          |
|---------------------|-----------|------------------|-----------------------------------------|
| `grpid`             | —         | (external_id)    | Cast to string                          |
| `date`              | —         | `measured_at`    | Epoch → UTC datetime                    |
| measure type 10     | 10        | `systolic_bp`    | `value * 10^unit` (mmHg)               |
| measure type 9      | 9         | `diastolic_bp`   | `value * 10^unit` (mmHg)               |
| measure type 11     | 11        | (heart_pulse)    | Available but not mapped to unified     |

## Sleep: Sleep v2 - Get → `unified_sleep`

| Withings Field          | Unified Field    | Transformation                          |
|-------------------------|------------------|-----------------------------------------|
| `id`                    | (external_id)    | Cast to string                          |
| `startdate`             | `bedtime`        | Epoch → UTC datetime                    |
| `enddate`               | `wake_time`      | Epoch → UTC datetime                    |
| (enddate)               | `sleep_date`     | wake_time.date()                        |
| enddate − startdate     | `total_seconds`  | Difference in seconds                   |
| `deepsleepduration`     | `deep_seconds`   | Already in seconds                      |
| `lightsleepduration`    | `light_seconds`  | Already in seconds                      |
| `remsleepduration`      | `rem_seconds`    | Already in seconds                      |
| `wakeupduration`        | `awake_seconds`  | Already in seconds                      |
| `hr_average`            | —                | Not mapped (no unified field)           |

## Withings Measure Value System

Withings stores all numeric measurements as an integer pair `(value, unit)` where the
real value equals `value * 10^unit`.

Examples:
- Weight 78.5 kg → `value=78500, unit=-3` → 78500 × 10⁻³ = 78.5
- Fat ratio 18.5% → `value=1850, unit=-2` → 1850 × 10⁻² = 18.5
- BP 120 mmHg → `value=120, unit=0` → 120 × 10⁰ = 120

## Confidence

- Scale-measured weight/body comp: **1.0**
- BPM cuff measurements: **1.0**
- Sleep tracking (Sleep Monitor / Sleep Mat): **1.0**

## Notes

- Withings uses epoch (Unix) timestamps throughout; no ISO 8601 date strings.
- All durations (sleep stages) are in seconds — no unit conversion needed.
- The `category` field distinguishes real measurements (1) from user objectives (2).
- The `attrib` field indicates measurement attribution (0 = device, 1 = device ambiguous, 2 = manual, 4 = manual during creation).
- Heart pulse from BP measurements (type 11) is available but has no target field in `unified_body_metrics`.
