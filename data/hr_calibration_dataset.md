# HR Calibration Dataset — Data Dictionary

Synthetic longitudinal resting heart rate readings for baseline training,
personal offset convergence, and walk-forward backtesting.

## Files

| File | Description |
|------|-------------|
| `hr_calibration_dataset.csv` | Long-format device readings (model input) |
| `hr_calibration_truth.csv` | Per-user ground truth (evaluation only) |
| `hr_calibration_dataset.md` | This data dictionary |

## `hr_calibration_dataset.csv`

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Simulated user identifier (`user_01` … `user_20`) |
| `date` | date (ISO) | Night of measurement, consecutive from `2026-01-01` |
| `device_type` | string | `chest_strap` (trusted anchor) or `wrist` (device to calibrate) |
| `hr_reading` | float | Resting heart rate in bpm |
| `is_resting` | bool | Always `True` (resting-state dataset) |

Missing readings (~10.1% per device) are omitted entirely.

## `hr_calibration_truth.csv`

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Matches readings file |
| `true_baseline` | float | User's stable resting HR baseline (bpm), drawn from N(62.0, 8.0) |
| `personal_offset` | float | Wrist bias vs chest strap (bpm), drawn from N(0.0, 5.0) |

## Generative model

Per user:
- `true_baseline ~ N(62.0, 8.0)`
- `personal_offset ~ N(0.0, 5.0)`

Per night:
- `true_hr_night = true_baseline + N(0, 2.0)`
- `chest_strap = true_hr_night + N(0, 1.5)`
- `wrist = true_hr_night + personal_offset + N(0, 4.5)`
- Each device independently missing with probability 10%

## Dataset summary

| Stat | Value |
|------|-------|
| Users | 20 |
| Nights per user | 90 |
| Total rows | 3237 |
| Expected rows (no missing) | 3600 |
| Observed missing rate | 10.1% |
| Observed population offset mean | 2.1 bpm |
| Observed population offset std | 7.333 bpm |
| Random seed | 42 |

## Usage

- **Population prior:** pool nightly `wrist - chest_strap` deltas across users.
- **Personal calibration:** feed per-user deltas into `OffsetState` in date order.
- **Backtest:** walk-forward expanding window; predict corrected wrist vs chest anchor.
