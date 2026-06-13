"""Synthetic longitudinal resting-HR calibration dataset.

Generates nightly paired chest-strap vs wrist readings for N users over
90 consecutive nights. Designed to train/validate population priors and
demonstrate personal offset convergence + walk-forward backtesting.

Output schema (long format)::
    user_id, date, device_type, hr_reading, is_resting
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from hr_selection import config

# --------------------------------------------------------------------------
# Default generative parameters (HR preset)
# --------------------------------------------------------------------------
N_USERS = 20
N_NIGHTS = 90
START_DATE = date(2026, 1, 1)

TRUE_BASELINE_MEAN = 62.0
TRUE_BASELINE_STD = 8.0
PERSONAL_OFFSET_MEAN = 0.0
PERSONAL_OFFSET_STD = 5.0
NIGHT_VAR_STD = 2.0
CHEST_NOISE_STD = 1.5
WRIST_NOISE_STD = 4.5
MISSING_RATE = 0.10

DEVICE_CHEST = "chest_strap"
DEVICE_WRIST = "wrist"


@dataclass(frozen=True)
class MetricConfig:
    """Generative parameters for a paired anchor-vs-wrist calibration metric."""

    name: str
    unit: str
    baseline_mean: float
    baseline_std: float
    offset_mean: float
    offset_std: float
    night_var_std: float
    anchor_noise_std: float
    wrist_noise_std: float
    decimal_places: int = 1


HR_METRIC = MetricConfig(
    name="heart_rate",
    unit="bpm",
    baseline_mean=TRUE_BASELINE_MEAN,
    baseline_std=TRUE_BASELINE_STD,
    offset_mean=PERSONAL_OFFSET_MEAN,
    offset_std=PERSONAL_OFFSET_STD,
    night_var_std=NIGHT_VAR_STD,
    anchor_noise_std=CHEST_NOISE_STD,
    wrist_noise_std=WRIST_NOISE_STD,
    decimal_places=1,
)

HRV_METRIC = MetricConfig(
    name="hrv_rmssd",
    unit="ms",
    baseline_mean=45.0,
    baseline_std=12.0,
    offset_mean=0.0,
    offset_std=8.0,
    night_var_std=4.0,
    anchor_noise_std=2.0,
    wrist_noise_std=6.0,
    decimal_places=1,
)


@dataclass
class CalibrationManifest:
    """Summary of a generated calibration dataset."""

    n_users: int
    n_nights: int
    n_rows: int
    n_expected: int
    missing_rate: float
    population_offset_mean: float
    population_offset_std: float
    seed: int
    start_date: str
    metric_name: str
    metric_unit: str
    paths: dict[str, str]


def generate_calibration_data(
    rng: np.random.Generator | None = None,
    *,
    n_users: int = N_USERS,
    n_nights: int = N_NIGHTS,
    start_date: date = START_DATE,
    missing_rate: float = MISSING_RATE,
    seed: int | None = None,
    metric: MetricConfig = HR_METRIC,
) -> tuple[pd.DataFrame, pd.DataFrame, CalibrationManifest]:
    """Generate long-format readings + per-user ground-truth sidecar.

    Returns
    -------
    readings_df
        Columns: user_id, date, device_type, hr_reading, is_resting
    truth_df
        Columns: user_id, true_baseline, personal_offset
    manifest
        Generation summary statistics
    """
    if seed is None:
        seed = config.SEED
    if rng is None:
        rng = np.random.default_rng(seed)

    truth_rows: list[dict] = []
    reading_rows: list[dict] = []

    for u in range(n_users):
        user_id = f"user_{u + 1:02d}"
        true_baseline = float(rng.normal(metric.baseline_mean, metric.baseline_std))
        personal_offset = float(rng.normal(metric.offset_mean, metric.offset_std))

        truth_rows.append(
            {
                "user_id": user_id,
                "true_baseline": round(true_baseline, metric.decimal_places),
                "personal_offset": round(personal_offset, metric.decimal_places),
            }
        )

        for night in range(n_nights):
            day = start_date + timedelta(days=night)
            day_str = day.isoformat()
            true_night = true_baseline + float(rng.normal(0, metric.night_var_std))

            chest_val = true_night + float(rng.normal(0, metric.anchor_noise_std))
            wrist_val = true_night + personal_offset + float(rng.normal(0, metric.wrist_noise_std))

            if rng.random() >= missing_rate:
                reading_rows.append(
                    {
                        "user_id": user_id,
                        "date": day_str,
                        "device_type": DEVICE_CHEST,
                        "hr_reading": round(chest_val, metric.decimal_places),
                        "is_resting": True,
                    }
                )
            if rng.random() >= missing_rate:
                reading_rows.append(
                    {
                        "user_id": user_id,
                        "date": day_str,
                        "device_type": DEVICE_WRIST,
                        "hr_reading": round(wrist_val, metric.decimal_places),
                        "is_resting": True,
                    }
                )

    readings_df = pd.DataFrame(reading_rows)
    truth_df = pd.DataFrame(truth_rows)

    paired = _build_paired_nights(readings_df)
    pop_mean = float(paired["delta"].mean()) if len(paired) else 0.0
    pop_std = float(paired["delta"].std()) if len(paired) else 0.0

    n_expected = n_users * n_nights * 2
    manifest = CalibrationManifest(
        n_users=n_users,
        n_nights=n_nights,
        n_rows=len(readings_df),
        n_expected=n_expected,
        missing_rate=1.0 - len(readings_df) / n_expected,
        population_offset_mean=round(pop_mean, 3),
        population_offset_std=round(pop_std, 3),
        seed=seed,
        start_date=start_date.isoformat(),
        metric_name=metric.name,
        metric_unit=metric.unit,
        paths={},
    )
    return readings_df, truth_df, manifest


def _build_paired_nights(readings_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to paired chest/wrist per user/night and compute delta."""
    if readings_df.empty:
        return pd.DataFrame(columns=["user_id", "date", "chest", "wrist", "delta"])

    wide = readings_df.pivot_table(
        index=["user_id", "date"],
        columns="device_type",
        values="hr_reading",
        aggfunc="first",
    ).reset_index()

    if DEVICE_CHEST not in wide.columns or DEVICE_WRIST not in wide.columns:
        return pd.DataFrame(columns=["user_id", "date", "chest", "wrist", "delta"])

    wide = wide.dropna(subset=[DEVICE_CHEST, DEVICE_WRIST])
    wide = wide.rename(columns={DEVICE_CHEST: "chest", DEVICE_WRIST: "wrist"})
    wide["delta"] = wide["wrist"] - wide["chest"]
    return wide


def build_paired_nights(readings_df: pd.DataFrame) -> pd.DataFrame:
    """Public helper: paired nightly chest/wrist with delta column."""
    return _build_paired_nights(readings_df)


def write_calibration_dataset(
    out_dir: Path | None = None,
    *,
    seed: int | None = None,
    n_users: int = N_USERS,
    n_nights: int = N_NIGHTS,
    metric: MetricConfig = HR_METRIC,
) -> CalibrationManifest:
    """Generate and write CSV + truth sidecar + data dictionary."""
    root = Path(out_dir) if out_dir is not None else config.DATA_DIR
    root.mkdir(parents=True, exist_ok=True)

    readings_df, truth_df, manifest = generate_calibration_data(
        seed=seed, n_users=n_users, n_nights=n_nights, metric=metric
    )

    csv_path = root / "hr_calibration_dataset.csv"
    truth_path = root / "hr_calibration_truth.csv"
    dict_path = root / "hr_calibration_dataset.md"

    readings_df.to_csv(csv_path, index=False)
    truth_df.to_csv(truth_path, index=False)
    dict_path.write_text(_data_dictionary_text(manifest, metric))

    manifest.paths = {
        "readings": str(csv_path),
        "truth": str(truth_path),
        "dictionary": str(dict_path),
    }
    return manifest


def _data_dictionary_text(manifest: CalibrationManifest, metric: MetricConfig) -> str:
    return f"""# HR Calibration Dataset — Data Dictionary

Synthetic longitudinal resting {metric.name} readings for baseline training,
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
| `user_id` | string | Simulated user identifier (`user_01` … `user_{manifest.n_users:02d}`) |
| `date` | date (ISO) | Night of measurement, consecutive from `{manifest.start_date}` |
| `device_type` | string | `chest_strap` (trusted anchor) or `wrist` (device to calibrate) |
| `hr_reading` | float | Resting {metric.name} in {metric.unit} |
| `is_resting` | bool | Always `True` (resting-state dataset) |

Missing readings (~{manifest.missing_rate:.1%} per device) are omitted entirely.

## `hr_calibration_truth.csv`

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Matches readings file |
| `true_baseline` | float | User's stable baseline ({metric.unit}), drawn from N({metric.baseline_mean}, {metric.baseline_std}) |
| `personal_offset` | float | Wrist bias vs chest strap ({metric.unit}), drawn from N({metric.offset_mean}, {metric.offset_std}) |

## Generative model

Per user:
- `true_baseline ~ N({metric.baseline_mean}, {metric.baseline_std})`
- `personal_offset ~ N({metric.offset_mean}, {metric.offset_std})`

Per night:
- `true_night = true_baseline + N(0, {metric.night_var_std})`
- `chest_strap = true_night + N(0, {metric.anchor_noise_std})`
- `wrist = true_night + personal_offset + N(0, {metric.wrist_noise_std})`
- Each device independently missing with probability {MISSING_RATE:.0%}

## Dataset summary

| Stat | Value |
|------|-------|
| Metric | {metric.name} ({metric.unit}) |
| Users | {manifest.n_users} |
| Nights per user | {manifest.n_nights} |
| Total rows | {manifest.n_rows} |
| Expected rows (no missing) | {manifest.n_expected} |
| Observed missing rate | {manifest.missing_rate:.1%} |
| Observed population offset mean | {manifest.population_offset_mean} {metric.unit} |
| Observed population offset std | {manifest.population_offset_std} {metric.unit} |
| Random seed | {manifest.seed} |

## Usage

- **Population prior:** pool nightly `wrist - chest_strap` deltas across users.
- **Personal calibration:** feed per-user deltas into `OffsetState` in date order.
- **Backtest:** walk-forward expanding window; predict corrected wrist vs chest anchor.
"""
