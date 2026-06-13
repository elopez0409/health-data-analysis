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
# Default generative parameters
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
    paths: dict[str, str]


def generate_calibration_data(
    rng: np.random.Generator | None = None,
    *,
    n_users: int = N_USERS,
    n_nights: int = N_NIGHTS,
    start_date: date = START_DATE,
    missing_rate: float = MISSING_RATE,
    seed: int | None = None,
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
        true_baseline = float(rng.normal(TRUE_BASELINE_MEAN, TRUE_BASELINE_STD))
        personal_offset = float(rng.normal(PERSONAL_OFFSET_MEAN, PERSONAL_OFFSET_STD))

        truth_rows.append(
            {
                "user_id": user_id,
                "true_baseline": round(true_baseline, 2),
                "personal_offset": round(personal_offset, 2),
            }
        )

        for night in range(n_nights):
            day = start_date + timedelta(days=night)
            day_str = day.isoformat()
            true_hr_night = true_baseline + float(rng.normal(0, NIGHT_VAR_STD))

            chest_hr = true_hr_night + float(rng.normal(0, CHEST_NOISE_STD))
            wrist_hr = true_hr_night + personal_offset + float(rng.normal(0, WRIST_NOISE_STD))

            if rng.random() >= missing_rate:
                reading_rows.append(
                    {
                        "user_id": user_id,
                        "date": day_str,
                        "device_type": DEVICE_CHEST,
                        "hr_reading": round(chest_hr, 1),
                        "is_resting": True,
                    }
                )
            if rng.random() >= missing_rate:
                reading_rows.append(
                    {
                        "user_id": user_id,
                        "date": day_str,
                        "device_type": DEVICE_WRIST,
                        "hr_reading": round(wrist_hr, 1),
                        "is_resting": True,
                    }
                )

    readings_df = pd.DataFrame(reading_rows)
    truth_df = pd.DataFrame(truth_rows)

    # Observed population offset stats from paired nights
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
) -> CalibrationManifest:
    """Generate and write CSV + truth sidecar + data dictionary."""
    root = Path(out_dir) if out_dir is not None else config.DATA_DIR
    root.mkdir(parents=True, exist_ok=True)

    readings_df, truth_df, manifest = generate_calibration_data(
        seed=seed, n_users=n_users, n_nights=n_nights
    )

    csv_path = root / "hr_calibration_dataset.csv"
    truth_path = root / "hr_calibration_truth.csv"
    dict_path = root / "hr_calibration_dataset.md"

    readings_df.to_csv(csv_path, index=False)
    truth_df.to_csv(truth_path, index=False)
    dict_path.write_text(_data_dictionary_text(manifest))

    manifest.paths = {
        "readings": str(csv_path),
        "truth": str(truth_path),
        "dictionary": str(dict_path),
    }
    return manifest


def _data_dictionary_text(manifest: CalibrationManifest) -> str:
    return f"""# HR Calibration Dataset — Data Dictionary

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
| `user_id` | string | Simulated user identifier (`user_01` … `user_{manifest.n_users:02d}`) |
| `date` | date (ISO) | Night of measurement, consecutive from `{manifest.start_date}` |
| `device_type` | string | `chest_strap` (trusted anchor) or `wrist` (device to calibrate) |
| `hr_reading` | float | Resting heart rate in bpm |
| `is_resting` | bool | Always `True` (resting-state dataset) |

Missing readings (~{manifest.missing_rate:.1%} per device) are omitted entirely.

## `hr_calibration_truth.csv`

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Matches readings file |
| `true_baseline` | float | User's stable resting HR baseline (bpm), drawn from N({TRUE_BASELINE_MEAN}, {TRUE_BASELINE_STD}) |
| `personal_offset` | float | Wrist bias vs chest strap (bpm), drawn from N({PERSONAL_OFFSET_MEAN}, {PERSONAL_OFFSET_STD}) |

## Generative model

Per user:
- `true_baseline ~ N({TRUE_BASELINE_MEAN}, {TRUE_BASELINE_STD})`
- `personal_offset ~ N({PERSONAL_OFFSET_MEAN}, {PERSONAL_OFFSET_STD})`

Per night:
- `true_hr_night = true_baseline + N(0, {NIGHT_VAR_STD})`
- `chest_strap = true_hr_night + N(0, {CHEST_NOISE_STD})`
- `wrist = true_hr_night + personal_offset + N(0, {WRIST_NOISE_STD})`
- Each device independently missing with probability {MISSING_RATE:.0%}

## Dataset summary

| Stat | Value |
|------|-------|
| Users | {manifest.n_users} |
| Nights per user | {manifest.n_nights} |
| Total rows | {manifest.n_rows} |
| Expected rows (no missing) | {manifest.n_expected} |
| Observed missing rate | {manifest.missing_rate:.1%} |
| Observed population offset mean | {manifest.population_offset_mean} bpm |
| Observed population offset std | {manifest.population_offset_std} bpm |
| Random seed | {manifest.seed} |

## Usage

- **Population prior:** pool nightly `wrist - chest_strap` deltas across users.
- **Personal calibration:** feed per-user deltas into `OffsetState` in date order.
- **Backtest:** walk-forward expanding window; predict corrected wrist vs chest anchor.
"""
