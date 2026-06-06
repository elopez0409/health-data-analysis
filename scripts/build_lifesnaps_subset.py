"""
Build a ~500MB analysis subset from the LifeSnaps Fitbit Kaggle dataset.

Downloads via kagglehub, discovers available files, selects 30 users with the
best cross-metric coverage over a 90-day window, and writes normalized Parquet
tables to data/subset/.

Usage:
    python scripts/build_lifesnaps_subset.py
    python scripts/build_lifesnaps_subset.py --discover-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import polars as pl

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "subset"
DATASET_SLUG = "skywescar/lifesnaps-fitbit-dataset"

TARGET_USERS = 30
TARGET_DAYS = 90
MAX_SUBSET_MB = 500


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def download_dataset() -> Path:
    """Download the dataset and return the local path."""
    import kagglehub

    print(f"Downloading dataset: {DATASET_SLUG}")
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Dataset downloaded to: {path}")
    return Path(path)


def discover_files(dataset_path: Path) -> list[dict]:
    """Walk the dataset directory and report files with sizes."""
    files: list[dict] = []
    for p in sorted(dataset_path.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            files.append({"path": p, "name": p.name, "size_mb": size_mb})
    return files


def print_file_manifest(files: list[dict]) -> None:
    """Print a manifest of all discovered files."""
    total_mb = sum(f["size_mb"] for f in files)
    print(f"\n{'='*70}")
    print(f"Dataset file manifest ({len(files)} files, {total_mb:.1f} MB total)")
    print(f"{'='*70}")
    for f in files:
        print(f"  {f['size_mb']:>8.1f} MB  {f['path'].relative_to(f['path'].parents[2])}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# File selection heuristics
# ---------------------------------------------------------------------------

DAILY_CSV_PATTERNS = [
    "daily_fitbit_sema_df_unprocessed.csv",
    "daily_fitbit_sema_df.csv",
    "daily_fitbit",
]

RELEVANT_FILE_KEYWORDS = [
    "daily_activity", "dailyActivity",
    "daily_steps", "dailySteps",
    "daily_calories", "dailyCalories",
    "sleep", "Sleep",
    "heart_rate", "heartrate", "heartRate",
    "resting_heart_rate", "restingHeartRate",
    "active_minutes", "activeMinutes",
    "sedentary", "Sedentary",
    "minutely", "minuteSteps",
    "daily_fitbit",
]


def select_relevant_files(files: list[dict]) -> list[dict]:
    """Select files relevant to our analysis based on naming patterns."""
    selected = []
    for f in files:
        name_lower = f["name"].lower()
        # Always include the aggregated daily CSV
        if any(pat.lower() in name_lower for pat in DAILY_CSV_PATTERNS):
            f["role"] = "daily_aggregated"
            selected.append(f)
            continue
        # Include files matching our target keywords
        if any(kw.lower() in name_lower for kw in RELEVANT_FILE_KEYWORDS):
            f["role"] = "individual"
            selected.append(f)

    return selected


# ---------------------------------------------------------------------------
# Loading strategies depending on dataset structure
# ---------------------------------------------------------------------------


def load_daily_aggregated(csv_path: Path) -> pl.LazyFrame:
    """
    Load the aggregated daily CSV lazily. This file has one row per user per
    date with columns for each Fitbit metric type.
    """
    print(f"Scanning aggregated daily CSV: {csv_path.name} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    lf = pl.scan_csv(csv_path, infer_schema_length=5000, ignore_errors=True)
    return lf


def identify_columns(schema: dict[str, pl.DataType]) -> dict[str, list[str]]:
    """
    Categorize columns from the daily aggregated CSV into metric groups.
    Returns a mapping of group -> list of column names.
    """
    cols = list(schema.keys())
    groups: dict[str, list[str]] = {
        "id": [],
        "date": [],
        "steps": [],
        "calories": [],
        "active_minutes": [],
        "sedentary": [],
        "sleep": [],
        "heart_rate": [],
        "other": [],
    }

    for col in cols:
        cl = col.lower()
        if cl in ("id", "user_id", "userid", "participant_id"):
            groups["id"].append(col)
        elif cl in ("date", "day", "timestamp", "datetime"):
            groups["date"].append(col)
        elif "step" in cl and "goal" not in cl:
            groups["steps"].append(col)
        elif cl == "calories":
            groups["calories"].append(col)
        elif any(kw in cl for kw in ("lightly_active", "moderately_active",
                                      "fairly_active", "very_active")):
            groups["active_minutes"].append(col)
        elif "sedentary" in cl:
            groups["sedentary"].append(col)
        elif any(kw in cl for kw in ("sleep_duration", "minutesasleep",
                                      "sleep_efficiency", "minutestofallasleep",
                                      "minutesawake", "minutesafterwakeup",
                                      "sleep_deep", "sleep_light", "sleep_rem",
                                      "sleep_wake")):
            groups["sleep"].append(col)
        elif cl == "resting_hr" or "restingheartrate" in cl or cl == "bpm":
            groups["heart_rate"].append(col)
        else:
            groups["other"].append(col)

    return groups


# ---------------------------------------------------------------------------
# Individual file loading (if dataset has per-type CSVs)
# ---------------------------------------------------------------------------


def find_file_by_keywords(files: list[dict], keywords: list[str]) -> Path | None:
    """Find the first file matching any of the keywords."""
    for f in files:
        name_lower = f["name"].lower()
        if any(kw.lower() in name_lower for kw in keywords):
            return f["path"]
    return None


def load_individual_files(selected_files: list[dict]) -> dict[str, pl.LazyFrame]:
    """Load individual per-type CSV files lazily."""
    frames: dict[str, pl.LazyFrame] = {}

    activity_path = find_file_by_keywords(selected_files, ["dailyActivity", "daily_activity"])
    if activity_path:
        frames["activity"] = pl.scan_csv(activity_path, infer_schema_length=5000, ignore_errors=True)

    sleep_path = find_file_by_keywords(selected_files, ["sleep", "Sleep"])
    if sleep_path:
        frames["sleep"] = pl.scan_csv(sleep_path, infer_schema_length=5000, ignore_errors=True)

    hr_path = find_file_by_keywords(selected_files, ["resting_heart_rate", "restingHeartRate", "daily_heart_rate"])
    if hr_path:
        frames["heart_rate"] = pl.scan_csv(hr_path, infer_schema_length=5000, ignore_errors=True)

    steps_path = find_file_by_keywords(selected_files, ["dailySteps", "daily_steps"])
    if steps_path and "activity" not in frames:
        frames["steps"] = pl.scan_csv(steps_path, infer_schema_length=5000, ignore_errors=True)

    calories_path = find_file_by_keywords(selected_files, ["dailyCalories", "daily_calories"])
    if calories_path and "activity" not in frames:
        frames["calories"] = pl.scan_csv(calories_path, infer_schema_length=5000, ignore_errors=True)

    return frames


# ---------------------------------------------------------------------------
# User selection: pick users with best cross-metric coverage
# ---------------------------------------------------------------------------


def score_users_aggregated(lf: pl.LazyFrame, id_col: str, date_col: str, metric_cols: list[str]) -> pl.DataFrame:
    """
    Score each user by how many dates have non-null values across key metric
    columns. Returns a DataFrame with user_id and coverage_score.
    """
    coverage_expr = [
        pl.col(c).is_not_null().cast(pl.Int32).alias(f"_has_{c}")
        for c in metric_cols
    ]
    scored = (
        lf.select([pl.col(id_col), pl.col(date_col)] + coverage_expr)
        .group_by(id_col)
        .agg([
            pl.len().alias("total_days"),
            *[pl.col(f"_has_{c}").sum().alias(f"coverage_{c}") for c in metric_cols],
        ])
        .with_columns(
            sum([pl.col(f"coverage_{c}") for c in metric_cols]).alias("coverage_score")
        )
        .sort("coverage_score", descending=True)
        .collect()
    )
    return scored


def score_users_individual(frames: dict[str, pl.LazyFrame]) -> pl.DataFrame:
    """Score users across individual files by counting distinct dates per metric."""
    user_scores: dict[str, int] = {}

    for name, lf in frames.items():
        schema = lf.collect_schema()
        cols = list(schema.names())
        id_col = _find_id_col(cols)
        if not id_col:
            continue
        users = lf.select(pl.col(id_col)).unique().collect()
        for uid in users[id_col].to_list():
            user_scores[uid] = user_scores.get(uid, 0) + 1

    return pl.DataFrame({
        "user_id": list(user_scores.keys()),
        "coverage_score": list(user_scores.values()),
    }).sort("coverage_score", descending=True)


def _find_id_col(cols: list[str]) -> str | None:
    """Heuristic to find the user ID column."""
    for candidate in ["Id", "id", "user_id", "userId", "UserId", "participant_id", "ID"]:
        if candidate in cols:
            return candidate
    for c in cols:
        if "id" in c.lower() and "record" not in c.lower():
            return c
    return None


def _find_date_col(cols: list[str]) -> str | None:
    """Heuristic to find the date column."""
    for candidate in ["date", "Date", "ActivityDate", "activity_date", "SleepDay", "day", "timestamp"]:
        if candidate in cols:
            return candidate
    for c in cols:
        if "date" in c.lower() or "day" in c.lower():
            return c
    return None


# ---------------------------------------------------------------------------
# Best date window selection
# ---------------------------------------------------------------------------


def select_best_window(lf: pl.LazyFrame, id_col: str, date_col: str,
                       user_ids: list, n_days: int) -> tuple[date, date]:
    """
    Given a filtered LazyFrame of selected users, find the n_days window
    with the most total non-null records.
    """
    dates_df = (
        lf.filter(pl.col(id_col).is_in(user_ids))
        .select(pl.col(date_col))
        .unique()
        .sort(date_col)
        .collect()
    )

    if dates_df.is_empty():
        raise ValueError("No dates found for selected users")

    all_dates = dates_df[date_col].to_list()

    # Convert to date objects if strings
    if isinstance(all_dates[0], str):
        all_dates = [date.fromisoformat(d) if isinstance(d, str) else d for d in all_dates]

    if len(all_dates) <= n_days:
        return all_dates[0], all_dates[-1]

    # Sliding window: pick the window with the most dates that have data
    date_counts = (
        lf.filter(pl.col(id_col).is_in(user_ids))
        .group_by(date_col)
        .agg(pl.len().alias("n_records"))
        .sort(date_col)
        .collect()
    )

    date_vals = date_counts[date_col].to_list()
    record_counts = date_counts["n_records"].to_list()

    if isinstance(date_vals[0], str):
        date_vals = [date.fromisoformat(d) for d in date_vals]

    best_score = 0
    best_start_idx = 0
    for i in range(len(date_vals) - n_days + 1):
        window_score = sum(record_counts[i:i + n_days])
        if window_score > best_score:
            best_score = window_score
            best_start_idx = i

    return date_vals[best_start_idx], date_vals[min(best_start_idx + n_days - 1, len(date_vals) - 1)]


# ---------------------------------------------------------------------------
# Normalization into unified tables
# ---------------------------------------------------------------------------


def normalize_daily_metrics_aggregated(
    lf: pl.LazyFrame,
    id_col: str,
    date_col: str,
    groups: dict[str, list[str]],
    user_ids: list,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Normalize aggregated daily CSV into unified_daily_metrics schema."""
    select_cols = [id_col, date_col]
    select_cols += groups.get("steps", [])
    select_cols += groups.get("calories", [])
    select_cols += groups.get("active_minutes", [])
    select_cols += groups.get("sedentary", [])
    # Deduplicate
    select_cols = list(dict.fromkeys(select_cols))

    df = (
        lf.select(select_cols)
        .filter(pl.col(id_col).is_in(user_ids))
        .filter(pl.col(date_col) >= str(start_date))
        .filter(pl.col(date_col) <= str(end_date))
        .collect()
    )

    rename_map = {id_col: "user_id", date_col: "date"}

    step_col = groups["steps"][0] if groups["steps"] else None
    cal_col = groups["calories"][0] if groups["calories"] else None
    sed_col = groups["sedentary"][0] if groups["sedentary"] else None

    # Find active minute sub-columns
    active_cols = groups.get("active_minutes", [])
    lightly_col = next((c for c in active_cols if "light" in c.lower()), None)
    fairly_col = next((c for c in active_cols if "fair" in c.lower() or "moderate" in c.lower()), None)
    very_col = next((c for c in active_cols if "very" in c.lower()), None)

    if step_col:
        rename_map[step_col] = "steps"
    if cal_col:
        rename_map[cal_col] = "calories"
    if sed_col:
        rename_map[sed_col] = "sedentary_minutes"
    if lightly_col:
        rename_map[lightly_col] = "lightly_active_minutes"
    if fairly_col:
        rename_map[fairly_col] = "fairly_active_minutes"
    if very_col:
        rename_map[very_col] = "very_active_minutes"

    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure expected columns exist
    for col in ["steps", "calories", "sedentary_minutes",
                "lightly_active_minutes", "fairly_active_minutes", "very_active_minutes"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Compute active_minutes as sum
    df = df.with_columns(
        (
            pl.col("lightly_active_minutes").fill_null(0)
            + pl.col("fairly_active_minutes").fill_null(0)
            + pl.col("very_active_minutes").fill_null(0)
        ).alias("active_minutes")
    )

    df = df.with_columns([
        pl.lit("fitbit").alias("source"),
        pl.lit(0.9).alias("confidence"),
    ])

    final_cols = ["user_id", "date", "steps", "calories", "sedentary_minutes",
                  "lightly_active_minutes", "fairly_active_minutes",
                  "very_active_minutes", "active_minutes", "source", "confidence"]
    return df.select([c for c in final_cols if c in df.columns])


def normalize_daily_metrics_individual(
    frames: dict[str, pl.LazyFrame],
    user_ids: list,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Normalize individual per-type files into unified_daily_metrics."""
    activity_lf = frames.get("activity")
    if activity_lf is None:
        steps_lf = frames.get("steps")
        calories_lf = frames.get("calories")
        if steps_lf is None and calories_lf is None:
            return pl.DataFrame()
        # Merge steps + calories
        parts = []
        for name, lf in [("steps", steps_lf), ("calories", calories_lf)]:
            if lf is None:
                continue
            schema = lf.collect_schema()
            cols = list(schema.names())
            id_col = _find_id_col(cols)
            date_col = _find_date_col(cols)
            if id_col and date_col:
                part = (
                    lf.filter(pl.col(id_col).is_in(user_ids))
                    .filter(pl.col(date_col) >= str(start_date))
                    .filter(pl.col(date_col) <= str(end_date))
                    .collect()
                )
                parts.append((name, part, id_col, date_col))
        if not parts:
            return pl.DataFrame()
        # Use first available as base
        _, base_df, base_id, base_date = parts[0]
        base_df = base_df.rename({base_id: "user_id", base_date: "date"})
        return base_df.with_columns([
            pl.lit("fitbit").alias("source"),
            pl.lit(0.9).alias("confidence"),
        ])

    # Use the activity file
    schema = activity_lf.collect_schema()
    cols = list(schema.names())
    id_col = _find_id_col(cols)
    date_col = _find_date_col(cols)
    if not id_col or not date_col:
        print(f"WARNING: Could not find id/date columns in activity file. Columns: {cols[:20]}")
        return pl.DataFrame()

    df = (
        activity_lf.filter(pl.col(id_col).is_in(user_ids))
        .filter(pl.col(date_col) >= str(start_date))
        .filter(pl.col(date_col) <= str(end_date))
        .collect()
    )

    rename_map = {id_col: "user_id", date_col: "date"}
    # Map known Fitbit daily activity columns
    col_mapping = {
        "totalsteps": "steps", "steps": "steps",
        "calories": "calories", "totalcalories": "calories",
        "sedentaryminutes": "sedentary_minutes", "sedentary_minutes": "sedentary_minutes",
        "lightlyactiveminutes": "lightly_active_minutes", "lightly_active_minutes": "lightly_active_minutes",
        "fairlyactiveminutes": "fairly_active_minutes", "fairly_active_minutes": "fairly_active_minutes",
        "veryactiveminutes": "very_active_minutes", "very_active_minutes": "very_active_minutes",
    }
    for col in df.columns:
        mapped = col_mapping.get(col.lower())
        if mapped and col != mapped:
            rename_map[col] = mapped

    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["steps", "calories", "sedentary_minutes",
                "lightly_active_minutes", "fairly_active_minutes", "very_active_minutes"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    df = df.with_columns(
        (
            pl.col("lightly_active_minutes").fill_null(0)
            + pl.col("fairly_active_minutes").fill_null(0)
            + pl.col("very_active_minutes").fill_null(0)
        ).alias("active_minutes")
    )

    df = df.with_columns([
        pl.lit("fitbit").alias("source"),
        pl.lit(0.9).alias("confidence"),
    ])

    final_cols = ["user_id", "date", "steps", "calories", "sedentary_minutes",
                  "lightly_active_minutes", "fairly_active_minutes",
                  "very_active_minutes", "active_minutes", "source", "confidence"]
    return df.select([c for c in final_cols if c in df.columns])


def normalize_sleep_aggregated(
    lf: pl.LazyFrame,
    id_col: str,
    date_col: str,
    groups: dict[str, list[str]],
    user_ids: list,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Normalize aggregated daily CSV sleep columns into unified_sleep."""
    sleep_cols = groups.get("sleep", [])
    if not sleep_cols:
        return pl.DataFrame()

    select_cols = list(dict.fromkeys([id_col, date_col] + sleep_cols))
    df = (
        lf.select(select_cols)
        .filter(pl.col(id_col).is_in(user_ids))
        .filter(pl.col(date_col) >= str(start_date))
        .filter(pl.col(date_col) <= str(end_date))
        .collect()
    )

    rename_map = {id_col: "user_id", date_col: "sleep_date"}

    # Try to map sleep columns — avoid duplicates by tracking which target names are taken
    mapped_targets: set[str] = set()
    # Prefer minutesAsleep over sleep_duration for duration
    duration_candidates = [c for c in sleep_cols if c.lower() == "minutesasleep"]
    if not duration_candidates:
        duration_candidates = [c for c in sleep_cols if "duration" in c.lower()]
    if duration_candidates:
        rename_map[duration_candidates[0]] = "duration_minutes"
        mapped_targets.add("duration_minutes")

    for col in sleep_cols:
        cl = col.lower()
        if col in rename_map:
            continue
        if "efficiency" in cl and "efficiency" not in mapped_targets:
            rename_map[col] = "efficiency"
            mapped_targets.add("efficiency")
        elif "start" in cl and "time" in cl and "start_time" not in mapped_targets:
            rename_map[col] = "start_time"
            mapped_targets.add("start_time")
        elif "end" in cl and "time" in cl and "end_time" not in mapped_targets:
            rename_map[col] = "end_time"
            mapped_targets.add("end_time")

    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    # If duration came from sleep_duration (which may be in ms), convert to minutes
    if "duration_minutes" in df.columns:
        sample = df["duration_minutes"].drop_nulls().head(5).to_list()
        if sample and any(v > 1440 for v in sample if v is not None):
            df = df.with_columns(
                (pl.col("duration_minutes") / 60000.0).alias("duration_minutes")
            )

    for col in ["duration_minutes", "efficiency", "start_time", "end_time"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    df = df.with_columns([
        pl.lit("fitbit").alias("source"),
        pl.lit(1.0).alias("confidence"),
    ])

    final_cols = ["user_id", "sleep_date", "duration_minutes", "efficiency",
                  "start_time", "end_time", "source", "confidence"]
    return df.select([c for c in final_cols if c in df.columns])


def normalize_sleep_individual(
    frames: dict[str, pl.LazyFrame],
    user_ids: list,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Normalize individual sleep file into unified_sleep."""
    sleep_lf = frames.get("sleep")
    if sleep_lf is None:
        return pl.DataFrame()

    schema = sleep_lf.collect_schema()
    cols = list(schema.names())
    id_col = _find_id_col(cols)
    date_col = _find_date_col(cols)
    if not id_col or not date_col:
        print(f"WARNING: Could not find id/date columns in sleep file. Columns: {cols[:20]}")
        return pl.DataFrame()

    df = (
        sleep_lf.filter(pl.col(id_col).is_in(user_ids))
        .filter(pl.col(date_col) >= str(start_date))
        .filter(pl.col(date_col) <= str(end_date))
        .collect()
    )

    rename_map = {id_col: "user_id", date_col: "sleep_date"}
    col_mapping = {
        "totalminutesasleep": "duration_minutes",
        "minutesasleep": "duration_minutes",
        "total_minutes_asleep": "duration_minutes",
        "duration": "duration_minutes",
        "efficiency": "efficiency",
        "sleepefficiency": "efficiency",
        "sleep_efficiency": "efficiency",
        "starttime": "start_time",
        "start_time": "start_time",
        "endtime": "end_time",
        "end_time": "end_time",
    }
    for col in df.columns:
        mapped = col_mapping.get(col.lower())
        if mapped and col != mapped:
            rename_map[col] = mapped

    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["duration_minutes", "efficiency", "start_time", "end_time"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    df = df.with_columns([
        pl.lit("fitbit").alias("source"),
        pl.lit(1.0).alias("confidence"),
    ])

    final_cols = ["user_id", "sleep_date", "duration_minutes", "efficiency",
                  "start_time", "end_time", "source", "confidence"]
    return df.select([c for c in final_cols if c in df.columns])


def normalize_heart_rate_aggregated(
    lf: pl.LazyFrame,
    id_col: str,
    date_col: str,
    groups: dict[str, list[str]],
    user_ids: list,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Normalize aggregated daily CSV heart rate columns into unified_heart_rate."""
    hr_cols = groups.get("heart_rate", [])
    if not hr_cols:
        return pl.DataFrame()

    select_cols = list(dict.fromkeys([id_col, date_col] + hr_cols))
    df = (
        lf.select(select_cols)
        .filter(pl.col(id_col).is_in(user_ids))
        .filter(pl.col(date_col) >= str(start_date))
        .filter(pl.col(date_col) <= str(end_date))
        .collect()
    )

    rename_map = {id_col: "user_id", date_col: "date"}
    for col in hr_cols:
        cl = col.lower()
        if "resting" in cl:
            rename_map[col] = "resting_hr"
        elif cl == "bpm":
            rename_map[col] = "avg_hr"
        elif "avg" in cl or "mean" in cl or "average" in cl:
            rename_map[col] = "avg_hr"

    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    if "resting_hr" not in df.columns and "avg_hr" not in df.columns:
        remaining_hr = [c for c in df.columns if c not in ("user_id", "date")]
        if remaining_hr:
            df = df.rename({remaining_hr[0]: "resting_hr"})

    for col in ["resting_hr", "avg_hr"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    df = df.with_columns([
        pl.lit("fitbit").alias("source"),
        pl.lit(1.0).alias("confidence"),
    ])

    final_cols = ["user_id", "date", "resting_hr", "avg_hr", "source", "confidence"]
    return df.select([c for c in final_cols if c in df.columns])


def normalize_heart_rate_individual(
    frames: dict[str, pl.LazyFrame],
    user_ids: list,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Normalize individual heart rate file into unified_heart_rate."""
    hr_lf = frames.get("heart_rate")
    if hr_lf is None:
        return pl.DataFrame()

    schema = hr_lf.collect_schema()
    cols = list(schema.names())
    id_col = _find_id_col(cols)
    date_col = _find_date_col(cols)
    if not id_col or not date_col:
        print(f"WARNING: Could not find id/date columns in HR file. Columns: {cols[:20]}")
        return pl.DataFrame()

    df = (
        hr_lf.filter(pl.col(id_col).is_in(user_ids))
        .filter(pl.col(date_col) >= str(start_date))
        .filter(pl.col(date_col) <= str(end_date))
        .collect()
    )

    rename_map = {id_col: "user_id", date_col: "date"}
    col_mapping = {
        "restingheartrate": "resting_hr",
        "resting_heart_rate": "resting_hr",
        "value": "resting_hr",
        "avg_hr": "avg_hr",
        "averageheartrate": "avg_hr",
    }
    for col in df.columns:
        mapped = col_mapping.get(col.lower())
        if mapped and col != mapped:
            rename_map[col] = mapped

    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["resting_hr", "avg_hr"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    df = df.with_columns([
        pl.lit("fitbit").alias("source"),
        pl.lit(1.0).alias("confidence"),
    ])

    final_cols = ["user_id", "date", "resting_hr", "avg_hr", "source", "confidence"]
    return df.select([c for c in final_cols if c in df.columns])


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------


def write_manifest(
    output_dir: Path,
    user_ids: list,
    start_date: date,
    end_date: date,
    tables: dict[str, pl.DataFrame],
) -> None:
    """Write a JSON manifest documenting the subset."""
    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "dataset": DATASET_SLUG,
        "n_users": len(user_ids),
        "user_ids": [str(u) for u in user_ids],
        "date_range": {"start": str(start_date), "end": str(end_date)},
        "tables": {},
    }

    total_bytes = 0
    for name, df in tables.items():
        path = output_dir / f"{name}.parquet"
        size_bytes = path.stat().st_size if path.exists() else 0
        total_bytes += size_bytes
        manifest["tables"][name] = {
            "rows": len(df),
            "columns": df.columns,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
        }

    manifest["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nManifest written to {manifest_path}")
    print(f"Total subset size: {manifest['total_size_mb']:.1f} MB")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LifeSnaps analysis subset")
    parser.add_argument("--discover-only", action="store_true",
                        help="Only download and print file manifest, don't process")
    parser.add_argument("--users", type=int, default=TARGET_USERS,
                        help=f"Number of users to select (default: {TARGET_USERS})")
    parser.add_argument("--days", type=int, default=TARGET_DAYS,
                        help=f"Number of days in the window (default: {TARGET_DAYS})")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Use an already-downloaded dataset path instead of downloading")
    args = parser.parse_args()

    # Step 1: Download
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = download_dataset()

    # Step 2: Discover files
    files = discover_files(dataset_path)
    print_file_manifest(files)

    if args.discover_only:
        print("Discovery complete. Use --dataset-path to skip re-downloading.")
        print(f"  --dataset-path {dataset_path}")
        return

    # Step 3: Determine loading strategy
    selected = select_relevant_files(files)
    if not selected:
        print("ERROR: No relevant files found in dataset. Available files:")
        for f in files:
            print(f"  {f['name']}")
        sys.exit(1)

    total_selected_mb = sum(f["size_mb"] for f in selected)
    print(f"\nSelected {len(selected)} relevant files ({total_selected_mb:.1f} MB):")
    for f in selected:
        print(f"  [{f.get('role', '?')}] {f['name']} ({f['size_mb']:.1f} MB)")

    # Decide strategy: aggregated CSV or individual files
    agg_file = next((f for f in selected if f.get("role") == "daily_aggregated"), None)
    use_aggregated = agg_file is not None

    if use_aggregated:
        print(f"\nUsing aggregated daily CSV strategy: {agg_file['name']}")
        lf = load_daily_aggregated(agg_file["path"])
        schema = lf.collect_schema()
        cols = list(schema.names())
        print(f"  Columns ({len(cols)}): {cols[:30]}{'...' if len(cols) > 30 else ''}")

        id_col = _find_id_col(cols)
        date_col = _find_date_col(cols)
        if not id_col or not date_col:
            print(f"ERROR: Cannot identify id column ({id_col}) or date column ({date_col})")
            print(f"  All columns: {cols}")
            sys.exit(1)

        print(f"  ID column: {id_col}, Date column: {date_col}")

        groups = identify_columns(schema)
        print(f"\n  Column groups:")
        for group, group_cols in groups.items():
            if group_cols and group != "other":
                print(f"    {group}: {group_cols}")

        # Pick metric columns to score coverage
        metric_cols = []
        for g in ["steps", "calories", "sleep", "heart_rate"]:
            if groups[g]:
                metric_cols.append(groups[g][0])

        if not metric_cols:
            print("ERROR: Could not identify any metric columns for scoring")
            sys.exit(1)

        # Score users
        print(f"\n  Scoring users by coverage across: {metric_cols}")
        user_scores = score_users_aggregated(lf, id_col, date_col, metric_cols)
        selected_users = user_scores[id_col].head(args.users).to_list()
        print(f"  Selected {len(selected_users)} users")

        # Select date window
        print(f"\n  Selecting best {args.days}-day window...")
        start_date, end_date = select_best_window(lf, id_col, date_col, selected_users, args.days)
        print(f"  Window: {start_date} to {end_date}")

        # Normalize
        print("\n  Normalizing daily metrics...")
        daily_df = normalize_daily_metrics_aggregated(lf, id_col, date_col, groups, selected_users, start_date, end_date)
        print(f"    -> {len(daily_df)} rows")

        print("  Normalizing sleep...")
        sleep_df = normalize_sleep_aggregated(lf, id_col, date_col, groups, selected_users, start_date, end_date)
        print(f"    -> {len(sleep_df)} rows")

        print("  Normalizing heart rate...")
        hr_df = normalize_heart_rate_aggregated(lf, id_col, date_col, groups, selected_users, start_date, end_date)
        print(f"    -> {len(hr_df)} rows")

    else:
        print("\nUsing individual file strategy")
        frames = load_individual_files(selected)
        if not frames:
            print("ERROR: Could not load any relevant files")
            sys.exit(1)

        print(f"  Loaded {len(frames)} file types: {list(frames.keys())}")

        # Score users
        user_scores = score_users_individual(frames)
        selected_users = user_scores["user_id"].head(args.users).to_list()
        print(f"  Selected {len(selected_users)} users")

        # Find date range from first available frame
        first_frame_name = list(frames.keys())[0]
        first_lf = frames[first_frame_name]
        first_cols = list(first_lf.collect_schema().names())
        first_id_col = _find_id_col(first_cols)
        first_date_col = _find_date_col(first_cols)

        if first_id_col and first_date_col:
            start_date, end_date = select_best_window(
                first_lf, first_id_col, first_date_col, selected_users, args.days
            )
        else:
            # Fallback: use a broad range
            start_date = date(2022, 1, 1)
            end_date = date(2022, 12, 31)

        print(f"  Window: {start_date} to {end_date}")

        # Normalize
        print("\n  Normalizing daily metrics...")
        daily_df = normalize_daily_metrics_individual(frames, selected_users, start_date, end_date)
        print(f"    -> {len(daily_df)} rows")

        print("  Normalizing sleep...")
        sleep_df = normalize_sleep_individual(frames, selected_users, start_date, end_date)
        print(f"    -> {len(sleep_df)} rows")

        print("  Normalizing heart rate...")
        hr_df = normalize_heart_rate_individual(frames, selected_users, start_date, end_date)
        print(f"    -> {len(hr_df)} rows")

    # Step 4: Write Parquet files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {}
    if not daily_df.is_empty():
        path = OUTPUT_DIR / "unified_daily_metrics.parquet"
        daily_df.write_parquet(path)
        tables["unified_daily_metrics"] = daily_df
        print(f"\n  Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    if not sleep_df.is_empty():
        path = OUTPUT_DIR / "unified_sleep.parquet"
        sleep_df.write_parquet(path)
        tables["unified_sleep"] = sleep_df
        print(f"  Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    if not hr_df.is_empty():
        path = OUTPUT_DIR / "unified_heart_rate.parquet"
        hr_df.write_parquet(path)
        tables["unified_heart_rate"] = hr_df
        print(f"  Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    # Step 5: Check size constraint
    total_size_mb = sum(
        (OUTPUT_DIR / f"{name}.parquet").stat().st_size / (1024 * 1024)
        for name in tables
    )
    if total_size_mb > MAX_SUBSET_MB:
        print(f"\nWARNING: Total subset size ({total_size_mb:.1f} MB) exceeds {MAX_SUBSET_MB} MB target.")
        print("Consider reducing --users or --days.")
    else:
        print(f"\nSubset size: {total_size_mb:.1f} MB (within {MAX_SUBSET_MB} MB target)")

    # Step 6: Write manifest
    write_manifest(OUTPUT_DIR, selected_users, start_date, end_date, tables)

    print("\nDone! Parquet files are ready for analysis in notebooks/04_lifesnaps_analysis.ipynb")


if __name__ == "__main__":
    main()
