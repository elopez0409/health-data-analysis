"""
Generate synthetic multi-device wearable data for 5 users over 14 days.

Uses realistic distributions derived from the LifeSnaps Fitbit dataset and
published norms for WHOOP, Oura, Garmin, and Strava. Produces raw JSON payloads
matching each provider's API schema, then normalizes into unified Parquet tables.

Each user has a "ground truth" physiological state; per-device noise simulates
real-world sensor disagreement for demonstrating conflict resolution.

Usage:
    python scripts/seed_synthetic.py
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
NORM_DIR = ROOT / "data" / "normalized"

SEED = 42
RNG = np.random.default_rng(SEED)

N_DAYS = 14
START_DATE = date(2024, 3, 1)

# ---------------------------------------------------------------------------
# User definitions
# ---------------------------------------------------------------------------

USERS = [
    {
        "id": "a1b2c3d4-1111-4000-8000-000000000001",
        "name": "user_01",
        "primary": "fitbit",
        "secondary": ["oura"],
        "description": "Fitbit + Oura overlap (sleep + HR on both)",
    },
    {
        "id": "a1b2c3d4-2222-4000-8000-000000000002",
        "name": "user_02",
        "primary": "whoop",
        "secondary": ["garmin"],
        "description": "WHOOP + Garmin (recovery vs body-battery)",
    },
    {
        "id": "a1b2c3d4-3333-4000-8000-000000000003",
        "name": "user_03",
        "primary": "oura",
        "secondary": ["strava", "fitbit"],
        "description": "Oura daily + Strava workouts + Fitbit steps",
    },
    {
        "id": "a1b2c3d4-4444-4000-8000-000000000004",
        "name": "user_04",
        "primary": "garmin",
        "secondary": ["whoop"],
        "description": "Garmin GPS activities + WHOOP strain/recovery",
    },
    {
        "id": "a1b2c3d4-5555-4000-8000-000000000005",
        "name": "user_05",
        "primary": "fitbit",
        "secondary": ["oura", "whoop", "garmin", "strava"],
        "description": "Power user — full overlap for conflict resolution",
    },
]

# ---------------------------------------------------------------------------
# Distribution parameters (from LifeSnaps + published literature)
# ---------------------------------------------------------------------------

DISTRIBUTIONS = {
    "steps": {"mean": 7877, "std": 5437, "min": 0, "max": 43000},
    "calories": {"mean": 2165, "std": 695, "min": 800, "max": 5300},
    "active_minutes": {"mean": 200, "std": 135, "min": 0, "max": 660},
    "sedentary_minutes": {"mean": 700, "std": 200, "min": 200, "max": 1200},
    "sleep_duration_min": {"mean": 399, "std": 94, "min": 180, "max": 660},
    "sleep_efficiency": {"mean": 94, "std": 3.6, "min": 75, "max": 100},
    "resting_hr": {"mean": 68, "std": 5.7, "min": 55, "max": 85},
    "avg_hr": {"mean": 80, "std": 7.8, "min": 55, "max": 130},
    "hrv_rmssd": {"mean": 45, "std": 20, "min": 15, "max": 120},
    "recovery_score": {"mean": 55, "std": 20, "min": 1, "max": 100},
    "strain": {"mean": 10, "std": 4, "min": 0, "max": 21},
    "body_battery": {"mean": 70, "std": 15, "min": 5, "max": 100},
    "stress_level": {"mean": 35, "std": 15, "min": 1, "max": 100},
    "readiness_score": {"mean": 75, "std": 12, "min": 40, "max": 98},
}


def sample(key: str) -> float:
    d = DISTRIBUTIONS[key]
    val = RNG.normal(d["mean"], d["std"])
    return float(np.clip(val, d["min"], d["max"]))


# ---------------------------------------------------------------------------
# Ground truth generation
# ---------------------------------------------------------------------------


def generate_ground_truth(user: dict) -> list[dict]:
    """Generate a base physiological state for each day."""
    days = []
    # Per-user baseline offsets (some users are fitter, sleep more, etc.)
    user_offset = {
        "resting_hr": RNG.normal(0, 3),
        "steps": RNG.normal(0, 2000),
        "sleep_duration": RNG.normal(0, 30),
        "hrv": RNG.normal(0, 10),
    }

    for i in range(N_DAYS):
        d = START_DATE + timedelta(days=i)
        steps = max(0, int(sample("steps") + user_offset["steps"]))
        rhr = sample("resting_hr") + user_offset["resting_hr"]
        rhr = float(np.clip(rhr, 50, 90))
        sleep_min = sample("sleep_duration_min") + user_offset["sleep_duration"]
        sleep_min = float(np.clip(sleep_min, 120, 720))
        hrv = sample("hrv_rmssd") + user_offset["hrv"]
        hrv = float(np.clip(hrv, 10, 150))

        # Workouts: ~40% chance of a workout on any given day
        has_workout = RNG.random() < 0.4
        workout_type = RNG.choice(["Run", "Ride", "Walk", "Swim", "Yoga"])
        workout_duration_sec = int(RNG.normal(2700, 900)) if has_workout else 0
        workout_distance_m = workout_duration_sec * RNG.uniform(1.5, 4.0) if has_workout else 0

        days.append({
            "date": d,
            "steps": steps,
            "calories": int(sample("calories")),
            "active_minutes": int(sample("active_minutes")),
            "sedentary_minutes": int(sample("sedentary_minutes")),
            "sleep_duration_min": round(sleep_min),
            "sleep_efficiency": round(sample("sleep_efficiency"), 1),
            "resting_hr": round(rhr, 1),
            "avg_hr": round(rhr + RNG.normal(12, 3), 1),
            "hrv_rmssd": round(hrv, 1),
            "recovery_score": round(sample("recovery_score")),
            "strain": round(sample("strain"), 1),
            "body_battery": round(sample("body_battery")),
            "stress_level": round(sample("stress_level")),
            "readiness_score": round(sample("readiness_score")),
            "has_workout": has_workout,
            "workout_type": str(workout_type),
            "workout_duration_sec": workout_duration_sec,
            "workout_distance_m": round(workout_distance_m, 1),
            "workout_avg_hr": round(rhr + RNG.normal(40, 10), 1) if has_workout else None,
            "workout_max_hr": round(rhr + RNG.normal(60, 12), 1) if has_workout else None,
            "workout_calories": round(workout_duration_sec * RNG.uniform(0.12, 0.22)) if has_workout else None,
        })
    return days


# ---------------------------------------------------------------------------
# Missing data patterns
# ---------------------------------------------------------------------------

MISSING_PATTERNS = {
    "a1b2c3d4-1111-4000-8000-000000000001": {"oura": [2, 5, 11]},  # user_01 forgets Oura 3 days
    "a1b2c3d4-2222-4000-8000-000000000002": {"whoop": [4, 7]},     # user_02 WHOOP uncharged 2 days
    "a1b2c3d4-3333-4000-8000-000000000003": {"fitbit": [9, 10]},   # user_03 Fitbit left at home
    "a1b2c3d4-4444-4000-8000-000000000004": {"garmin": [6]},       # user_04 Garmin charging
    "a1b2c3d4-5555-4000-8000-000000000005": {"oura": [3], "strava": [1, 8, 12]},  # user_05 sporadic
}


def is_missing(user_id: str, provider: str, day_idx: int) -> bool:
    patterns = MISSING_PATTERNS.get(user_id, {})
    return day_idx in patterns.get(provider, [])


# ---------------------------------------------------------------------------
# Raw JSON generators (match existing Pydantic schemas)
# ---------------------------------------------------------------------------


def gen_fitbit_daily(truth: dict, user_id: str) -> dict:
    """Generate Fitbit activity summary + sleep log JSON."""
    d = truth["date"]
    bedtime = datetime(d.year, d.month, d.day, 23, 0) - timedelta(minutes=int(RNG.normal(30, 15)))
    wake = bedtime + timedelta(minutes=truth["sleep_duration_min"])

    return {
        "activity_summary": {
            "date": d.isoformat(),
            "summary": {
                "steps": truth["steps"],
                "caloriesOut": truth["calories"],
                "activityCalories": int(truth["calories"] * 0.4),
                "sedentaryMinutes": truth["sedentary_minutes"],
                "lightlyActiveMinutes": int(truth["active_minutes"] * 0.5),
                "fairlyActiveMinutes": int(truth["active_minutes"] * 0.3),
                "veryActiveMinutes": int(truth["active_minutes"] * 0.2),
                "restingHeartRate": round(truth["resting_hr"]),
            },
        },
        "sleep": {
            "logId": int(RNG.integers(100000, 999999)),
            "dateOfSleep": d.isoformat(),
            "startTime": bedtime.isoformat(),
            "endTime": wake.isoformat(),
            "duration": int(truth["sleep_duration_min"] * 60 * 1000),
            "efficiency": round(truth["sleep_efficiency"]),
            "minutesAsleep": round(truth["sleep_duration_min"] * truth["sleep_efficiency"] / 100),
            "minutesAwake": round(truth["sleep_duration_min"] * (100 - truth["sleep_efficiency"]) / 100),
            "levels": {
                "summary": {
                    "deep": {"minutes": round(truth["sleep_duration_min"] * 0.15)},
                    "light": {"minutes": round(truth["sleep_duration_min"] * 0.50)},
                    "rem": {"minutes": round(truth["sleep_duration_min"] * 0.22)},
                    "wake": {"minutes": round(truth["sleep_duration_min"] * 0.13)},
                }
            },
        },
    }


def gen_oura_daily(truth: dict, user_id: str) -> dict:
    """Generate Oura daily sleep + readiness + activity JSON."""
    d = truth["date"]
    # Oura reports slightly different HR (finger sensor vs wrist)
    oura_rhr = truth["resting_hr"] + RNG.normal(-1.5, 1.0)
    oura_rhr = float(np.clip(oura_rhr, 45, 95))
    # Sleep timing offset: Oura detects sleep 5-15 min earlier
    offset_min = int(RNG.uniform(5, 15))
    bedtime = datetime(d.year, d.month, d.day, 22, 45) - timedelta(minutes=int(RNG.normal(20, 10)))
    wake = bedtime + timedelta(minutes=truth["sleep_duration_min"] + offset_min)
    sleep_sec = int(truth["sleep_duration_min"] * 60)

    return {
        "daily_sleep": {
            "id": str(uuid.uuid4()),
            "day": d.isoformat(),
            "score": min(100, max(1, round(truth["sleep_efficiency"] * 1.05))),
            "timestamp": bedtime.isoformat() + "+00:00",
            "total_sleep_duration": sleep_sec,
            "rem_sleep_duration": round(sleep_sec * 0.23),
            "deep_sleep_duration": round(sleep_sec * 0.16),
            "light_sleep_duration": round(sleep_sec * 0.48),
            "awake_time": round(sleep_sec * 0.13),
            "bedtime_start": bedtime.isoformat() + "+00:00",
            "bedtime_end": wake.isoformat() + "+00:00",
            "time_in_bed": int(sleep_sec * 1.08),
            "efficiency": round(truth["sleep_efficiency"]),
            "average_heart_rate": round(oura_rhr + 5, 1),
            "lowest_heart_rate": round(oura_rhr - 3),
            "average_hrv": round(truth["hrv_rmssd"]),
        },
        "daily_readiness": {
            "id": str(uuid.uuid4()),
            "day": d.isoformat(),
            "score": truth["readiness_score"],
            "timestamp": wake.isoformat() + "+00:00",
            "temperature_deviation": round(RNG.normal(0, 0.3), 2),
            "temperature_trend_deviation": round(RNG.normal(0, 0.1), 2),
            "contributors": {
                "activity_balance": round(RNG.uniform(60, 95)),
                "body_temperature": round(RNG.uniform(70, 100)),
                "hrv_balance": round(RNG.uniform(50, 95)),
                "resting_heart_rate": round(RNG.uniform(60, 100)),
                "sleep_balance": round(RNG.uniform(55, 95)),
            },
        },
        "daily_activity": {
            "id": str(uuid.uuid4()),
            "day": d.isoformat(),
            "score": min(100, max(1, round(truth["active_minutes"] / 2.5))),
            "active_calories": int(truth["calories"] * 0.35),
            "total_calories": truth["calories"],
            "steps": truth["steps"],
            "equivalent_walking_distance": truth["steps"],  # rough proxy
            "high_activity_time": truth["active_minutes"] * 12,  # seconds
            "medium_activity_time": truth["active_minutes"] * 18,
            "low_activity_time": truth["active_minutes"] * 30,
            "sedentary_time": truth["sedentary_minutes"] * 60,
        },
    }


def gen_whoop_daily(truth: dict, user_id: str) -> dict:
    """Generate WHOOP recovery + sleep + workout JSON."""
    d = truth["date"]
    # WHOOP HR uses optical sensor on wrist (bicep band historically more accurate)
    whoop_rhr = truth["resting_hr"] + RNG.normal(0.5, 1.5)
    whoop_rhr = float(np.clip(whoop_rhr, 45, 95))
    bedtime = datetime(d.year, d.month, d.day, 23, 10) - timedelta(minutes=int(RNG.normal(25, 10)))
    wake = bedtime + timedelta(minutes=truth["sleep_duration_min"])
    sleep_ms = int(truth["sleep_duration_min"] * 60 * 1000)

    result = {
        "recovery": {
            "cycle_id": int(RNG.integers(10000, 99999)),
            "sleep_id": int(RNG.integers(10000, 99999)),
            "user_id": int(RNG.integers(1000, 9999)),
            "created_at": wake.isoformat() + "Z",
            "score_state": "SCORED",
            "score": {
                "user_calibrating": False,
                "recovery_score": truth["recovery_score"],
                "resting_heart_rate": round(whoop_rhr, 1),
                "hrv_rmssd_milli": round(truth["hrv_rmssd"], 1),
                "spo2_percentage": round(RNG.uniform(95, 99), 1),
                "skin_temp_celsius": round(RNG.normal(33.5, 0.5), 1),
            },
        },
        "sleep": {
            "id": int(RNG.integers(10000, 99999)),
            "user_id": int(RNG.integers(1000, 9999)),
            "created_at": bedtime.isoformat() + "Z",
            "start": bedtime.isoformat() + "Z",
            "end": wake.isoformat() + "Z",
            "nap": False,
            "score_state": "SCORED",
            "score": {
                "stage_summary": {
                    "total_in_bed_time_milli": int(sleep_ms * 1.05),
                    "total_awake_time_milli": int(sleep_ms * 0.08),
                    "total_no_data_time_milli": 0,
                    "total_light_sleep_time_milli": int(sleep_ms * 0.48),
                    "total_slow_wave_sleep_time_milli": int(sleep_ms * 0.17),
                    "total_rem_sleep_time_milli": int(sleep_ms * 0.22),
                    "sleep_cycle_count": int(RNG.integers(3, 6)),
                    "disturbance_count": int(RNG.integers(0, 8)),
                },
                "respiratory_rate": round(RNG.normal(15.5, 1.2), 1),
                "sleep_performance_percentage": round(truth["sleep_efficiency"] * RNG.uniform(0.95, 1.05), 1),
                "sleep_efficiency_percentage": round(truth["sleep_efficiency"], 1),
            },
        },
    }

    if truth["has_workout"]:
        result["workout"] = {
            "id": int(RNG.integers(10000, 99999)),
            "user_id": int(RNG.integers(1000, 9999)),
            "created_at": (datetime.combine(d, datetime.min.time()) + timedelta(hours=int(RNG.uniform(7, 18)))).isoformat() + "Z",
            "start": (datetime.combine(d, datetime.min.time()) + timedelta(hours=int(RNG.uniform(7, 18)))).isoformat() + "Z",
            "end": (datetime.combine(d, datetime.min.time()) + timedelta(hours=int(RNG.uniform(7, 18)), seconds=truth["workout_duration_sec"])).isoformat() + "Z",
            "sport_id": int(RNG.integers(0, 80)),
            "score_state": "SCORED",
            "score": {
                "strain": truth["strain"],
                "average_heart_rate": round(truth["workout_avg_hr"]) if truth["workout_avg_hr"] else None,
                "max_heart_rate": round(truth["workout_max_hr"]) if truth["workout_max_hr"] else None,
                "kilojoule": round(truth["workout_calories"] * 4.184, 1) if truth["workout_calories"] else 0,
                "percent_recorded": round(RNG.uniform(90, 100), 1),
                "distance_meter": round(truth["workout_distance_m"], 1),
            },
        }

    return result


def gen_garmin_daily(truth: dict, user_id: str) -> dict:
    """Generate Garmin sleep + activity + daily summary JSON."""
    d = truth["date"]
    # Garmin calories are typically 10-20% higher than Fitbit
    garmin_cals = int(truth["calories"] * RNG.uniform(1.10, 1.20))
    garmin_rhr = truth["resting_hr"] + RNG.normal(1.0, 2.0)
    garmin_rhr = float(np.clip(garmin_rhr, 45, 95))
    bedtime_epoch = int(datetime(d.year, d.month, d.day, 23, 0).timestamp()) - int(RNG.normal(1800, 600))
    sleep_sec = int(truth["sleep_duration_min"] * 60)

    result = {
        "daily_summary": {
            "calendarDate": d.isoformat(),
            "steps": truth["steps"],
            "activeKilocalories": int(garmin_cals * 0.4),
            "burnedKilocalories": garmin_cals,
            "restingHeartRateInBeatsPerMinute": round(garmin_rhr),
            "averageHeartRateInBeatsPerMinute": round(garmin_rhr + RNG.normal(12, 3)),
            "maxHeartRateInBeatsPerMinute": round(garmin_rhr + RNG.normal(55, 15)),
            "averageStressLevel": truth["stress_level"],
            "bodyBatteryChargedValue": truth["body_battery"],
            "bodyBatteryDrainedValue": max(5, truth["body_battery"] - int(RNG.normal(40, 15))),
            "moderateIntensityDurationInSeconds": truth["active_minutes"] * 30,
            "vigorousIntensityDurationInSeconds": truth["active_minutes"] * 12,
        },
        "sleep": {
            "startTimeInSeconds": bedtime_epoch,
            "durationInSeconds": sleep_sec,
            "deepSleepDurationInSeconds": round(sleep_sec * 0.14),
            "lightSleepDurationInSeconds": round(sleep_sec * 0.52),
            "remSleepDurationInSeconds": round(sleep_sec * 0.21),
            "awakeDurationInSeconds": round(sleep_sec * 0.13),
        },
    }

    if truth["has_workout"]:
        workout_start = int(datetime.combine(d, datetime.min.time()).timestamp()) + int(RNG.uniform(25200, 64800))
        result["activity"] = {
            "activityId": int(RNG.integers(100000, 999999)),
            "activityType": truth["workout_type"].upper(),
            "startTimeInSeconds": workout_start,
            "durationInSeconds": truth["workout_duration_sec"],
            "distanceInMeters": round(truth["workout_distance_m"], 1),
            "activeKilocalories": truth["workout_calories"] if truth["workout_calories"] else 0,
            "averageHeartRateInBeatsPerMinute": round(truth["workout_avg_hr"] + RNG.normal(2, 1)) if truth["workout_avg_hr"] else None,
            "maxHeartRateInBeatsPerMinute": round(truth["workout_max_hr"] + RNG.normal(2, 2)) if truth["workout_max_hr"] else None,
        }

    return result


def gen_strava_activity(truth: dict, user_id: str) -> dict | None:
    """Generate Strava activity JSON (only if workout day)."""
    if not truth["has_workout"]:
        return None
    d = truth["date"]
    start = datetime.combine(d, datetime.min.time()) + timedelta(hours=int(RNG.uniform(6, 19)))

    return {
        "id": int(RNG.integers(1000000, 9999999)),
        "name": f"Morning {truth['workout_type']}",
        "type": truth["workout_type"],
        "sport_type": truth["workout_type"],
        "start_date": start.isoformat() + "Z",
        "start_date_local": start.isoformat(),
        "elapsed_time": truth["workout_duration_sec"],
        "moving_time": int(truth["workout_duration_sec"] * RNG.uniform(0.85, 0.98)),
        "distance": round(truth["workout_distance_m"], 1),
        "total_elevation_gain": round(RNG.uniform(10, 200), 1),
        "average_heartrate": truth["workout_avg_hr"],
        "max_heartrate": truth["workout_max_hr"],
        "kilojoules": round(truth["workout_calories"] * 4.184, 1) if truth["workout_calories"] else None,
        "calories": truth["workout_calories"],
    }


# ---------------------------------------------------------------------------
# Write raw JSON files
# ---------------------------------------------------------------------------


def write_raw_files(user: dict, ground_truth: list[dict]) -> dict:
    """Write per-provider JSON files and return file counts."""
    uid = user["id"]
    providers_to_gen = [user["primary"]] + user["secondary"]
    counts = {p: 0 for p in providers_to_gen}

    for day_idx, truth in enumerate(ground_truth):
        d = truth["date"]
        filename_base = f"{uid}_{d.isoformat()}"

        for provider in providers_to_gen:
            if is_missing(uid, provider, day_idx):
                continue

            if provider == "fitbit":
                payload = gen_fitbit_daily(truth, uid)
            elif provider == "oura":
                payload = gen_oura_daily(truth, uid)
            elif provider == "whoop":
                payload = gen_whoop_daily(truth, uid)
            elif provider == "garmin":
                payload = gen_garmin_daily(truth, uid)
            elif provider == "strava":
                payload = gen_strava_activity(truth, uid)
                if payload is None:
                    continue
            else:
                continue

            out_path = RAW_DIR / provider / f"{filename_base}.json"
            out_path.write_text(json.dumps(payload, indent=2, default=str))
            counts[provider] += 1

    return counts


# ---------------------------------------------------------------------------
# Normalization (lightweight, no DB required)
# ---------------------------------------------------------------------------


def normalize_all() -> dict[str, pl.DataFrame]:
    """Read all raw JSON and produce unified DataFrames."""
    daily_rows = []
    sleep_rows = []
    hr_rows = []
    activity_rows = []

    for provider_dir in RAW_DIR.iterdir():
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name

        for json_file in sorted(provider_dir.glob("*.json")):
            user_id = json_file.stem.rsplit("_", 1)[0]  # UUID part
            data = json.loads(json_file.read_text())

            if provider == "fitbit":
                _norm_fitbit(data, user_id, provider, daily_rows, sleep_rows, hr_rows)
            elif provider == "oura":
                _norm_oura(data, user_id, provider, daily_rows, sleep_rows, hr_rows)
            elif provider == "whoop":
                _norm_whoop(data, user_id, provider, daily_rows, sleep_rows, hr_rows, activity_rows)
            elif provider == "garmin":
                _norm_garmin(data, user_id, provider, daily_rows, sleep_rows, hr_rows, activity_rows)
            elif provider == "strava":
                _norm_strava(data, user_id, provider, activity_rows)

    tables = {}
    if daily_rows:
        tables["unified_daily_metrics"] = pl.DataFrame(daily_rows)
    if sleep_rows:
        tables["unified_sleep"] = pl.DataFrame(sleep_rows)
    if hr_rows:
        tables["unified_heart_rate"] = pl.DataFrame(hr_rows)
    if activity_rows:
        tables["unified_activities"] = pl.DataFrame(activity_rows)
    return tables


def _norm_fitbit(data, user_id, provider, daily_rows, sleep_rows, hr_rows):
    summary = data.get("activity_summary", {}).get("summary", {})
    d = data.get("activity_summary", {}).get("date")
    if summary and d:
        daily_rows.append({
            "user_id": user_id,
            "date": d,
            "steps": summary.get("steps"),
            "calories": summary.get("caloriesOut"),
            "sedentary_minutes": summary.get("sedentaryMinutes"),
            "lightly_active_minutes": summary.get("lightlyActiveMinutes"),
            "fairly_active_minutes": summary.get("fairlyActiveMinutes"),
            "very_active_minutes": summary.get("veryActiveMinutes"),
            "active_minutes": (summary.get("lightlyActiveMinutes", 0) +
                              summary.get("fairlyActiveMinutes", 0) +
                              summary.get("veryActiveMinutes", 0)),
            "resting_hr": summary.get("restingHeartRate"),
            "source": provider,
            "confidence": 0.9,
        })
        if summary.get("restingHeartRate"):
            hr_rows.append({
                "user_id": user_id,
                "date": d,
                "resting_hr": float(summary["restingHeartRate"]),
                "avg_hr": None,
                "source": provider,
                "confidence": 1.0,
            })

    sleep = data.get("sleep", {})
    if sleep and sleep.get("dateOfSleep"):
        sleep_rows.append({
            "user_id": user_id,
            "sleep_date": sleep["dateOfSleep"],
            "duration_minutes": sleep.get("minutesAsleep"),
            "efficiency": sleep.get("efficiency"),
            "start_time": sleep.get("startTime"),
            "end_time": sleep.get("endTime"),
            "source": provider,
            "confidence": 1.0,
        })


def _norm_oura(data, user_id, provider, daily_rows, sleep_rows, hr_rows):
    ds = data.get("daily_sleep", {})
    if ds and ds.get("day"):
        sleep_rows.append({
            "user_id": user_id,
            "sleep_date": ds["day"],
            "duration_minutes": ds.get("total_sleep_duration", 0) / 60 if ds.get("total_sleep_duration") else None,
            "efficiency": ds.get("efficiency"),
            "start_time": ds.get("bedtime_start"),
            "end_time": ds.get("bedtime_end"),
            "source": provider,
            "confidence": 1.0,
        })
        if ds.get("lowest_heart_rate"):
            hr_rows.append({
                "user_id": user_id,
                "date": ds["day"],
                "resting_hr": float(ds["lowest_heart_rate"]),
                "avg_hr": ds.get("average_heart_rate"),
                "source": provider,
                "confidence": 1.0,
            })

    da = data.get("daily_activity", {})
    if da and da.get("day"):
        daily_rows.append({
            "user_id": user_id,
            "date": da["day"],
            "steps": da.get("steps"),
            "calories": da.get("total_calories"),
            "sedentary_minutes": da.get("sedentary_time", 0) // 60 if da.get("sedentary_time") else None,
            "lightly_active_minutes": da.get("low_activity_time", 0) // 60 if da.get("low_activity_time") else None,
            "fairly_active_minutes": da.get("medium_activity_time", 0) // 60 if da.get("medium_activity_time") else None,
            "very_active_minutes": da.get("high_activity_time", 0) // 60 if da.get("high_activity_time") else None,
            "active_minutes": ((da.get("low_activity_time", 0) + da.get("medium_activity_time", 0) + da.get("high_activity_time", 0)) // 60),
            "resting_hr": None,
            "source": provider,
            "confidence": 0.9,
        })


def _norm_whoop(data, user_id, provider, daily_rows, sleep_rows, hr_rows, activity_rows):
    rec = data.get("recovery", {})
    score = rec.get("score", {})
    if score:
        created = rec.get("created_at", "")[:10]
        if score.get("resting_heart_rate"):
            hr_rows.append({
                "user_id": user_id,
                "date": created,
                "resting_hr": score["resting_heart_rate"],
                "avg_hr": None,
                "source": provider,
                "confidence": 1.0,
            })

    slp = data.get("sleep", {})
    slp_score = slp.get("score", {})
    stage = slp_score.get("stage_summary", {})
    if slp.get("start"):
        total_ms = (stage.get("total_light_sleep_time_milli", 0) +
                    stage.get("total_slow_wave_sleep_time_milli", 0) +
                    stage.get("total_rem_sleep_time_milli", 0))
        sleep_rows.append({
            "user_id": user_id,
            "sleep_date": slp["start"][:10],
            "duration_minutes": round(total_ms / 60000, 1) if total_ms else None,
            "efficiency": slp_score.get("sleep_efficiency_percentage"),
            "start_time": slp.get("start"),
            "end_time": slp.get("end"),
            "source": provider,
            "confidence": 1.0,
        })

    wkt = data.get("workout", {})
    wkt_score = wkt.get("score", {})
    if wkt and wkt.get("start"):
        activity_rows.append({
            "user_id": user_id,
            "date": wkt["start"][:10],
            "activity_type": f"sport_{wkt.get('sport_id', 0)}",
            "duration_seconds": None,
            "distance_meters": wkt_score.get("distance_meter"),
            "calories": round(wkt_score.get("kilojoule", 0) / 4.184) if wkt_score.get("kilojoule") else None,
            "avg_heart_rate": wkt_score.get("average_heart_rate"),
            "max_heart_rate": wkt_score.get("max_heart_rate"),
            "source": provider,
            "confidence": 1.0,
        })


def _norm_garmin(data, user_id, provider, daily_rows, sleep_rows, hr_rows, activity_rows):
    ds = data.get("daily_summary", {})
    if ds and ds.get("calendarDate"):
        d = ds["calendarDate"]
        daily_rows.append({
            "user_id": user_id,
            "date": d,
            "steps": ds.get("steps"),
            "calories": ds.get("burnedKilocalories"),
            "sedentary_minutes": None,
            "lightly_active_minutes": None,
            "fairly_active_minutes": ds.get("moderateIntensityDurationInSeconds", 0) // 60 if ds.get("moderateIntensityDurationInSeconds") else None,
            "very_active_minutes": ds.get("vigorousIntensityDurationInSeconds", 0) // 60 if ds.get("vigorousIntensityDurationInSeconds") else None,
            "active_minutes": ((ds.get("moderateIntensityDurationInSeconds", 0) + ds.get("vigorousIntensityDurationInSeconds", 0)) // 60),
            "resting_hr": ds.get("restingHeartRateInBeatsPerMinute"),
            "source": provider,
            "confidence": 0.9,
        })
        if ds.get("restingHeartRateInBeatsPerMinute"):
            hr_rows.append({
                "user_id": user_id,
                "date": d,
                "resting_hr": float(ds["restingHeartRateInBeatsPerMinute"]),
                "avg_hr": ds.get("averageHeartRateInBeatsPerMinute"),
                "source": provider,
                "confidence": 1.0,
            })

    slp = data.get("sleep", {})
    if slp and slp.get("durationInSeconds"):
        sleep_date = ds.get("calendarDate", "") if ds else ""
        sleep_rows.append({
            "user_id": user_id,
            "sleep_date": sleep_date,
            "duration_minutes": slp["durationInSeconds"] / 60,
            "efficiency": None,
            "start_time": None,
            "end_time": None,
            "source": provider,
            "confidence": 0.85,
        })

    act = data.get("activity", {})
    if act and act.get("activityId"):
        activity_rows.append({
            "user_id": user_id,
            "date": ds.get("calendarDate", "") if ds else "",
            "activity_type": act.get("activityType", "Unknown"),
            "duration_seconds": act.get("durationInSeconds"),
            "distance_meters": act.get("distanceInMeters"),
            "calories": act.get("activeKilocalories"),
            "avg_heart_rate": act.get("averageHeartRateInBeatsPerMinute"),
            "max_heart_rate": act.get("maxHeartRateInBeatsPerMinute"),
            "source": provider,
            "confidence": 1.0,
        })


def _norm_strava(data, user_id, provider, activity_rows):
    if data and data.get("id"):
        activity_rows.append({
            "user_id": user_id,
            "date": data.get("start_date", "")[:10],
            "activity_type": data.get("type", "Unknown"),
            "duration_seconds": data.get("elapsed_time"),
            "distance_meters": data.get("distance"),
            "calories": data.get("calories"),
            "avg_heart_rate": data.get("average_heartrate"),
            "max_heart_rate": data.get("max_heartrate"),
            "source": provider,
            "confidence": 1.0,
        })


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(tables: dict[str, pl.DataFrame], file_counts: dict) -> None:
    manifest = {
        "data_type": "synthetic",
        "description": "Synthetic multi-device data generated from LifeSnaps distributions + published norms",
        "created_at": datetime.utcnow().isoformat(),
        "seed": SEED,
        "n_users": len(USERS),
        "n_days": N_DAYS,
        "date_range": {"start": START_DATE.isoformat(), "end": (START_DATE + timedelta(days=N_DAYS - 1)).isoformat()},
        "users": [{
            "id": u["id"],
            "name": u["name"],
            "primary": u["primary"],
            "secondary": u["secondary"],
            "description": u["description"],
        } for u in USERS],
        "raw_files_per_provider": file_counts,
        "tables": {
            name: {"rows": len(df), "columns": df.columns}
            for name, df in tables.items()
        },
        "conflicts_introduced": [
            "HR disagreement: devices report resting HR within +/-3 bpm (wrist vs finger bias)",
            "Sleep timing offset: Fitbit vs Oura bedtime differs by 5-15 min",
            "Calorie divergence: Garmin reports 10-20% higher than Fitbit",
            "Activity double-counting: Strava workouts also in Garmin/WHOOP",
            "Missing data: user_01 missing Oura days 2,5,11; user_02 missing WHOOP days 4,7",
        ],
    }
    (NORM_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Generating synthetic multi-device wearable data")
    print(f"  Users: {len(USERS)}, Days: {N_DAYS}, Start: {START_DATE}")
    print()

    # Ensure directories exist
    for provider in ["fitbit", "oura", "whoop", "garmin", "strava"]:
        (RAW_DIR / provider).mkdir(parents=True, exist_ok=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    # Generate raw data
    all_file_counts: dict[str, int] = {}
    for user in USERS:
        print(f"  {user['name']} ({user['primary']} + {user['secondary']})")
        gt = generate_ground_truth(user)
        counts = write_raw_files(user, gt)
        for p, c in counts.items():
            all_file_counts[p] = all_file_counts.get(p, 0) + c
        print(f"    Files: {counts}")

    print(f"\n  Total raw files: {sum(all_file_counts.values())}")
    print(f"  Per provider: {all_file_counts}")

    # Normalize
    print("\n  Normalizing to unified tables...")
    tables = normalize_all()
    for name, df in tables.items():
        path = NORM_DIR / f"{name}.parquet"
        df.write_parquet(path)
        print(f"    {name}: {len(df)} rows -> {path.name}")

    # Write manifest
    write_manifest(tables, all_file_counts)
    print(f"\n  Manifest: {NORM_DIR / 'manifest.json'}")

    # Summary of conflicts
    print("\n  Conflicts introduced:")
    print("    - Resting HR varies +/-3 bpm across devices for same user/day")
    print("    - Sleep start/end times offset 5-15 min between Fitbit and Oura")
    print("    - Garmin calories 10-20% higher than Fitbit for same user")
    print("    - Workout appears in Strava + Garmin + WHOOP (triple-counted)")
    print("    - Missing data days per user (simulates forgotten/uncharged devices)")
    print("\n  Done!")


if __name__ == "__main__":
    main()
