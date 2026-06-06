import uuid
from datetime import date, datetime, timezone

from app.models.raw import RawFitbitActivity, RawFitbitHeartRate, RawFitbitSleep
from app.models.unified import (
    UnifiedDailyMetrics,
    UnifiedHeartRate,
    UnifiedSleep,
)


def normalize_fitbit_sleep(raw: RawFitbitSleep) -> UnifiedSleep:
    """Pure function: raw Fitbit sleep log -> unified sleep row.

    Units:
    - duration: milliseconds -> seconds
    - stages: Fitbit provides minutes per stage -> convert to seconds
    """
    data = raw.payload

    sleep_date = date.fromisoformat(data["dateOfSleep"])
    bedtime = datetime.fromisoformat(data["startTime"]).replace(tzinfo=timezone.utc)
    wake_time = datetime.fromisoformat(data["endTime"]).replace(tzinfo=timezone.utc)
    total_seconds = data["duration"] / 1000.0

    # Stage durations from levels.summary
    deep_seconds = None
    light_seconds = None
    rem_seconds = None
    awake_seconds = None

    levels = data.get("levels", {})
    summary = levels.get("summary", {})
    if summary:
        if "deep" in summary:
            deep_seconds = summary["deep"].get("minutes", 0) * 60.0
        if "light" in summary:
            light_seconds = summary["light"].get("minutes", 0) * 60.0
        if "rem" in summary:
            rem_seconds = summary["rem"].get("minutes", 0) * 60.0
        if "wake" in summary:
            awake_seconds = summary["wake"].get("minutes", 0) * 60.0

    sleep_score = data.get("efficiency")

    return UnifiedSleep(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="fitbit",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        sleep_date=sleep_date,
        bedtime=bedtime,
        wake_time=wake_time,
        total_seconds=total_seconds,
        deep_seconds=deep_seconds,
        light_seconds=light_seconds,
        rem_seconds=rem_seconds,
        awake_seconds=awake_seconds,
        sleep_score=float(sleep_score) if sleep_score else None,
    )


def normalize_fitbit_activity(raw: RawFitbitActivity) -> UnifiedDailyMetrics:
    """Pure function: raw Fitbit daily activity summary -> unified daily metrics."""
    data = raw.payload
    summary = data.get("summary", {})

    date_str = raw.external_id.replace("activity-", "")
    metric_date = date.fromisoformat(date_str)

    return UnifiedDailyMetrics(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="fitbit",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        metric_date=metric_date,
        steps=summary.get("steps"),
        calories_total=float(summary.get("caloriesOut", 0)) if summary.get("caloriesOut") else None,
        calories_active=float(summary.get("activityCalories", 0)) if summary.get("activityCalories") else None,
        resting_heart_rate=float(summary.get("restingHeartRate")) if summary.get("restingHeartRate") else None,
    )


def normalize_fitbit_heart_rate(raw: RawFitbitHeartRate) -> UnifiedHeartRate:
    """Pure function: raw Fitbit HR intraday sample -> unified heart rate row."""
    data = raw.payload
    date_str = data["date"]
    time_str = data["time"]
    recorded_at = datetime.fromisoformat(f"{date_str}T{time_str}").replace(
        tzinfo=timezone.utc
    )

    return UnifiedHeartRate(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="fitbit",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        recorded_at=recorded_at,
        bpm=data["value"],
        context="resting",
    )
