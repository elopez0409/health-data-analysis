import uuid
from datetime import datetime, timezone

from app.models.raw import RawWhoopRecovery, RawWhoopSleep, RawWhoopWorkout
from app.models.unified import UnifiedActivity, UnifiedDailyMetrics, UnifiedSleep
from app.providers.whoop.models import WhoopRecovery, WhoopSleep, WhoopWorkout

KJ_TO_KCAL = 0.239006


def normalize_whoop_recovery(raw: RawWhoopRecovery) -> UnifiedDailyMetrics:
    """Pure function: raw WHOOP recovery → unified daily metrics.

    Units:
    - hrv_rmssd_milli: divide by 1000 → standard ms (WHOOP reports micro-scale milli)
    - resting_heart_rate: bpm (no conversion)
    - recovery_score: 0–100 percentage
    """
    data = WhoopRecovery(**raw.payload)
    score = data.score

    return UnifiedDailyMetrics(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="whoop",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        metric_date=data.created_at.date(),
        recovery_score=score.recovery_score,
        hrv_rmssd=score.hrv_rmssd_milli / 1000.0,
        resting_heart_rate=score.resting_heart_rate,
    )


def normalize_whoop_sleep(raw: RawWhoopSleep) -> UnifiedSleep:
    """Pure function: raw WHOOP sleep → unified sleep.

    Units:
    - All stage durations: milliseconds → seconds (÷ 1000)
    - start/end: ISO timestamps (direct)
    """
    data = WhoopSleep(**raw.payload)
    stages = data.score.stage_summary

    total_sleep_milli = (
        stages.total_light_sleep_time_milli
        + stages.total_slow_wave_sleep_time_milli
        + stages.total_rem_sleep_time_milli
    )

    return UnifiedSleep(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="whoop",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        sleep_date=data.start.date(),
        bedtime=data.start,
        wake_time=data.end,
        total_seconds=total_sleep_milli / 1000.0,
        deep_seconds=stages.total_slow_wave_sleep_time_milli / 1000.0,
        light_seconds=stages.total_light_sleep_time_milli / 1000.0,
        rem_seconds=stages.total_rem_sleep_time_milli / 1000.0,
        awake_seconds=stages.total_awake_time_milli / 1000.0,
        sleep_score=data.score.sleep_performance_percentage,
    )


def normalize_whoop_workout(raw: RawWhoopWorkout) -> UnifiedActivity:
    """Pure function: raw WHOOP workout → unified activity.

    Units:
    - strain: dimensionless WHOOP strain score (0–21)
    - kilojoule → calories (× 0.239006)
    - distance_meter: already in meters
    - duration: computed from start/end timestamps
    """
    data = WhoopWorkout(**raw.payload)
    score = data.score

    duration = (data.end - data.start).total_seconds()
    calories = score.kilojoule * KJ_TO_KCAL

    return UnifiedActivity(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="whoop",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        activity_type=f"whoop_sport_{data.sport_id}",
        started_at=data.start,
        ended_at=data.end,
        duration_seconds=duration,
        distance_meters=score.distance_meter,
        calories=calories,
        avg_heart_rate_bpm=float(score.average_heart_rate) if score.average_heart_rate else None,
        max_heart_rate_bpm=float(score.max_heart_rate) if score.max_heart_rate else None,
        elevation_gain_meters=score.altitude_gain_meter,
        title=None,
    )
