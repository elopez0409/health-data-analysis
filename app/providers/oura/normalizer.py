import uuid
from datetime import date, datetime, timezone

from app.models.raw import RawOuraDailyActivity, RawOuraDailyReadiness, RawOuraDailySleep
from app.models.unified import UnifiedDailyMetrics, UnifiedSleep
from app.providers.oura.models import OuraDailyActivity, OuraDailyReadiness, OuraDailySleep


def normalize_oura_sleep(raw: RawOuraDailySleep) -> UnifiedSleep:
    """Pure function: raw Oura daily sleep -> unified sleep row.

    Units: Oura provides all durations in seconds natively — no conversion needed.
    """
    data = OuraDailySleep(**raw.payload)

    sleep_date = date.fromisoformat(data.day)

    bedtime = None
    if data.bedtime_start:
        bedtime = data.bedtime_start if data.bedtime_start.tzinfo else data.bedtime_start.replace(tzinfo=timezone.utc)

    wake_time = None
    if data.bedtime_end:
        wake_time = data.bedtime_end if data.bedtime_end.tzinfo else data.bedtime_end.replace(tzinfo=timezone.utc)

    return UnifiedSleep(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="oura",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        sleep_date=sleep_date,
        bedtime=bedtime,
        wake_time=wake_time,
        total_seconds=float(data.total_sleep_duration) if data.total_sleep_duration is not None else None,
        deep_seconds=float(data.deep_sleep_duration) if data.deep_sleep_duration is not None else None,
        light_seconds=float(data.light_sleep_duration) if data.light_sleep_duration is not None else None,
        rem_seconds=float(data.rem_sleep_duration) if data.rem_sleep_duration is not None else None,
        awake_seconds=float(data.awake_time) if data.awake_time is not None else None,
        sleep_score=float(data.score) if data.score is not None else None,
    )


def normalize_oura_readiness(raw: RawOuraDailyReadiness) -> UnifiedDailyMetrics:
    """Pure function: raw Oura daily readiness -> unified daily metrics row.

    Maps readiness score to readiness_score. Extracts HRV baseline from
    hrv_balance contributor when available.
    """
    data = OuraDailyReadiness(**raw.payload)

    metric_date = date.fromisoformat(data.day)

    hrv_rmssd = None
    if data.hrv_balance and isinstance(data.hrv_balance, dict):
        hrv_rmssd = data.hrv_balance.get("value")
        if hrv_rmssd is not None:
            hrv_rmssd = float(hrv_rmssd)

    return UnifiedDailyMetrics(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="oura",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        metric_date=metric_date,
        readiness_score=float(data.score) if data.score is not None else None,
        hrv_rmssd=hrv_rmssd,
    )


def normalize_oura_activity(raw: RawOuraDailyActivity) -> UnifiedDailyMetrics:
    """Pure function: raw Oura daily activity -> unified daily metrics row.

    Maps active_calories, steps, and total_calories.
    """
    data = OuraDailyActivity(**raw.payload)

    metric_date = date.fromisoformat(data.day)

    return UnifiedDailyMetrics(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="oura",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        metric_date=metric_date,
        steps=data.steps,
        calories_total=float(data.total_calories) if data.total_calories is not None else None,
        calories_active=float(data.active_calories) if data.active_calories is not None else None,
    )
