import uuid
from datetime import datetime, timedelta, timezone

from app.models.raw import RawGarminActivity, RawGarminSleep
from app.models.unified import UnifiedActivity, UnifiedSleep
from app.providers.garmin.models import GarminActivitySummary, GarminSleepSummary


def normalize_garmin_activity(raw: RawGarminActivity) -> UnifiedActivity:
    """Pure function: raw Garmin activity -> unified activity row.

    Garmin uses SI units natively — meters, seconds, kcal, bpm —
    so no unit conversion is needed.
    """
    data = GarminActivitySummary(**raw.payload)

    started_at = datetime.fromtimestamp(data.start_time_in_seconds, tz=timezone.utc)
    ended_at = started_at + timedelta(seconds=data.duration_in_seconds)

    return UnifiedActivity(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="garmin",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        activity_type=data.activity_type,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=float(data.duration_in_seconds),
        distance_meters=data.distance_in_meters,
        calories=data.active_kilocalories,
        avg_heart_rate_bpm=data.average_heart_rate_in_bpm,
        max_heart_rate_bpm=data.max_heart_rate_in_bpm,
        elevation_gain_meters=None,
        title=None,
    )


def normalize_garmin_sleep(raw: RawGarminSleep) -> UnifiedSleep:
    """Pure function: raw Garmin sleep -> unified sleep row.

    Timestamps are epoch seconds; durations already in seconds.
    """
    data = GarminSleepSummary(**raw.payload)

    bedtime = datetime.fromtimestamp(data.start_time_in_seconds, tz=timezone.utc)
    wake_time = bedtime + timedelta(seconds=data.duration_in_seconds)

    return UnifiedSleep(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="garmin",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        sleep_date=bedtime.date(),
        bedtime=bedtime,
        wake_time=wake_time,
        total_seconds=float(data.duration_in_seconds),
        deep_seconds=float(data.deep_sleep_duration_in_seconds),
        light_seconds=float(data.light_sleep_duration_in_seconds),
        rem_seconds=float(data.rem_sleep_duration_in_seconds),
        awake_seconds=float(data.awake_duration_in_seconds),
        sleep_score=None,
    )
