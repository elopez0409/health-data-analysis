import uuid
from datetime import datetime, timedelta, timezone

from app.models.raw import RawStravaActivity
from app.models.unified import UnifiedActivity
from app.providers.strava.models import StravaActivity


def normalize_strava_activity(
    raw: RawStravaActivity,
) -> UnifiedActivity:
    """Pure function: raw Strava activity -> unified activity row.

    Units:
    - distance: already in meters from Strava
    - duration: elapsed_time already in seconds
    - elevation: total_elevation_gain already in meters
    - HR: bpm (no conversion needed)
    - calories: Strava provides kilojoules for rides, direct calories for others
    """
    data = StravaActivity(**raw.payload)

    calories = data.calories
    if calories is None and data.kilojoules is not None:
        # Strava kilojoules -> kcal (1 kJ ≈ 0.239 kcal, but Strava kJ are
        # already metabolic work so 1:1 with kcal is standard convention)
        calories = data.kilojoules

    started_at = data.start_date.replace(tzinfo=timezone.utc) if data.start_date.tzinfo is None else data.start_date
    ended_at = started_at + timedelta(seconds=data.elapsed_time)

    return UnifiedActivity(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="strava",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        activity_type=data.sport_type or data.type,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=float(data.elapsed_time),
        distance_meters=data.distance,
        calories=calories,
        avg_heart_rate_bpm=data.average_heartrate,
        max_heart_rate_bpm=data.max_heartrate,
        elevation_gain_meters=data.total_elevation_gain,
        title=data.name,
    )
