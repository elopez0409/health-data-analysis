"""Normalizer dispatch: given a raw record, route to the correct normalizer."""

from app.models.raw import (
    RawFitbitActivity,
    RawFitbitHeartRate,
    RawFitbitSleep,
    RawGarminActivity,
    RawGarminSleep,
    RawOuraDailyActivity,
    RawOuraDailyReadiness,
    RawOuraDailySleep,
    RawStravaActivity,
    RawWithingsBloodPressure,
    RawWithingsSleep,
    RawWithingsWeight,
    RawWhoopRecovery,
    RawWhoopSleep,
    RawWhoopWorkout,
)
from app.models.unified import (
    UnifiedActivity,
    UnifiedBodyMetrics,
    UnifiedDailyMetrics,
    UnifiedHeartRate,
    UnifiedSleep,
)

from app.models.base import Base


def normalize(raw_record) -> Base | None:
    """Dispatch a raw record to the appropriate normalizer.

    Returns a unified model instance, or None if no normalizer is registered.
    """
    record_type = type(raw_record)

    if record_type is RawStravaActivity:
        from app.providers.strava.normalizer import normalize_strava_activity
        return normalize_strava_activity(raw_record)

    if record_type is RawFitbitSleep:
        from app.providers.fitbit.normalizer import normalize_fitbit_sleep
        return normalize_fitbit_sleep(raw_record)

    if record_type is RawFitbitActivity:
        from app.providers.fitbit.normalizer import normalize_fitbit_activity
        return normalize_fitbit_activity(raw_record)

    if record_type is RawFitbitHeartRate:
        from app.providers.fitbit.normalizer import normalize_fitbit_heart_rate
        return normalize_fitbit_heart_rate(raw_record)

    if record_type is RawOuraDailySleep:
        from app.providers.oura.normalizer import normalize_oura_sleep
        return normalize_oura_sleep(raw_record)

    if record_type is RawOuraDailyReadiness:
        from app.providers.oura.normalizer import normalize_oura_readiness
        return normalize_oura_readiness(raw_record)

    if record_type is RawOuraDailyActivity:
        from app.providers.oura.normalizer import normalize_oura_activity
        return normalize_oura_activity(raw_record)

    if record_type is RawWithingsWeight:
        from app.providers.withings.normalizer import normalize_withings_weight
        return normalize_withings_weight(raw_record)

    if record_type is RawWithingsSleep:
        from app.providers.withings.normalizer import normalize_withings_sleep
        return normalize_withings_sleep(raw_record)

    if record_type is RawWithingsBloodPressure:
        from app.providers.withings.normalizer import normalize_withings_bp
        return normalize_withings_bp(raw_record)

    if record_type is RawWhoopRecovery:
        from app.providers.whoop.normalizer import normalize_whoop_recovery
        return normalize_whoop_recovery(raw_record)

    if record_type is RawWhoopSleep:
        from app.providers.whoop.normalizer import normalize_whoop_sleep
        return normalize_whoop_sleep(raw_record)

    if record_type is RawWhoopWorkout:
        from app.providers.whoop.normalizer import normalize_whoop_workout
        return normalize_whoop_workout(raw_record)

    if record_type is RawGarminActivity:
        from app.providers.garmin.normalizer import normalize_garmin_activity
        return normalize_garmin_activity(raw_record)

    if record_type is RawGarminSleep:
        from app.providers.garmin.normalizer import normalize_garmin_sleep
        return normalize_garmin_sleep(raw_record)

    return None
