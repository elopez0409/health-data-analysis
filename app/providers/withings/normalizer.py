import uuid
from datetime import datetime, timezone

from app.models.raw import RawWithingsBloodPressure, RawWithingsSleep, RawWithingsWeight
from app.models.unified import UnifiedBodyMetrics, UnifiedSleep
from app.providers.withings.models import (
    WithingsMeasureGroup,
    WithingsMeasureType,
    WithingsSleepSeries,
)


def normalize_withings_weight(raw: RawWithingsWeight) -> UnifiedBodyMetrics:
    """Pure function: raw Withings weight measure group → unified body metrics.

    Withings stores measures as ``value * 10^unit``; WithingsMeasureGroup.get_measure
    handles the conversion.
    """
    grp = WithingsMeasureGroup(**raw.payload)
    measured_at = datetime.fromtimestamp(grp.date, tz=timezone.utc)

    return UnifiedBodyMetrics(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="withings",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        measured_at=measured_at,
        weight_kg=grp.get_measure(WithingsMeasureType.WEIGHT_KG),
        body_fat_pct=grp.get_measure(WithingsMeasureType.FAT_RATIO_PCT),
        muscle_mass_kg=grp.get_measure(WithingsMeasureType.MUSCLE_MASS_KG),
    )


def normalize_withings_sleep(raw: RawWithingsSleep) -> UnifiedSleep:
    """Pure function: raw Withings sleep series → unified sleep row.

    Epoch timestamps are converted to UTC datetimes.  Stage durations are
    already in seconds — no unit conversion needed.
    """
    series = WithingsSleepSeries(**raw.payload)
    bedtime = datetime.fromtimestamp(series.startdate, tz=timezone.utc)
    wake_time = datetime.fromtimestamp(series.enddate, tz=timezone.utc)
    total_seconds = float(series.enddate - series.startdate)

    return UnifiedSleep(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="withings",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        sleep_date=wake_time.date(),
        bedtime=bedtime,
        wake_time=wake_time,
        total_seconds=total_seconds,
        deep_seconds=float(series.deepsleepduration) if series.deepsleepduration is not None else None,
        light_seconds=float(series.lightsleepduration) if series.lightsleepduration is not None else None,
        rem_seconds=float(series.remsleepduration) if series.remsleepduration is not None else None,
        awake_seconds=float(series.wakeupduration) if series.wakeupduration is not None else None,
    )


def normalize_withings_bp(raw: RawWithingsBloodPressure) -> UnifiedBodyMetrics:
    """Pure function: raw Withings blood pressure measure group → unified body metrics."""
    grp = WithingsMeasureGroup(**raw.payload)
    measured_at = datetime.fromtimestamp(grp.date, tz=timezone.utc)

    return UnifiedBodyMetrics(
        id=uuid.uuid4(),
        user_id=raw.user_id,
        source="withings",
        source_record_id=raw.id,
        ingested_at=datetime.now(timezone.utc),
        confidence=1.0,
        measured_at=measured_at,
        systolic_bp=grp.get_measure(WithingsMeasureType.SYSTOLIC_BP),
        diastolic_bp=grp.get_measure(WithingsMeasureType.DIASTOLIC_BP),
    )
