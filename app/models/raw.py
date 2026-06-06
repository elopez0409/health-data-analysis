import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin


class RawRecordMixin(UUIDPrimaryKeyMixin):
    """Common columns for all raw tier tables."""

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)


# --- Strava ---


class RawStravaActivity(RawRecordMixin, Base):
    __tablename__ = "raw_strava_activities"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_strava_activities"
        ),
    )


# --- Fitbit ---


class RawFitbitSleep(RawRecordMixin, Base):
    __tablename__ = "raw_fitbit_sleep"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_fitbit_sleep"
        ),
    )


class RawFitbitHeartRate(RawRecordMixin, Base):
    __tablename__ = "raw_fitbit_heart_rate"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_fitbit_heart_rate"
        ),
    )


class RawFitbitActivity(RawRecordMixin, Base):
    __tablename__ = "raw_fitbit_activity"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_fitbit_activity"
        ),
    )


# --- Oura ---


class RawOuraDailySleep(RawRecordMixin, Base):
    __tablename__ = "raw_oura_daily_sleep"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_oura_daily_sleep"
        ),
    )


class RawOuraDailyReadiness(RawRecordMixin, Base):
    __tablename__ = "raw_oura_daily_readiness"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_oura_daily_readiness"
        ),
    )


class RawOuraDailyActivity(RawRecordMixin, Base):
    __tablename__ = "raw_oura_daily_activity"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_oura_daily_activity"
        ),
    )


# --- Withings ---


class RawWithingsWeight(RawRecordMixin, Base):
    __tablename__ = "raw_withings_weight"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_withings_weight"
        ),
    )


class RawWithingsSleep(RawRecordMixin, Base):
    __tablename__ = "raw_withings_sleep"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_withings_sleep"
        ),
    )


class RawWithingsBloodPressure(RawRecordMixin, Base):
    __tablename__ = "raw_withings_blood_pressure"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_withings_bp"
        ),
    )


# --- WHOOP ---


class RawWhoopRecovery(RawRecordMixin, Base):
    __tablename__ = "raw_whoop_recovery"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_whoop_recovery"
        ),
    )


class RawWhoopSleep(RawRecordMixin, Base):
    __tablename__ = "raw_whoop_sleep"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_whoop_sleep"
        ),
    )


class RawWhoopWorkout(RawRecordMixin, Base):
    __tablename__ = "raw_whoop_workout"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_whoop_workout"
        ),
    )


# --- Garmin ---


class RawGarminActivity(RawRecordMixin, Base):
    __tablename__ = "raw_garmin_activity"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_garmin_activity"
        ),
    )


class RawGarminSleep(RawRecordMixin, Base):
    __tablename__ = "raw_garmin_sleep"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_id", "user_id", name="uq_raw_garmin_sleep"
        ),
    )
