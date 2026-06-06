import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin


class UnifiedMixin(UUIDPrimaryKeyMixin):
    """Common columns for all unified tier tables."""

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    source_record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)


class UnifiedActivity(UnifiedMixin, Base):
    __tablename__ = "unified_activities"
    __table_args__ = (
        UniqueConstraint("source", "source_record_id", name="uq_unified_activities"),
    )

    activity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    distance_meters: Mapped[float | None] = mapped_column(Float, nullable=True)
    calories: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_heart_rate_bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_heart_rate_bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    elevation_gain_meters: Mapped[float | None] = mapped_column(Float, nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)


class UnifiedSleep(UnifiedMixin, Base):
    __tablename__ = "unified_sleep"
    __table_args__ = (
        UniqueConstraint("source", "source_record_id", name="uq_unified_sleep"),
    )

    sleep_date: Mapped[date] = mapped_column(Date, nullable=False)
    bedtime: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    wake_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    deep_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    light_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    rem_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    awake_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class UnifiedHeartRate(UnifiedMixin, Base):
    __tablename__ = "unified_heart_rate"
    __table_args__ = (
        UniqueConstraint("source", "source_record_id", name="uq_unified_heart_rate"),
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    bpm: Mapped[int] = mapped_column(Integer, nullable=False)
    context: Mapped[str | None] = mapped_column(String(50), nullable=True)


class UnifiedDailyMetrics(UnifiedMixin, Base):
    __tablename__ = "unified_daily_metrics"
    __table_args__ = (
        UniqueConstraint(
            "source", "source_record_id", name="uq_unified_daily_metrics"
        ),
    )

    metric_date: Mapped[date] = mapped_column(Date, nullable=False)
    steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    calories_total: Mapped[float | None] = mapped_column(Float, nullable=True)
    calories_active: Mapped[float | None] = mapped_column(Float, nullable=True)
    hrv_rmssd: Mapped[float | None] = mapped_column(Float, nullable=True)
    resting_heart_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    readiness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    strain_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recovery_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class UnifiedBodyMetrics(UnifiedMixin, Base):
    __tablename__ = "unified_body_metrics"
    __table_args__ = (
        UniqueConstraint(
            "source", "source_record_id", name="uq_unified_body_metrics"
        ),
    )

    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_fat_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    muscle_mass_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    systolic_bp: Mapped[float | None] = mapped_column(Float, nullable=True)
    diastolic_bp: Mapped[float | None] = mapped_column(Float, nullable=True)
