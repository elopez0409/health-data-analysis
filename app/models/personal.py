"""SQLAlchemy models for the per-user living HR model."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin


class PersonalHrState(UUIDPrimaryKeyMixin, Base):
    """Per-user, per-source offset state for the living HR model."""

    __tablename__ = "personal_hr_state"
    __table_args__ = (
        UniqueConstraint("user_id", "source", name="uq_personal_hr_state"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    trusted_source: Mapped[str | None] = mapped_column(String(50), nullable=True)
    offset_mean: Mapped[float] = mapped_column(Float, default=0.0)
    offset_var: Mapped[float] = mapped_column(Float, default=0.0)
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    ewma_offset: Mapped[float] = mapped_column(Float, default=0.0)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
    )
    baseline_profile: Mapped[dict] = mapped_column(JSONB, default=dict)


class HrAnomaly(UUIDPrimaryKeyMixin, Base):
    """Detected shift in a device's HR offset (e.g. firmware update)."""

    __tablename__ = "hr_anomalies"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
    )
    shift_bpm: Mapped[float] = mapped_column(Float, nullable=False)
    prev_offset: Mapped[float | None] = mapped_column(Float, nullable=True)
    new_offset: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[str] = mapped_column(String(20), default="warning")
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
