import uuid

from sqlalchemy import Boolean, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Athlete(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "athletes"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_athlete_user_id"),
        UniqueConstraint("catapult_athlete_id", name="uq_athlete_catapult_id"),
    )

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    sport: Mapped[str | None] = mapped_column(String(100), nullable=True)
    position: Mapped[str | None] = mapped_column(String(50), nullable=True)
    jersey_number: Mapped[str | None] = mapped_column(String(10), nullable=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, default=uuid.uuid4
    )
    catapult_athlete_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
