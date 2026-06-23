"""Add athletes table.

Revision ID: 004
Revises: 003
Create Date: 2026-06-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "athletes",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("sport", sa.String(100), nullable=True),
        sa.Column("position", sa.String(50), nullable=True),
        sa.Column("jersey_number", sa.String(10), nullable=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("catapult_athlete_id", sa.String(100), nullable=True),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true")),
        sa.UniqueConstraint("user_id", name="uq_athlete_user_id"),
        sa.UniqueConstraint("catapult_athlete_id", name="uq_athlete_catapult_id"),
    )


def downgrade() -> None:
    op.drop_table("athletes")
