"""Personal HR state and anomaly tables for the living model.

Revision ID: 002
Revises: 001
Create Date: 2026-06-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "personal_hr_state",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("trusted_source", sa.String(50), nullable=True),
        sa.Column("offset_mean", sa.Float, server_default="0.0"),
        sa.Column("offset_var", sa.Float, server_default="0.0"),
        sa.Column("n_samples", sa.Integer, server_default="0"),
        sa.Column("ci_low", sa.Float, nullable=True),
        sa.Column("ci_high", sa.Float, nullable=True),
        sa.Column("ewma_offset", sa.Float, server_default="0.0"),
        sa.Column(
            "last_updated",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("baseline_profile", JSONB, server_default="{}"),
        sa.UniqueConstraint("user_id", "source", name="uq_personal_hr_state"),
    )
    op.create_index(
        "ix_personal_hr_state_user_id",
        "personal_hr_state",
        ["user_id"],
    )

    op.create_table(
        "hr_anomalies",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("shift_bpm", sa.Float, nullable=False),
        sa.Column("prev_offset", sa.Float, nullable=True),
        sa.Column("new_offset", sa.Float, nullable=True),
        sa.Column("severity", sa.String(20), server_default="warning"),
        sa.Column("resolved", sa.Boolean, server_default="false"),
        sa.Column("details", JSONB, server_default="{}"),
    )
    op.create_index("ix_hr_anomalies_user_id", "hr_anomalies", ["user_id"])
    op.create_index(
        "ix_hr_anomalies_user_source",
        "hr_anomalies",
        ["user_id", "source"],
    )


def downgrade() -> None:
    op.drop_index("ix_hr_anomalies_user_source", table_name="hr_anomalies")
    op.drop_index("ix_hr_anomalies_user_id", table_name="hr_anomalies")
    op.drop_table("hr_anomalies")
    op.drop_index("ix_personal_hr_state_user_id", table_name="personal_hr_state")
    op.drop_table("personal_hr_state")
