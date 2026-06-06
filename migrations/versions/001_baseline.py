"""Baseline: TimescaleDB, provider_tokens, raw tier, unified tier

Revision ID: 001
Revises: None
Create Date: 2026-05-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_raw_table(name: str, constraint_name: str) -> None:
    op.create_table(
        name,
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("external_id", sa.String, nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.UniqueConstraint("provider", "external_id", "user_id", name=constraint_name),
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    # Provider tokens
    op.create_table(
        "provider_tokens",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("access_token", sa.String, nullable=False),
        sa.Column("refresh_token", sa.String, nullable=True),
        sa.Column("token_type", sa.String(50), server_default="Bearer"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scopes", sa.String, nullable=True),
        sa.Column("extra", JSONB, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("user_id", "provider", name="uq_user_provider"),
    )

    # Ingestion cursors
    op.create_table(
        "ingestion_cursors",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("resource", sa.String(100), nullable=False),
        sa.Column("last_value", sa.String, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("user_id", "provider", "resource", name="uq_cursor"),
    )

    # --- Raw tier tables ---
    _create_raw_table("raw_strava_activities", "uq_raw_strava_activities")
    _create_raw_table("raw_fitbit_sleep", "uq_raw_fitbit_sleep")
    _create_raw_table("raw_fitbit_heart_rate", "uq_raw_fitbit_heart_rate")
    _create_raw_table("raw_fitbit_activity", "uq_raw_fitbit_activity")
    _create_raw_table("raw_oura_daily_sleep", "uq_raw_oura_daily_sleep")
    _create_raw_table("raw_oura_daily_readiness", "uq_raw_oura_daily_readiness")
    _create_raw_table("raw_oura_daily_activity", "uq_raw_oura_daily_activity")
    _create_raw_table("raw_withings_weight", "uq_raw_withings_weight")
    _create_raw_table("raw_withings_sleep", "uq_raw_withings_sleep")
    _create_raw_table("raw_withings_blood_pressure", "uq_raw_withings_bp")
    _create_raw_table("raw_whoop_recovery", "uq_raw_whoop_recovery")
    _create_raw_table("raw_whoop_sleep", "uq_raw_whoop_sleep")
    _create_raw_table("raw_whoop_workout", "uq_raw_whoop_workout")
    _create_raw_table("raw_garmin_activity", "uq_raw_garmin_activity")
    _create_raw_table("raw_garmin_sleep", "uq_raw_garmin_sleep")

    # --- Unified tier tables ---

    # Unified activities
    op.create_table(
        "unified_activities",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_record_id", UUID(as_uuid=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("activity_type", sa.String(100), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("distance_meters", sa.Float, nullable=True),
        sa.Column("calories", sa.Float, nullable=True),
        sa.Column("avg_heart_rate_bpm", sa.Float, nullable=True),
        sa.Column("max_heart_rate_bpm", sa.Float, nullable=True),
        sa.Column("elevation_gain_meters", sa.Float, nullable=True),
        sa.Column("title", sa.String, nullable=True),
        sa.UniqueConstraint("source", "source_record_id", name="uq_unified_activities"),
    )

    # Unified sleep
    op.create_table(
        "unified_sleep",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_record_id", UUID(as_uuid=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("sleep_date", sa.Date, nullable=False),
        sa.Column("bedtime", sa.DateTime(timezone=True), nullable=True),
        sa.Column("wake_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_seconds", sa.Float, nullable=True),
        sa.Column("deep_seconds", sa.Float, nullable=True),
        sa.Column("light_seconds", sa.Float, nullable=True),
        sa.Column("rem_seconds", sa.Float, nullable=True),
        sa.Column("awake_seconds", sa.Float, nullable=True),
        sa.Column("sleep_score", sa.Float, nullable=True),
        sa.UniqueConstraint("source", "source_record_id", name="uq_unified_sleep"),
    )

    # Unified heart rate (will be converted to hypertable)
    op.create_table(
        "unified_heart_rate",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_record_id", UUID(as_uuid=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bpm", sa.Integer, nullable=False),
        sa.Column("context", sa.String(50), nullable=True),
        sa.UniqueConstraint("source", "source_record_id", name="uq_unified_heart_rate"),
    )

    # Convert heart rate to TimescaleDB hypertable
    op.execute(
        "SELECT create_hypertable('unified_heart_rate', 'recorded_at', "
        "chunk_time_interval => INTERVAL '7 days', migrate_data => true)"
    )

    # Unified daily metrics
    op.create_table(
        "unified_daily_metrics",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_record_id", UUID(as_uuid=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("metric_date", sa.Date, nullable=False),
        sa.Column("steps", sa.Integer, nullable=True),
        sa.Column("calories_total", sa.Float, nullable=True),
        sa.Column("calories_active", sa.Float, nullable=True),
        sa.Column("hrv_rmssd", sa.Float, nullable=True),
        sa.Column("resting_heart_rate", sa.Float, nullable=True),
        sa.Column("readiness_score", sa.Float, nullable=True),
        sa.Column("strain_score", sa.Float, nullable=True),
        sa.Column("recovery_score", sa.Float, nullable=True),
        sa.UniqueConstraint("source", "source_record_id", name="uq_unified_daily_metrics"),
    )

    # Unified body metrics
    op.create_table(
        "unified_body_metrics",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_record_id", UUID(as_uuid=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("measured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("weight_kg", sa.Float, nullable=True),
        sa.Column("body_fat_pct", sa.Float, nullable=True),
        sa.Column("muscle_mass_kg", sa.Float, nullable=True),
        sa.Column("systolic_bp", sa.Float, nullable=True),
        sa.Column("diastolic_bp", sa.Float, nullable=True),
        sa.UniqueConstraint("source", "source_record_id", name="uq_unified_body_metrics"),
    )


def downgrade() -> None:
    op.drop_table("unified_body_metrics")
    op.drop_table("unified_daily_metrics")
    op.drop_table("unified_heart_rate")
    op.drop_table("unified_sleep")
    op.drop_table("unified_activities")
    op.drop_table("raw_garmin_sleep")
    op.drop_table("raw_garmin_activity")
    op.drop_table("raw_whoop_workout")
    op.drop_table("raw_whoop_sleep")
    op.drop_table("raw_whoop_recovery")
    op.drop_table("raw_withings_blood_pressure")
    op.drop_table("raw_withings_sleep")
    op.drop_table("raw_withings_weight")
    op.drop_table("raw_oura_daily_activity")
    op.drop_table("raw_oura_daily_readiness")
    op.drop_table("raw_oura_daily_sleep")
    op.drop_table("raw_fitbit_activity")
    op.drop_table("raw_fitbit_heart_rate")
    op.drop_table("raw_fitbit_sleep")
    op.drop_table("raw_strava_activities")
    op.drop_table("ingestion_cursors")
    op.drop_table("provider_tokens")
