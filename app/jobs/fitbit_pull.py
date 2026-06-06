import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_factory
from app.logging import get_logger
from app.models.raw import RawFitbitActivity, RawFitbitSleep
from app.models.unified import UnifiedDailyMetrics, UnifiedSleep
from app.providers.fitbit.client import FitbitClient
from app.providers.fitbit.normalizer import (
    normalize_fitbit_activity,
    normalize_fitbit_sleep,
)
from app.providers.token_store import get_token, is_token_expired, upsert_token
from app.schemas.common import Provider

logger = get_logger(__name__)


async def _get_cursor(
    session: AsyncSession, user_id: uuid.UUID, resource: str
) -> datetime | None:
    result = await session.execute(
        text(
            "SELECT last_value FROM ingestion_cursors "
            "WHERE user_id = :uid AND provider = 'fitbit' AND resource = :resource"
        ),
        {"uid": user_id, "resource": resource},
    )
    row = result.first()
    if row:
        return datetime.fromisoformat(row[0])
    return None


async def _update_cursor(
    session: AsyncSession, user_id: uuid.UUID, resource: str, value: datetime
) -> None:
    await session.execute(
        text(
            "INSERT INTO ingestion_cursors (user_id, provider, resource, last_value, updated_at) "
            "VALUES (:uid, 'fitbit', :resource, :val, now()) "
            "ON CONFLICT ON CONSTRAINT uq_cursor "
            "DO UPDATE SET last_value = :val, updated_at = now()"
        ),
        {"uid": user_id, "resource": resource, "val": value.isoformat()},
    )


async def fitbit_pull_job(user_id: uuid.UUID | None = None) -> int:
    """Pull Fitbit sleep and activity data for a user."""
    from app.config import settings

    if user_id is None:
        user_id = uuid.UUID(settings.default_user_id)

    async with async_session_factory() as session:
        token = await get_token(session, user_id, Provider.FITBIT)
        if not token:
            logger.warning("fitbit_pull_no_token", user_id=str(user_id))
            return 0

        if is_token_expired(token):
            client_tmp = FitbitClient()
            new_tokens = await client_tmp.refresh_token(token.refresh_token)
            expires_at = datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + new_tokens["expires_in"],
                tz=timezone.utc,
            )
            await upsert_token(
                session=session,
                user_id=user_id,
                provider=Provider.FITBIT,
                access_token=new_tokens["access_token"],
                refresh_token=new_tokens.get("refresh_token", token.refresh_token),
                expires_at=expires_at,
            )
            access_token = new_tokens["access_token"]
        else:
            access_token = token.access_token

        since = await _get_cursor(session, user_id, "sleep")
        client = FitbitClient(access_token=access_token)
        raw_records = await client.pull(user_id, since=since)

        new_count = 0

        for record in raw_records:
            # Determine if sleep or activity
            if record.external_id.startswith("activity-"):
                stmt = pg_insert(RawFitbitActivity).values(
                    user_id=user_id,
                    provider="fitbit",
                    external_id=record.external_id,
                    fetched_at=record.fetched_at or datetime.now(timezone.utc),
                    payload=record.payload,
                    payload_hash=record.payload_hash,
                )
                stmt = stmt.on_conflict_on_constraint("uq_raw_fitbit_activity").do_nothing()
                result = await session.execute(stmt)

                if result.rowcount > 0:
                    raw_row = (
                        await session.execute(
                            select(RawFitbitActivity).where(
                                RawFitbitActivity.external_id == record.external_id,
                                RawFitbitActivity.user_id == user_id,
                            )
                        )
                    ).scalar_one()

                    unified = normalize_fitbit_activity(raw_row)
                    u_stmt = pg_insert(UnifiedDailyMetrics).values(
                        id=unified.id,
                        user_id=unified.user_id,
                        source=unified.source,
                        source_record_id=unified.source_record_id,
                        ingested_at=unified.ingested_at,
                        confidence=unified.confidence,
                        metric_date=unified.metric_date,
                        steps=unified.steps,
                        calories_total=unified.calories_total,
                        calories_active=unified.calories_active,
                        resting_heart_rate=unified.resting_heart_rate,
                    )
                    u_stmt = u_stmt.on_conflict_on_constraint("uq_unified_daily_metrics").do_nothing()
                    await session.execute(u_stmt)
                    new_count += 1
            else:
                stmt = pg_insert(RawFitbitSleep).values(
                    user_id=user_id,
                    provider="fitbit",
                    external_id=record.external_id,
                    fetched_at=record.fetched_at or datetime.now(timezone.utc),
                    payload=record.payload,
                    payload_hash=record.payload_hash,
                )
                stmt = stmt.on_conflict_on_constraint("uq_raw_fitbit_sleep").do_nothing()
                result = await session.execute(stmt)

                if result.rowcount > 0:
                    raw_row = (
                        await session.execute(
                            select(RawFitbitSleep).where(
                                RawFitbitSleep.external_id == record.external_id,
                                RawFitbitSleep.user_id == user_id,
                            )
                        )
                    ).scalar_one()

                    unified = normalize_fitbit_sleep(raw_row)
                    u_stmt = pg_insert(UnifiedSleep).values(
                        id=unified.id,
                        user_id=unified.user_id,
                        source=unified.source,
                        source_record_id=unified.source_record_id,
                        ingested_at=unified.ingested_at,
                        confidence=unified.confidence,
                        sleep_date=unified.sleep_date,
                        bedtime=unified.bedtime,
                        wake_time=unified.wake_time,
                        total_seconds=unified.total_seconds,
                        deep_seconds=unified.deep_seconds,
                        light_seconds=unified.light_seconds,
                        rem_seconds=unified.rem_seconds,
                        awake_seconds=unified.awake_seconds,
                        sleep_score=unified.sleep_score,
                    )
                    u_stmt = u_stmt.on_conflict_on_constraint("uq_unified_sleep").do_nothing()
                    await session.execute(u_stmt)
                    new_count += 1

        if new_count > 0:
            await _update_cursor(session, user_id, "sleep", datetime.now(timezone.utc))

        await session.commit()
        logger.info("fitbit_pull_complete", user_id=str(user_id), new_count=new_count)
        return new_count
