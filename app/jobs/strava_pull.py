import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_factory
from app.logging import get_logger
from app.models.raw import RawStravaActivity
from app.models.unified import UnifiedActivity
from app.providers.base import RawRecord
from app.providers.strava.client import StravaClient
from app.providers.strava.normalizer import normalize_strava_activity
from app.providers.token_store import get_token, is_token_expired, upsert_token
from app.schemas.common import Provider

logger = get_logger(__name__)


async def _get_cursor(session: AsyncSession, user_id: uuid.UUID) -> datetime | None:
    result = await session.execute(
        text(
            "SELECT last_value FROM ingestion_cursors "
            "WHERE user_id = :uid AND provider = 'strava' AND resource = 'activities'"
        ),
        {"uid": user_id},
    )
    row = result.first()
    if row:
        return datetime.fromisoformat(row[0])
    return None


async def _update_cursor(
    session: AsyncSession, user_id: uuid.UUID, value: datetime
) -> None:
    await session.execute(
        text(
            "INSERT INTO ingestion_cursors (user_id, provider, resource, last_value, updated_at) "
            "VALUES (:uid, 'strava', 'activities', :val, now()) "
            "ON CONFLICT ON CONSTRAINT uq_cursor "
            "DO UPDATE SET last_value = :val, updated_at = now()"
        ),
        {"uid": user_id, "val": value.isoformat()},
    )


async def strava_pull_job(user_id: uuid.UUID | None = None) -> int:
    """Pull Strava activities for a user. Returns count of new records."""
    from app.config import settings

    if user_id is None:
        user_id = uuid.UUID(settings.default_user_id)

    async with async_session_factory() as session:
        token = await get_token(session, user_id, Provider.STRAVA)
        if not token:
            logger.warning("strava_pull_no_token", user_id=str(user_id))
            return 0

        # Refresh if expired
        if is_token_expired(token):
            client_tmp = StravaClient()
            new_tokens = await client_tmp.refresh_token(token.refresh_token)
            await upsert_token(
                session=session,
                user_id=user_id,
                provider=Provider.STRAVA,
                access_token=new_tokens["access_token"],
                refresh_token=new_tokens.get("refresh_token", token.refresh_token),
                expires_at=datetime.fromtimestamp(
                    new_tokens["expires_at"], tz=timezone.utc
                ),
            )
            access_token = new_tokens["access_token"]
        else:
            access_token = token.access_token

        since = await _get_cursor(session, user_id)

        client = StravaClient(access_token=access_token)
        raw_records = await client.pull(user_id, since=since)

        if not raw_records:
            logger.info("strava_pull_no_new", user_id=str(user_id))
            return 0

        new_count = 0
        latest_time = since

        for record in raw_records:
            # Upsert raw
            stmt = pg_insert(RawStravaActivity).values(
                user_id=user_id,
                provider="strava",
                external_id=record.external_id,
                fetched_at=record.fetched_at or datetime.now(timezone.utc),
                payload=record.payload,
                payload_hash=record.payload_hash,
            )
            stmt = stmt.on_conflict_on_constraint("uq_raw_strava_activities").do_nothing()
            result = await session.execute(stmt)

            if result.rowcount == 0:
                continue

            # Get the raw row back for normalization
            raw_row = (
                await session.execute(
                    select(RawStravaActivity).where(
                        RawStravaActivity.external_id == record.external_id,
                        RawStravaActivity.user_id == user_id,
                    )
                )
            ).scalar_one()

            # Normalize
            unified = normalize_strava_activity(raw_row)
            unified_stmt = pg_insert(UnifiedActivity).values(
                id=unified.id,
                user_id=unified.user_id,
                source=unified.source,
                source_record_id=unified.source_record_id,
                ingested_at=unified.ingested_at,
                confidence=unified.confidence,
                activity_type=unified.activity_type,
                started_at=unified.started_at,
                ended_at=unified.ended_at,
                duration_seconds=unified.duration_seconds,
                distance_meters=unified.distance_meters,
                calories=unified.calories,
                avg_heart_rate_bpm=unified.avg_heart_rate_bpm,
                max_heart_rate_bpm=unified.max_heart_rate_bpm,
                elevation_gain_meters=unified.elevation_gain_meters,
                title=unified.title,
            )
            unified_stmt = unified_stmt.on_conflict_on_constraint(
                "uq_unified_activities"
            ).do_nothing()
            await session.execute(unified_stmt)
            new_count += 1

            # Track latest time for cursor
            activity_time = unified.started_at
            if latest_time is None or activity_time > latest_time:
                latest_time = activity_time

        if latest_time and latest_time != since:
            await _update_cursor(session, user_id, latest_time)

        await session.commit()
        logger.info("strava_pull_complete", user_id=str(user_id), new_count=new_count)
        return new_count
