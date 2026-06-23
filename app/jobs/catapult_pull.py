import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import async_session_factory
from app.logging import get_logger
from app.models.raw import RawCatapultActivity
from app.models.unified import UnifiedActivity
from app.providers.catapult.client import CatapultClient
from app.providers.catapult.normalizer import normalize_catapult_activity

logger = get_logger(__name__)


async def _get_cursor(session: AsyncSession, user_id: uuid.UUID) -> datetime | None:
    result = await session.execute(
        text(
            "SELECT last_value FROM ingestion_cursors "
            "WHERE user_id = :uid AND provider = 'catapult' AND resource = 'activities'"
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
            "VALUES (:uid, 'catapult', 'activities', :val, now()) "
            "ON CONFLICT ON CONSTRAINT uq_cursor "
            "DO UPDATE SET last_value = :val, updated_at = now()"
        ),
        {"uid": user_id, "val": value.isoformat()},
    )


async def _load_catapult_athlete_mapping(session: AsyncSession) -> dict[str, uuid.UUID]:
    """Build {catapult_athlete_id: user_id} mapping from athletes table."""
    from app.models.team import Athlete

    result = await session.execute(
        select(Athlete.catapult_athlete_id, Athlete.user_id).where(
            Athlete.catapult_athlete_id.isnot(None),
            Athlete.is_active.is_(True),
        )
    )
    return {row.catapult_athlete_id: row.user_id for row in result.all()}


async def catapult_pull_job(user_id: uuid.UUID | None = None) -> int:
    """Pull Catapult OpenField activity stats for the team. Returns count of new records."""
    default_user_id = uuid.UUID(settings.default_user_id)
    if user_id is None:
        user_id = default_user_id

    if not settings.catapult_api_token:
        logger.warning("catapult_pull_no_token", user_id=str(user_id))
        return 0

    async with async_session_factory() as session:
        # Load athlete mapping for resolving catapult athlete_ids to user_ids
        athlete_mapping = await _load_catapult_athlete_mapping(session)

        since = await _get_cursor(session, user_id)
        client = CatapultClient()
        raw_records = await client.pull(user_id, since=since)

        if not raw_records:
            logger.info("catapult_pull_no_new", user_id=str(user_id))
            return 0

        new_count = 0
        latest_time = since

        for record in raw_records:
            # Resolve user_id from athlete mapping, fall back to default
            record_athlete_id = str(record.payload.get("athlete_id", ""))
            resolved_user_id = athlete_mapping.get(record_athlete_id)
            if resolved_user_id is None:
                if record_athlete_id:
                    logger.warning(
                        "catapult_unmapped_athlete",
                        athlete_id=record_athlete_id,
                        external_id=record.external_id,
                    )
                resolved_user_id = default_user_id

            stmt = pg_insert(RawCatapultActivity).values(
                user_id=resolved_user_id,
                provider="catapult",
                external_id=record.external_id,
                fetched_at=record.fetched_at or datetime.now(timezone.utc),
                payload=record.payload,
                payload_hash=record.payload_hash,
            )
            stmt = stmt.on_conflict_on_constraint("uq_raw_catapult_activity").do_nothing()
            result = await session.execute(stmt)

            if result.rowcount == 0:
                continue

            raw_row = (
                await session.execute(
                    select(RawCatapultActivity).where(
                        RawCatapultActivity.external_id == record.external_id,
                        RawCatapultActivity.user_id == resolved_user_id,
                    )
                )
            ).scalar_one()

            unified = normalize_catapult_activity(raw_row)
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

            activity_time = unified.started_at
            if latest_time is None or activity_time > latest_time:
                latest_time = activity_time

        if latest_time and latest_time != since:
            await _update_cursor(session, user_id, latest_time)

        await session.commit()
        logger.info("catapult_pull_complete", user_id=str(user_id), new_count=new_count)
        return new_count
