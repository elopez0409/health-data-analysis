import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tokens import ProviderToken
from app.schemas.common import Provider


async def get_token(
    session: AsyncSession, user_id: uuid.UUID, provider: Provider
) -> ProviderToken | None:
    stmt = select(ProviderToken).where(
        ProviderToken.user_id == user_id,
        ProviderToken.provider == provider.value,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def upsert_token(
    session: AsyncSession,
    user_id: uuid.UUID,
    provider: Provider,
    access_token: str,
    refresh_token: str | None = None,
    expires_at: datetime | None = None,
    scopes: str | None = None,
    extra: dict | None = None,
) -> None:
    stmt = pg_insert(ProviderToken).values(
        user_id=user_id,
        provider=provider.value,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        scopes=scopes,
        extra=extra,
        updated_at=datetime.now(timezone.utc),
    )
    stmt = stmt.on_conflict_on_constraint("uq_user_provider").do_update(
        set_={
            "access_token": stmt.excluded.access_token,
            "refresh_token": stmt.excluded.refresh_token,
            "expires_at": stmt.excluded.expires_at,
            "scopes": stmt.excluded.scopes,
            "extra": stmt.excluded.extra,
            "updated_at": stmt.excluded.updated_at,
        }
    )
    await session.execute(stmt)
    await session.commit()


def is_token_expired(token: ProviderToken) -> bool:
    if token.expires_at is None:
        return False
    return datetime.now(timezone.utc) >= token.expires_at
