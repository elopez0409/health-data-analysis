import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.providers.registry import ProviderRegistry
from app.providers.token_store import upsert_token
from app.schemas.common import Provider

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/{provider}/start")
async def oauth_start(provider: str):
    try:
        prov = Provider(provider)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    client = ProviderRegistry.get_client(prov)
    state = f"{settings.default_user_id}:{provider}"
    url = await client.get_authorize_url(state=state)
    return RedirectResponse(url=url)


@router.get("/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: str = Query(...),
    state: str = Query(""),
    session: AsyncSession = Depends(get_session),
):
    try:
        prov = Provider(provider)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    client = ProviderRegistry.get_client(prov)
    token_data = await client.exchange_code(code)

    user_id = uuid.UUID(settings.default_user_id)

    expires_at = None
    if "expires_at" in token_data:
        expires_at = datetime.fromtimestamp(token_data["expires_at"], tz=timezone.utc)
    elif "expires_in" in token_data:
        expires_at = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + token_data["expires_in"],
            tz=timezone.utc,
        )

    await upsert_token(
        session=session,
        user_id=user_id,
        provider=prov,
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        expires_at=expires_at,
        scopes=token_data.get("scope"),
    )

    return {"status": "connected", "provider": provider}
