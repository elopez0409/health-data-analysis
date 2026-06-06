import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.db import async_session_factory
from app.models.raw import RawGarminActivity, RawGarminSleep
from app.providers.garmin.models import GarminActivityPush, GarminSleepPush

router = APIRouter(tags=["webhooks"])

_DEFAULT_USER = uuid.UUID(settings.default_user_id)


def _verify_signature(body: bytes, signature: str) -> bool:
    expected = hmac.new(
        settings.garmin_webhook_secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/webhooks/garmin")
async def garmin_webhook(
    request: Request,
    x_garmin_signature: str = Header(..., alias="X-Garmin-Signature"),
):
    body = await request.body()

    if not _verify_signature(body, x_garmin_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(body)
    now = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        if "activities" in payload:
            push = GarminActivityPush(**payload)
            for activity in push.activities:
                raw_payload = activity.model_dump(by_alias=True)
                payload_hash = hashlib.sha256(
                    json.dumps(raw_payload, sort_keys=True).encode()
                ).hexdigest()

                stmt = pg_insert(RawGarminActivity).values(
                    user_id=_DEFAULT_USER,
                    provider="garmin",
                    external_id=str(activity.activity_id),
                    fetched_at=now,
                    payload=raw_payload,
                    payload_hash=payload_hash,
                ).on_conflict_do_nothing(
                    constraint="uq_raw_garmin_activity",
                )
                await session.execute(stmt)

        if "sleeps" in payload:
            push = GarminSleepPush(**payload)
            for sleep in push.sleeps:
                raw_payload = sleep.model_dump(by_alias=True)
                payload_hash = hashlib.sha256(
                    json.dumps(raw_payload, sort_keys=True).encode()
                ).hexdigest()

                external_id = str(sleep.start_time_in_seconds)

                stmt = pg_insert(RawGarminSleep).values(
                    user_id=_DEFAULT_USER,
                    provider="garmin",
                    external_id=external_id,
                    fetched_at=now,
                    payload=raw_payload,
                    payload_hash=payload_hash,
                ).on_conflict_do_nothing(
                    constraint="uq_raw_garmin_sleep",
                )
                await session.execute(stmt)

        await session.commit()

    return {"status": "ok"}
