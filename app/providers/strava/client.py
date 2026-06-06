import time
import uuid
from datetime import datetime, timezone

import httpx

from app.logging import get_logger
from app.providers.base import ProviderClient, RawRecord
from app.providers.registry import ProviderRegistry
from app.providers.strava import auth
from app.providers.strava.models import StravaActivity, StravaAthlete
from app.schemas.common import ConnectionStatus, Provider

logger = get_logger(__name__)

STRAVA_API_BASE = "https://www.strava.com/api/v3"


@ProviderRegistry.register(Provider.STRAVA)
class StravaClient(ProviderClient):
    provider_name = Provider.STRAVA

    def __init__(self, access_token: str | None = None):
        self._access_token = access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def verify_connection(self, user_id: uuid.UUID) -> ConnectionStatus:
        start = time.time()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{STRAVA_API_BASE}/athlete",
                    headers=self._headers(),
                    timeout=10.0,
                )
                resp.raise_for_status()
                athlete = StravaAthlete(**resp.json())
                latency = (time.time() - start) * 1000
                return ConnectionStatus(
                    provider=Provider.STRAVA,
                    connected=True,
                    latency_ms=latency,
                    username=athlete.username or f"{athlete.firstname} {athlete.lastname}",
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error("strava_verify_failed", error=str(e))
            return ConnectionStatus(
                provider=Provider.STRAVA,
                connected=False,
                latency_ms=latency,
                error=str(e),
            )

    async def pull(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> list[RawRecord]:
        records = []
        page = 1
        per_page = 100

        params: dict = {"per_page": per_page}
        if since:
            params["after"] = int(since.timestamp())

        async with httpx.AsyncClient() as client:
            while True:
                params["page"] = page
                resp = await client.get(
                    f"{STRAVA_API_BASE}/athlete/activities",
                    headers=self._headers(),
                    params=params,
                    timeout=30.0,
                )
                resp.raise_for_status()
                activities = resp.json()

                if not activities:
                    break

                for activity_data in activities:
                    activity = StravaActivity(**activity_data)
                    records.append(
                        RawRecord(
                            external_id=str(activity.id),
                            payload=activity_data,
                            fetched_at=datetime.now(timezone.utc),
                        )
                    )

                if len(activities) < per_page:
                    break
                page += 1

        logger.info("strava_pull_complete", count=len(records))
        return records

    async def get_authorize_url(self, state: str) -> str:
        return auth.get_authorize_url(state)

    async def exchange_code(self, code: str) -> dict:
        return await auth.exchange_code(code)

    async def refresh_token(self, refresh_token_value: str) -> dict:
        return await auth.refresh_access_token(refresh_token_value)
