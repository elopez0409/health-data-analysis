import time
import uuid
from datetime import date, datetime, timedelta, timezone

import httpx

from app.logging import get_logger
from app.providers.base import ProviderClient, RawRecord
from app.providers.oura import auth
from app.providers.oura.models import OuraPersonalInfo
from app.providers.registry import ProviderRegistry
from app.schemas.common import ConnectionStatus, Provider

logger = get_logger(__name__)

OURA_API_BASE = "https://api.ouraring.com/v2/usercollection"


@ProviderRegistry.register(Provider.OURA)
class OuraClient(ProviderClient):
    provider_name = Provider.OURA

    def __init__(self, access_token: str | None = None):
        self._access_token = access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def verify_connection(self, user_id: uuid.UUID) -> ConnectionStatus:
        start = time.time()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{OURA_API_BASE}/personal_info",
                    headers=self._headers(),
                    timeout=10.0,
                )
                resp.raise_for_status()
                info = OuraPersonalInfo(**resp.json())
                latency = (time.time() - start) * 1000
                return ConnectionStatus(
                    provider=Provider.OURA,
                    connected=True,
                    latency_ms=latency,
                    username=info.email or info.id,
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error("oura_verify_failed", error=str(e))
            return ConnectionStatus(
                provider=Provider.OURA,
                connected=False,
                latency_ms=latency,
                error=str(e),
            )

    async def pull(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> list[RawRecord]:
        records: list[RawRecord] = []
        start_date = since.date() if since else date.today() - timedelta(days=30)

        records.extend(await self._pull_collection("daily_sleep", start_date))
        records.extend(await self._pull_collection("daily_readiness", start_date))
        records.extend(await self._pull_collection("daily_activity", start_date))

        logger.info("oura_pull_complete", count=len(records))
        return records

    async def _pull_collection(
        self, collection: str, start_date: date
    ) -> list[RawRecord]:
        records: list[RawRecord] = []
        params: dict = {"start_date": start_date.isoformat()}
        url: str | None = f"{OURA_API_BASE}/{collection}"

        async with httpx.AsyncClient() as client:
            while url:
                resp = await client.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=30.0,
                )
                resp.raise_for_status()
                body = resp.json()

                for item in body.get("data", []):
                    records.append(
                        RawRecord(
                            external_id=item["id"],
                            payload=item,
                            fetched_at=datetime.now(timezone.utc),
                        )
                    )

                url = body.get("next_token")
                if url:
                    params = {"next_token": url}
                    url = f"{OURA_API_BASE}/{collection}"
                else:
                    url = None

        return records

    async def get_authorize_url(self, state: str) -> str:
        return auth.get_authorize_url(state)

    async def exchange_code(self, code: str) -> dict:
        return await auth.exchange_code(code)

    async def refresh_token(self, refresh_token_value: str) -> dict:
        return await auth.refresh_access_token(refresh_token_value)
