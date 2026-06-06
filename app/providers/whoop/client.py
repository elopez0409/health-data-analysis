import time
import uuid
from datetime import datetime, timezone

import httpx

from app.logging import get_logger
from app.providers.base import ProviderClient, RawRecord
from app.providers.registry import ProviderRegistry
from app.providers.whoop import auth
from app.providers.whoop.models import (
    WhoopPaginatedResponse,
    WhoopUserProfile,
)
from app.schemas.common import ConnectionStatus, Provider

logger = get_logger(__name__)

WHOOP_API_BASE = "https://api.prod.whoop.com/developer"


@ProviderRegistry.register(Provider.WHOOP)
class WhoopClient(ProviderClient):
    provider_name = Provider.WHOOP

    def __init__(self, access_token: str | None = None):
        self._access_token = access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def verify_connection(self, user_id: uuid.UUID) -> ConnectionStatus:
        start = time.time()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{WHOOP_API_BASE}/v1/user/profile/basic",
                    headers=self._headers(),
                    timeout=10.0,
                )
                resp.raise_for_status()
                profile = WhoopUserProfile(**resp.json())
                latency = (time.time() - start) * 1000
                return ConnectionStatus(
                    provider=Provider.WHOOP,
                    connected=True,
                    latency_ms=latency,
                    username=(
                        f"{profile.first_name} {profile.last_name}"
                        if profile.first_name
                        else profile.email
                    ),
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error("whoop_verify_failed", error=str(e))
            return ConnectionStatus(
                provider=Provider.WHOOP,
                connected=False,
                latency_ms=latency,
                error=str(e),
            )

    async def _paginated_fetch(
        self,
        endpoint: str,
        resource_label: str,
        params: dict | None = None,
    ) -> list[dict]:
        all_records: list[dict] = []
        request_params = dict(params or {})

        async with httpx.AsyncClient() as client:
            while True:
                resp = await client.get(
                    f"{WHOOP_API_BASE}{endpoint}",
                    headers=self._headers(),
                    params=request_params,
                    timeout=30.0,
                )
                resp.raise_for_status()
                page = WhoopPaginatedResponse(**resp.json())
                all_records.extend(page.records)

                if not page.next_token:
                    break
                request_params["nextToken"] = page.next_token

        logger.info(f"whoop_{resource_label}_fetched", count=len(all_records))
        return all_records

    async def pull(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> list[RawRecord]:
        records: list[RawRecord] = []
        now = datetime.now(timezone.utc)

        endpoints = [
            ("/v1/recovery", "recovery", "cycle_id"),
            ("/v1/activity/sleep", "sleep", "id"),
            ("/v1/activity/workout", "workout", "id"),
        ]

        for endpoint, label, id_field in endpoints:
            raw_items = await self._paginated_fetch(endpoint, label)
            for item in raw_items:
                records.append(
                    RawRecord(
                        external_id=str(item[id_field]),
                        payload=item,
                        fetched_at=now,
                    )
                )

        logger.info("whoop_pull_complete", count=len(records))
        return records

    async def get_authorize_url(self, state: str) -> str:
        return auth.get_authorize_url(state)

    async def exchange_code(self, code: str) -> dict:
        return await auth.exchange_code(code)

    async def refresh_token(self, refresh_token_value: str) -> dict:
        return await auth.refresh_access_token(refresh_token_value)
