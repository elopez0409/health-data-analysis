import time
import uuid
from datetime import date, datetime, timedelta, timezone

import httpx

from app.logging import get_logger
from app.providers.base import ProviderClient, RawRecord
from app.providers.fitbit import auth
from app.providers.fitbit.models import FitbitProfile
from app.providers.registry import ProviderRegistry
from app.schemas.common import ConnectionStatus, Provider

logger = get_logger(__name__)

FITBIT_API_BASE = "https://api.fitbit.com"


@ProviderRegistry.register(Provider.FITBIT)
class FitbitClient(ProviderClient):
    provider_name = Provider.FITBIT

    def __init__(self, access_token: str | None = None):
        self._access_token = access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def verify_connection(self, user_id: uuid.UUID) -> ConnectionStatus:
        start = time.time()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{FITBIT_API_BASE}/1/user/-/profile.json",
                    headers=self._headers(),
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                profile = FitbitProfile(**data.get("user", {}))
                latency = (time.time() - start) * 1000
                return ConnectionStatus(
                    provider=Provider.FITBIT,
                    connected=True,
                    latency_ms=latency,
                    username=profile.displayName or profile.encodedId,
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error("fitbit_verify_failed", error=str(e))
            return ConnectionStatus(
                provider=Provider.FITBIT,
                connected=False,
                latency_ms=latency,
                error=str(e),
            )

    async def pull(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> list[RawRecord]:
        """Pull sleep logs and activity summaries."""
        records = []
        records.extend(await self._pull_sleep(since))
        records.extend(await self._pull_activity_summary(since))
        return records

    async def _pull_sleep(self, since: datetime | None = None) -> list[RawRecord]:
        records = []
        start_date = since.date() if since else date.today() - timedelta(days=30)
        end_date = date.today()

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{FITBIT_API_BASE}/1.2/user/-/sleep/date/{start_date}/{end_date}.json",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            for sleep_log in data.get("sleep", []):
                records.append(
                    RawRecord(
                        external_id=str(sleep_log["logId"]),
                        payload=sleep_log,
                        fetched_at=datetime.now(timezone.utc),
                    )
                )

        logger.info("fitbit_sleep_pull", count=len(records))
        return records

    async def _pull_activity_summary(
        self, since: datetime | None = None
    ) -> list[RawRecord]:
        records = []
        start_date = since.date() if since else date.today() - timedelta(days=30)
        end_date = date.today()

        async with httpx.AsyncClient() as client:
            current = start_date
            while current <= end_date:
                resp = await client.get(
                    f"{FITBIT_API_BASE}/1/user/-/activities/date/{current}.json",
                    headers=self._headers(),
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()

                records.append(
                    RawRecord(
                        external_id=f"activity-{current.isoformat()}",
                        payload=data,
                        fetched_at=datetime.now(timezone.utc),
                    )
                )
                current += timedelta(days=1)

        logger.info("fitbit_activity_pull", count=len(records))
        return records

    async def pull_heart_rate_intraday(
        self, target_date: date
    ) -> list[RawRecord]:
        """Pull intraday HR for a specific date (1-minute resolution)."""
        records = []
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{FITBIT_API_BASE}/1/user/-/activities/heart/date/{target_date}/1d/1min.json",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            intraday = data.get("activities-heart-intraday", {}).get("dataset", [])
            for i, entry in enumerate(intraday):
                records.append(
                    RawRecord(
                        external_id=f"hr-{target_date}-{i}",
                        payload={
                            "date": target_date.isoformat(),
                            "time": entry["time"],
                            "value": entry["value"],
                        },
                        fetched_at=datetime.now(timezone.utc),
                    )
                )

        return records

    async def get_authorize_url(self, state: str) -> str:
        return auth.get_authorize_url(state)

    async def exchange_code(self, code: str) -> dict:
        return await auth.exchange_code(code)

    async def refresh_token(self, refresh_token_value: str) -> dict:
        return await auth.refresh_access_token(refresh_token_value)
