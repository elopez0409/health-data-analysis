import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from app.logging import get_logger
from app.providers.base import ProviderClient, RawRecord
from app.providers.registry import ProviderRegistry
from app.providers.withings import auth
from app.providers.withings.models import (
    WithingsMeasureResponse,
    WithingsSleepResponse,
)
from app.schemas.common import ConnectionStatus, Provider

logger = get_logger(__name__)

WITHINGS_API_BASE = "https://wbsapi.withings.net"


@ProviderRegistry.register(Provider.WITHINGS)
class WithingsClient(ProviderClient):
    provider_name = Provider.WITHINGS

    def __init__(self, access_token: str | None = None):
        self._access_token = access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def verify_connection(self, user_id: uuid.UUID) -> ConnectionStatus:
        start = time.time()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{WITHINGS_API_BASE}/measure",
                    headers=self._headers(),
                    data={
                        "action": "getmeas",
                        "meastype": "1",
                        "category": "1",
                        "limit": "1",
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != 0:
                    raise ValueError(
                        f"Withings API error: status {data.get('status')}"
                    )
                latency = (time.time() - start) * 1000
                return ConnectionStatus(
                    provider=Provider.WITHINGS,
                    connected=True,
                    latency_ms=latency,
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error("withings_verify_failed", error=str(e))
            return ConnectionStatus(
                provider=Provider.WITHINGS,
                connected=False,
                latency_ms=latency,
                error=str(e),
            )

    async def pull(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> list[RawRecord]:
        records: list[RawRecord] = []
        records.extend(await self._pull_body_measurements(since))
        records.extend(await self._pull_blood_pressure(since))
        records.extend(await self._pull_sleep(since))
        logger.info("withings_pull_complete", count=len(records))
        return records

    async def _pull_body_measurements(
        self, since: datetime | None = None
    ) -> list[RawRecord]:
        data: dict[str, str] = {
            "action": "getmeas",
            "meastype": "1,6,76",
            "category": "1",
        }
        if since:
            data["startdate"] = str(int(since.timestamp()))
        data["enddate"] = str(int(datetime.now(timezone.utc).timestamp()))

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{WITHINGS_API_BASE}/measure",
                headers=self._headers(),
                data=data,
                timeout=30.0,
            )
            resp.raise_for_status()
            api_resp = WithingsMeasureResponse(**resp.json())

        records = []
        for grp in api_resp.body.measuregrps:
            records.append(
                RawRecord(
                    external_id=str(grp.grpid),
                    payload=grp.model_dump(),
                    fetched_at=datetime.now(timezone.utc),
                )
            )
        logger.info("withings_body_pull", count=len(records))
        return records

    async def _pull_blood_pressure(
        self, since: datetime | None = None
    ) -> list[RawRecord]:
        data: dict[str, str] = {
            "action": "getmeas",
            "meastype": "9,10,11",
            "category": "1",
        }
        if since:
            data["startdate"] = str(int(since.timestamp()))
        data["enddate"] = str(int(datetime.now(timezone.utc).timestamp()))

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{WITHINGS_API_BASE}/measure",
                headers=self._headers(),
                data=data,
                timeout=30.0,
            )
            resp.raise_for_status()
            api_resp = WithingsMeasureResponse(**resp.json())

        records = []
        for grp in api_resp.body.measuregrps:
            records.append(
                RawRecord(
                    external_id=str(grp.grpid),
                    payload=grp.model_dump(),
                    fetched_at=datetime.now(timezone.utc),
                )
            )
        logger.info("withings_bp_pull", count=len(records))
        return records

    async def _pull_sleep(
        self, since: datetime | None = None
    ) -> list[RawRecord]:
        start = since or (datetime.now(timezone.utc) - timedelta(days=30))
        data = {
            "action": "get",
            "startdate": str(int(start.timestamp())),
            "enddate": str(int(datetime.now(timezone.utc).timestamp())),
            "data_fields": (
                "deepsleepduration,lightsleepduration,remsleepduration,"
                "wakeupduration,hr_average,hr_min,hr_max"
            ),
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{WITHINGS_API_BASE}/v2/sleep",
                headers=self._headers(),
                data=data,
                timeout=30.0,
            )
            resp.raise_for_status()
            api_resp = WithingsSleepResponse(**resp.json())

        records = []
        for series in api_resp.body.series:
            ext_id = str(series.id) if series.id else f"sleep-{series.startdate}"
            records.append(
                RawRecord(
                    external_id=ext_id,
                    payload=series.model_dump(),
                    fetched_at=datetime.now(timezone.utc),
                )
            )
        logger.info("withings_sleep_pull", count=len(records))
        return records

    async def get_authorize_url(self, state: str) -> str:
        return auth.get_authorize_url(state)

    async def exchange_code(self, code: str) -> dict:
        return await auth.exchange_code(code)

    async def refresh_token(self, refresh_token_value: str) -> dict:
        return await auth.refresh_access_token(refresh_token_value)
