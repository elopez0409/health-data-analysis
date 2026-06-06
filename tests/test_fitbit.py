import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from app.models.raw import RawFitbitActivity, RawFitbitSleep
from app.providers.fitbit.client import FitbitClient
from app.providers.fitbit.normalizer import (
    normalize_fitbit_activity,
    normalize_fitbit_sleep,
)
from app.schemas.common import Provider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fitbit_sleep_payload() -> dict:
    return json.loads((FIXTURES / "fitbit_sleep.json").read_text())


@pytest.fixture
def fitbit_activity_payload() -> dict:
    return json.loads((FIXTURES / "fitbit_activity_summary.json").read_text())


@pytest.fixture
def fitbit_profile_payload() -> dict:
    return json.loads((FIXTURES / "fitbit_profile.json").read_text())


class TestFitbitClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_success(self, fitbit_profile_payload):
        respx.get("https://api.fitbit.com/1/user/-/profile.json").mock(
            return_value=Response(200, json=fitbit_profile_payload)
        )

        client = FitbitClient(access_token="test-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is True
        assert status.provider == Provider.FITBIT
        assert status.username == "Test User"

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self):
        respx.get("https://api.fitbit.com/1/user/-/profile.json").mock(
            return_value=Response(401, json={"errors": [{"errorType": "expired_token"}]})
        )

        client = FitbitClient(access_token="bad-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is False
        assert status.error is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_pull_sleep(self, fitbit_sleep_payload):
        respx.get(url__regex=r".*sleep/date.*").mock(
            return_value=Response(200, json={"sleep": [fitbit_sleep_payload]})
        )
        respx.get(url__regex=r".*activities/date.*").mock(
            return_value=Response(200, json={"summary": {"steps": 0}})
        )

        client = FitbitClient(access_token="test-token")
        records = await client.pull(uuid.uuid4())

        sleep_records = [r for r in records if not r.external_id.startswith("activity-")]
        assert len(sleep_records) == 1
        assert sleep_records[0].external_id == "44058888123"


class TestFitbitNormalizer:
    def test_normalize_sleep(self, fitbit_sleep_payload):
        raw = RawFitbitSleep(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="fitbit",
            external_id="44058888123",
            fetched_at=datetime.now(timezone.utc),
            payload=fitbit_sleep_payload,
            payload_hash="abc123",
        )

        unified = normalize_fitbit_sleep(raw)

        assert unified.sleep_date.isoformat() == "2026-05-20"
        assert unified.total_seconds == 28200.0
        assert unified.deep_seconds == 85 * 60.0
        assert unified.light_seconds == 210 * 60.0
        assert unified.rem_seconds == 130 * 60.0
        assert unified.awake_seconds == 45 * 60.0
        assert unified.sleep_score == 88.0
        assert unified.source == "fitbit"
        assert unified.confidence == 1.0

    def test_normalize_activity(self, fitbit_activity_payload):
        raw = RawFitbitActivity(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="fitbit",
            external_id="activity-2026-05-20",
            fetched_at=datetime.now(timezone.utc),
            payload=fitbit_activity_payload,
            payload_hash="abc123",
        )

        unified = normalize_fitbit_activity(raw)

        assert unified.metric_date.isoformat() == "2026-05-20"
        assert unified.steps == 12450
        assert unified.calories_total == 2350.0
        assert unified.calories_active == 980.0
        assert unified.resting_heart_rate == 58.0
        assert unified.source == "fitbit"
