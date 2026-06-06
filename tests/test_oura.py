import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from app.models.raw import RawOuraDailyActivity, RawOuraDailyReadiness, RawOuraDailySleep
from app.providers.oura.client import OuraClient
from app.providers.oura.normalizer import (
    normalize_oura_activity,
    normalize_oura_readiness,
    normalize_oura_sleep,
)
from app.schemas.common import Provider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def oura_sleep_payload() -> dict:
    return json.loads((FIXTURES / "oura_daily_sleep.json").read_text())


@pytest.fixture
def oura_readiness_payload() -> dict:
    return json.loads((FIXTURES / "oura_daily_readiness.json").read_text())


class TestOuraClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_success(self):
        respx.get("https://api.ouraring.com/v2/usercollection/personal_info").mock(
            return_value=Response(
                200,
                json={
                    "id": "user-abc-123",
                    "age": 32,
                    "email": "test@example.com",
                },
            )
        )

        client = OuraClient(access_token="test-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is True
        assert status.provider == Provider.OURA
        assert status.username == "test@example.com"
        assert status.latency_ms is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self):
        respx.get("https://api.ouraring.com/v2/usercollection/personal_info").mock(
            return_value=Response(401, json={"detail": "Unauthorized"})
        )

        client = OuraClient(access_token="bad-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is False
        assert status.error is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_pull_daily_sleep(self, oura_sleep_payload):
        respx.get(
            "https://api.ouraring.com/v2/usercollection/daily_sleep"
        ).mock(
            return_value=Response(
                200, json={"data": [oura_sleep_payload], "next_token": None}
            )
        )
        respx.get(
            "https://api.ouraring.com/v2/usercollection/daily_readiness"
        ).mock(return_value=Response(200, json={"data": [], "next_token": None}))
        respx.get(
            "https://api.ouraring.com/v2/usercollection/daily_activity"
        ).mock(return_value=Response(200, json={"data": [], "next_token": None}))

        client = OuraClient(access_token="test-token")
        records = await client.pull(uuid.uuid4())

        assert len(records) == 1
        assert records[0].external_id == "a1b2c3d4-sleep-0520"
        assert records[0].payload["score"] == 82


class TestOuraNormalizer:
    def test_normalize_sleep(self, oura_sleep_payload):
        raw = RawOuraDailySleep(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="oura",
            external_id="a1b2c3d4-sleep-0520",
            fetched_at=datetime.now(timezone.utc),
            payload=oura_sleep_payload,
            payload_hash="abc123",
        )

        unified = normalize_oura_sleep(raw)

        assert unified.sleep_date.isoformat() == "2026-05-20"
        assert unified.sleep_score == 82.0
        assert unified.total_seconds == 27540.0
        assert unified.rem_seconds == 6840.0
        assert unified.deep_seconds == 5280.0
        assert unified.light_seconds == 15420.0
        assert unified.awake_seconds == 2160.0
        assert unified.bedtime is not None
        assert unified.wake_time is not None
        assert unified.source == "oura"
        assert unified.confidence == 1.0

    def test_normalize_readiness(self, oura_readiness_payload):
        raw = RawOuraDailyReadiness(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="oura",
            external_id="a1b2c3d4-readiness-0520",
            fetched_at=datetime.now(timezone.utc),
            payload=oura_readiness_payload,
            payload_hash="abc123",
        )

        unified = normalize_oura_readiness(raw)

        assert unified.metric_date.isoformat() == "2026-05-20"
        assert unified.readiness_score == 78.0
        assert unified.hrv_rmssd == 42.0
        assert unified.source == "oura"
        assert unified.confidence == 1.0

    def test_normalize_activity(self):
        payload = {
            "id": "a1b2c3d4-activity-0520",
            "day": "2026-05-20",
            "score": 85,
            "timestamp": "2026-05-20T04:00:00+00:00",
            "active_calories": 480,
            "total_calories": 2250,
            "steps": 9870,
            "equivalent_walking_distance": 7650,
            "high_activity_time": 1800,
            "medium_activity_time": 3600,
            "low_activity_time": 7200,
            "sedentary_time": 28800,
            "resting_time": 14400,
        }

        raw = RawOuraDailyActivity(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="oura",
            external_id="a1b2c3d4-activity-0520",
            fetched_at=datetime.now(timezone.utc),
            payload=payload,
            payload_hash="abc123",
        )

        unified = normalize_oura_activity(raw)

        assert unified.metric_date.isoformat() == "2026-05-20"
        assert unified.steps == 9870
        assert unified.calories_total == 2250.0
        assert unified.calories_active == 480.0
        assert unified.source == "oura"
        assert unified.confidence == 1.0
