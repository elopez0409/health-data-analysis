import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from app.models.raw import RawWhoopRecovery, RawWhoopSleep, RawWhoopWorkout
from app.providers.whoop.client import WHOOP_API_BASE, WhoopClient
from app.providers.whoop.models import WhoopRecovery, WhoopSleep, WhoopWorkout
from app.providers.whoop.normalizer import (
    KJ_TO_KCAL,
    normalize_whoop_recovery,
    normalize_whoop_sleep,
    normalize_whoop_workout,
)
from app.schemas.common import Provider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def whoop_recovery_payload() -> dict:
    return json.loads((FIXTURES / "whoop_recovery.json").read_text())


@pytest.fixture
def whoop_sleep_payload() -> dict:
    return json.loads((FIXTURES / "whoop_sleep.json").read_text())


@pytest.fixture
def whoop_workout_payload() -> dict:
    return json.loads((FIXTURES / "whoop_workout.json").read_text())


@pytest.fixture
def whoop_profile_payload() -> dict:
    return {
        "user_id": 10129,
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
    }


class TestWhoopClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_success(self, whoop_profile_payload):
        respx.get(f"{WHOOP_API_BASE}/v1/user/profile/basic").mock(
            return_value=Response(200, json=whoop_profile_payload)
        )

        client = WhoopClient(access_token="test-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is True
        assert status.provider == Provider.WHOOP
        assert status.username == "Test User"
        assert status.latency_ms is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self):
        respx.get(f"{WHOOP_API_BASE}/v1/user/profile/basic").mock(
            return_value=Response(401, json={"error": "Unauthorized"})
        )

        client = WhoopClient(access_token="bad-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is False
        assert status.error is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_pull_all_resources(
        self,
        whoop_recovery_payload,
        whoop_sleep_payload,
        whoop_workout_payload,
    ):
        respx.get(f"{WHOOP_API_BASE}/v1/recovery").mock(
            return_value=Response(
                200,
                json={"records": [whoop_recovery_payload], "next_token": None},
            )
        )
        respx.get(f"{WHOOP_API_BASE}/v1/activity/sleep").mock(
            return_value=Response(
                200,
                json={"records": [whoop_sleep_payload], "next_token": None},
            )
        )
        respx.get(f"{WHOOP_API_BASE}/v1/activity/workout").mock(
            return_value=Response(
                200,
                json={"records": [whoop_workout_payload], "next_token": None},
            )
        )

        client = WhoopClient(access_token="test-token")
        records = await client.pull(uuid.uuid4())

        assert len(records) == 3
        external_ids = {r.external_id for r in records}
        assert "93845627" in external_ids
        assert "48271033" in external_ids
        assert "71923456" in external_ids

    @respx.mock
    @pytest.mark.asyncio
    async def test_pull_pagination(self, whoop_recovery_payload):
        route = respx.get(f"{WHOOP_API_BASE}/v1/recovery")
        route.side_effect = [
            Response(
                200,
                json={
                    "records": [whoop_recovery_payload],
                    "next_token": "page2token",
                },
            ),
            Response(
                200,
                json={"records": [whoop_recovery_payload], "next_token": None},
            ),
        ]
        respx.get(f"{WHOOP_API_BASE}/v1/activity/sleep").mock(
            return_value=Response(200, json={"records": [], "next_token": None})
        )
        respx.get(f"{WHOOP_API_BASE}/v1/activity/workout").mock(
            return_value=Response(200, json={"records": [], "next_token": None})
        )

        client = WhoopClient(access_token="test-token")
        records = await client.pull(uuid.uuid4())

        assert len(records) == 2


class TestWhoopNormalizers:
    def _make_raw_recovery(self, payload: dict) -> RawWhoopRecovery:
        return RawWhoopRecovery(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="whoop",
            external_id=str(payload["cycle_id"]),
            fetched_at=datetime.now(timezone.utc),
            payload=payload,
            payload_hash="abc123",
        )

    def _make_raw_sleep(self, payload: dict) -> RawWhoopSleep:
        return RawWhoopSleep(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="whoop",
            external_id=str(payload["id"]),
            fetched_at=datetime.now(timezone.utc),
            payload=payload,
            payload_hash="abc123",
        )

    def _make_raw_workout(self, payload: dict) -> RawWhoopWorkout:
        return RawWhoopWorkout(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="whoop",
            external_id=str(payload["id"]),
            fetched_at=datetime.now(timezone.utc),
            payload=payload,
            payload_hash="abc123",
        )

    def test_normalize_recovery(self, whoop_recovery_payload):
        raw = self._make_raw_recovery(whoop_recovery_payload)
        unified = normalize_whoop_recovery(raw)

        assert unified.source == "whoop"
        assert unified.recovery_score == 82.0
        assert unified.hrv_rmssd == pytest.approx(0.068432)
        assert unified.resting_heart_rate == 52.0
        assert unified.confidence == 1.0
        assert unified.metric_date == datetime(2026, 5, 25).date()
        assert unified.source_record_id == raw.id

    def test_normalize_sleep(self, whoop_sleep_payload):
        raw = self._make_raw_sleep(whoop_sleep_payload)
        unified = normalize_whoop_sleep(raw)

        assert unified.source == "whoop"
        assert unified.light_seconds == pytest.approx(10800.0)
        assert unified.deep_seconds == pytest.approx(5400.0)
        assert unified.rem_seconds == pytest.approx(6300.0)
        assert unified.awake_seconds == pytest.approx(2700.0)
        assert unified.total_seconds == pytest.approx(22500.0)
        assert unified.sleep_score == 88.0
        assert unified.bedtime == datetime(2026, 5, 24, 22, 45, tzinfo=timezone.utc)
        assert unified.wake_time == datetime(2026, 5, 25, 6, 30, tzinfo=timezone.utc)
        assert unified.confidence == 1.0

    def test_normalize_workout(self, whoop_workout_payload):
        raw = self._make_raw_workout(whoop_workout_payload)
        unified = normalize_whoop_workout(raw)

        assert unified.source == "whoop"
        assert unified.activity_type == "whoop_sport_0"
        assert unified.calories == pytest.approx(2850.5 * KJ_TO_KCAL, rel=1e-3)
        assert unified.distance_meters == 9750.0
        assert unified.avg_heart_rate_bpm == 152.0
        assert unified.max_heart_rate_bpm == 178.0
        assert unified.elevation_gain_meters == 85.3
        assert unified.duration_seconds == pytest.approx(3900.0)
        assert unified.started_at == datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)
        assert unified.ended_at == datetime(2026, 5, 25, 15, 5, tzinfo=timezone.utc)
        assert unified.confidence == 1.0
