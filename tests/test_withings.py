import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from app.models.raw import (
    RawWithingsBloodPressure,
    RawWithingsSleep,
    RawWithingsWeight,
)
from app.providers.withings.client import WithingsClient
from app.providers.withings.normalizer import (
    normalize_withings_bp,
    normalize_withings_sleep,
    normalize_withings_weight,
)
from app.schemas.common import Provider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def withings_weight_payload() -> dict:
    return json.loads((FIXTURES / "withings_weight.json").read_text())


@pytest.fixture
def withings_sleep_payload() -> dict:
    return json.loads((FIXTURES / "withings_sleep.json").read_text())


class TestWithingsClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_success(self):
        respx.post("https://wbsapi.withings.net/measure").mock(
            return_value=Response(
                200,
                json={
                    "status": 0,
                    "body": {
                        "updatetime": 1779262200,
                        "timezone": "America/Los_Angeles",
                        "measuregrps": [],
                    },
                },
            )
        )

        client = WithingsClient(access_token="test-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is True
        assert status.provider == Provider.WITHINGS
        assert status.latency_ms is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self):
        respx.post("https://wbsapi.withings.net/measure").mock(
            return_value=Response(401, json={"error": "invalid_token"})
        )

        client = WithingsClient(access_token="bad-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is False
        assert status.error is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_api_error_status(self):
        respx.post("https://wbsapi.withings.net/measure").mock(
            return_value=Response(
                200,
                json={"status": 401, "body": {}, "error": "invalid_token"},
            )
        )

        client = WithingsClient(access_token="expired-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is False
        assert "status 401" in status.error


class TestWithingsNormalizer:
    def test_normalize_weight(self, withings_weight_payload):
        raw = RawWithingsWeight(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="withings",
            external_id="2963912345",
            fetched_at=datetime.now(timezone.utc),
            payload=withings_weight_payload,
            payload_hash="abc123",
        )

        unified = normalize_withings_weight(raw)

        assert unified.source == "withings"
        assert unified.source_record_id == raw.id
        assert unified.confidence == 1.0
        assert unified.weight_kg == pytest.approx(78.5)
        assert unified.body_fat_pct == pytest.approx(18.5)
        assert unified.muscle_mass_kg == pytest.approx(35.2)
        assert unified.measured_at == datetime(2026, 5, 20, 7, 30, tzinfo=timezone.utc)
        assert unified.systolic_bp is None
        assert unified.diastolic_bp is None

    def test_normalize_sleep(self, withings_sleep_payload):
        raw = RawWithingsSleep(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="withings",
            external_id="1234567890",
            fetched_at=datetime.now(timezone.utc),
            payload=withings_sleep_payload,
            payload_hash="abc123",
        )

        unified = normalize_withings_sleep(raw)

        assert unified.source == "withings"
        assert unified.source_record_id == raw.id
        assert unified.confidence == 1.0
        assert unified.sleep_date == date(2026, 5, 20)
        assert unified.bedtime == datetime(2026, 5, 19, 23, 15, tzinfo=timezone.utc)
        assert unified.wake_time == datetime(2026, 5, 20, 7, 5, tzinfo=timezone.utc)
        assert unified.total_seconds == pytest.approx(28200.0)
        assert unified.deep_seconds == pytest.approx(5100.0)
        assert unified.light_seconds == pytest.approx(12600.0)
        assert unified.rem_seconds == pytest.approx(7800.0)
        assert unified.awake_seconds == pytest.approx(2700.0)

    def test_normalize_bp(self):
        bp_payload = {
            "grpid": 5551234567,
            "attrib": 0,
            "date": 1779264000,
            "category": 1,
            "measures": [
                {"value": 120, "type": 10, "unit": 0},
                {"value": 80, "type": 9, "unit": 0},
                {"value": 72, "type": 11, "unit": 0},
            ],
        }

        raw = RawWithingsBloodPressure(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="withings",
            external_id="5551234567",
            fetched_at=datetime.now(timezone.utc),
            payload=bp_payload,
            payload_hash="abc123",
        )

        unified = normalize_withings_bp(raw)

        assert unified.source == "withings"
        assert unified.source_record_id == raw.id
        assert unified.systolic_bp == pytest.approx(120.0)
        assert unified.diastolic_bp == pytest.approx(80.0)
        assert unified.measured_at == datetime(2026, 5, 20, 8, 0, tzinfo=timezone.utc)
        assert unified.weight_kg is None
        assert unified.body_fat_pct is None
