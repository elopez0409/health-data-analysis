import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from app.models.raw import RawGarminActivity, RawGarminSleep
from app.providers.garmin.client import GarminClient
from app.providers.garmin.normalizer import (
    normalize_garmin_activity,
    normalize_garmin_sleep,
)
from app.schemas.common import Provider
from app.webhooks.garmin import _verify_signature

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def garmin_activity_payload() -> dict:
    return json.loads((FIXTURES / "garmin_activity_push.json").read_text())


@pytest.fixture
def garmin_sleep_payload() -> dict:
    return json.loads((FIXTURES / "garmin_sleep_push.json").read_text())


class TestGarminClient:
    @pytest.mark.asyncio
    async def test_verify_connection_with_token(self):
        client = GarminClient(access_token="test-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is True
        assert status.provider == Provider.GARMIN
        assert status.error is None

    @pytest.mark.asyncio
    async def test_verify_connection_without_token(self):
        client = GarminClient(access_token=None)
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is False
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_pull_returns_empty(self):
        client = GarminClient(access_token="test-token")
        records = await client.pull(uuid.uuid4())

        assert records == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_authorize_url(self, monkeypatch):
        monkeypatch.setenv("GARMIN_CONSUMER_KEY", "test-key")
        monkeypatch.setenv("GARMIN_CONSUMER_SECRET", "test-secret")

        from app.config import settings as _settings
        monkeypatch.setattr(_settings, "garmin_consumer_key", "test-key")
        monkeypatch.setattr(_settings, "garmin_consumer_secret", "test-secret")

        respx.post(
            "https://connectapi.garmin.com/oauth-service/oauth/request_token"
        ).mock(
            return_value=Response(
                200, text="oauth_token=req_tok&oauth_token_secret=req_sec"
            )
        )

        client = GarminClient()
        url = await client.get_authorize_url(state="test-state")

        assert "oauthConfirm" in url
        assert "oauth_token=req_tok" in url


class TestGarminActivityNormalizer:
    def test_normalize_activity(self, garmin_activity_payload):
        activity_data = garmin_activity_payload["activities"][0]

        raw = RawGarminActivity(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="garmin",
            external_id=str(activity_data["activityId"]),
            fetched_at=datetime.now(timezone.utc),
            payload=activity_data,
            payload_hash="abc123",
        )

        unified = normalize_garmin_activity(raw)

        assert unified.activity_type == "RUNNING"
        assert unified.duration_seconds == 2820.0
        assert unified.distance_meters == 7542.3
        assert unified.calories == 485.0
        assert unified.avg_heart_rate_bpm == 152.0
        assert unified.max_heart_rate_bpm == 178.0
        assert unified.confidence == 1.0
        assert unified.source == "garmin"
        assert unified.source_record_id == raw.id

        expected_start = datetime(2025, 5, 20, 9, 50, 0, tzinfo=timezone.utc)
        assert unified.started_at == expected_start

    def test_normalize_activity_no_hr(self, garmin_activity_payload):
        activity_data = garmin_activity_payload["activities"][0].copy()
        del activity_data["averageHeartRateInBeatsPerMinute"]
        del activity_data["maxHeartRateInBeatsPerMinute"]

        raw = RawGarminActivity(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="garmin",
            external_id=str(activity_data["activityId"]),
            fetched_at=datetime.now(timezone.utc),
            payload=activity_data,
            payload_hash="abc123",
        )

        unified = normalize_garmin_activity(raw)

        assert unified.avg_heart_rate_bpm is None
        assert unified.max_heart_rate_bpm is None


class TestGarminSleepNormalizer:
    def test_normalize_sleep(self, garmin_sleep_payload):
        sleep_data = garmin_sleep_payload["sleeps"][0]

        raw = RawGarminSleep(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="garmin",
            external_id=str(sleep_data["startTimeInSeconds"]),
            fetched_at=datetime.now(timezone.utc),
            payload=sleep_data,
            payload_hash="abc123",
        )

        unified = normalize_garmin_sleep(raw)

        assert unified.total_seconds == 27900.0
        assert unified.deep_seconds == 5400.0
        assert unified.light_seconds == 12600.0
        assert unified.rem_seconds == 7200.0
        assert unified.awake_seconds == 2700.0
        assert unified.confidence == 1.0
        assert unified.source == "garmin"
        assert unified.source_record_id == raw.id

        expected_bedtime = datetime(2025, 5, 19, 23, 0, 0, tzinfo=timezone.utc)
        assert unified.bedtime == expected_bedtime
        assert unified.sleep_date == expected_bedtime.date()


class TestWebhookSignature:
    def test_valid_signature(self, monkeypatch):
        monkeypatch.setattr(
            "app.webhooks.garmin.settings",
            type("S", (), {"garmin_webhook_secret": "test-secret", "default_user_id": "00000000-0000-0000-0000-000000000001"})(),
        )

        body = b'{"activities": []}'
        sig = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()

        assert _verify_signature(body, sig) is True

    def test_invalid_signature(self, monkeypatch):
        monkeypatch.setattr(
            "app.webhooks.garmin.settings",
            type("S", (), {"garmin_webhook_secret": "test-secret", "default_user_id": "00000000-0000-0000-0000-000000000001"})(),
        )

        body = b'{"activities": []}'
        assert _verify_signature(body, "badsig") is False
