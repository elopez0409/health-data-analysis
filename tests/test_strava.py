import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from app.models.raw import RawStravaActivity
from app.providers.strava.client import StravaClient
from app.providers.strava.models import StravaActivity
from app.providers.strava.normalizer import normalize_strava_activity
from app.schemas.common import Provider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def strava_activity_payload() -> dict:
    return json.loads((FIXTURES / "strava_activity.json").read_text())


@pytest.fixture
def strava_athlete_payload() -> dict:
    return json.loads((FIXTURES / "strava_athlete.json").read_text())


class TestStravaClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_success(self, strava_athlete_payload):
        respx.get("https://www.strava.com/api/v3/athlete").mock(
            return_value=Response(200, json=strava_athlete_payload)
        )

        client = StravaClient(access_token="test-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is True
        assert status.provider == Provider.STRAVA
        assert status.username == "testrunner"
        assert status.latency_ms is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self):
        respx.get("https://www.strava.com/api/v3/athlete").mock(
            return_value=Response(401, json={"message": "Authorization Error"})
        )

        client = StravaClient(access_token="bad-token")
        status = await client.verify_connection(uuid.uuid4())

        assert status.connected is False
        assert status.error is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_pull_activities(self, strava_activity_payload):
        respx.get("https://www.strava.com/api/v3/athlete/activities").mock(
            return_value=Response(200, json=[strava_activity_payload])
        )

        client = StravaClient(access_token="test-token")
        records = await client.pull(uuid.uuid4())

        assert len(records) == 1
        assert records[0].external_id == "12345678901"
        assert records[0].payload["name"] == "Morning Run"

    @respx.mock
    @pytest.mark.asyncio
    async def test_pull_pagination(self, strava_activity_payload):
        page1 = [strava_activity_payload] * 100
        page2 = [strava_activity_payload]

        route = respx.get("https://www.strava.com/api/v3/athlete/activities")
        route.side_effect = [
            Response(200, json=page1),
            Response(200, json=page2),
        ]

        client = StravaClient(access_token="test-token")
        records = await client.pull(uuid.uuid4())

        assert len(records) == 101


class TestStravaNormalizer:
    def test_normalize_activity(self, strava_activity_payload):
        raw = RawStravaActivity(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="strava",
            external_id="12345678901",
            fetched_at=datetime.now(timezone.utc),
            payload=strava_activity_payload,
            payload_hash="abc123",
        )

        unified = normalize_strava_activity(raw)

        assert unified.activity_type == "TrailRun"
        assert unified.title == "Morning Run"
        assert unified.distance_meters == 10500.5
        assert unified.duration_seconds == 3600.0
        assert unified.elevation_gain_meters == 150.3
        assert unified.avg_heart_rate_bpm == 145.2
        assert unified.max_heart_rate_bpm == 172.0
        assert unified.calories == 720.0
        assert unified.confidence == 1.0
        assert unified.source == "strava"
        assert unified.source_record_id == raw.id

    def test_normalize_activity_with_kilojoules(self, strava_activity_payload):
        strava_activity_payload["calories"] = None
        strava_activity_payload["kilojoules"] = 850.0
        strava_activity_payload["type"] = "Ride"
        strava_activity_payload["sport_type"] = "Ride"

        raw = RawStravaActivity(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="strava",
            external_id="12345678901",
            fetched_at=datetime.now(timezone.utc),
            payload=strava_activity_payload,
            payload_hash="abc123",
        )

        unified = normalize_strava_activity(raw)

        assert unified.calories == 850.0
        assert unified.activity_type == "Ride"
