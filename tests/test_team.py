import uuid
from datetime import date, datetime, timedelta, timezone

import pytest

from app.models.personal import HrAnomaly, PersonalHrState
from app.models.team import Athlete
from app.models.unified import UnifiedDailyMetrics, UnifiedSleep
from app.schemas.team import AthleteCreate, AthleteOut, TeamReadinessResponse
from app.services.readiness import compute_status


# --- Helpers ---


def _make_metrics(
    user_id: uuid.UUID | None = None,
    metric_date: date | None = None,
    recovery_score: float | None = 80,
    readiness_score: float | None = 85,
    hrv_rmssd: float | None = 45.0,
    resting_heart_rate: float | None = 58.0,
    **kwargs,
) -> UnifiedDailyMetrics:
    return UnifiedDailyMetrics(
        id=uuid.uuid4(),
        user_id=user_id or uuid.uuid4(),
        source="oura",
        source_record_id=uuid.uuid4(),
        metric_date=metric_date or date.today(),
        recovery_score=recovery_score,
        readiness_score=readiness_score,
        hrv_rmssd=hrv_rmssd,
        resting_heart_rate=resting_heart_rate,
        **kwargs,
    )


def _make_sleep(
    user_id: uuid.UUID | None = None,
    sleep_date: date | None = None,
    total_seconds: float | None = 8 * 3600,
    sleep_score: float | None = 85,
) -> UnifiedSleep:
    return UnifiedSleep(
        id=uuid.uuid4(),
        user_id=user_id or uuid.uuid4(),
        source="oura",
        source_record_id=uuid.uuid4(),
        sleep_date=sleep_date or date.today(),
        total_seconds=total_seconds,
        sleep_score=sleep_score,
    )


def _make_athlete(name: str = "Test Athlete", **kwargs) -> Athlete:
    return Athlete(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        name=name,
        **kwargs,
    )


# --- compute_status tests ---


class TestComputeStatus:
    def test_no_data(self):
        status, reasons = compute_status(None, None, 0, None)
        assert status == "no_data"
        assert "no metrics" in reasons[0]

    def test_green_good_metrics(self):
        m = _make_metrics(recovery_score=80, readiness_score=85)
        s = _make_sleep(total_seconds=8 * 3600)
        status, reasons = compute_status(m, s, 0, None)
        assert status == "green"
        assert reasons == []

    def test_red_low_recovery(self):
        m = _make_metrics(recovery_score=20)
        s = _make_sleep()
        status, reasons = compute_status(m, s, 0, None)
        assert status == "red"
        assert any("recovery_score" in r and "red" in r for r in reasons)

    def test_red_low_readiness(self):
        m = _make_metrics(recovery_score=80, readiness_score=50)
        s = _make_sleep()
        status, reasons = compute_status(m, s, 0, None)
        assert status == "red"
        assert any("readiness_score" in r and "red" in r for r in reasons)

    def test_red_very_low_sleep(self):
        m = _make_metrics()
        s = _make_sleep(total_seconds=4 * 3600)
        status, reasons = compute_status(m, s, 0, None)
        assert status == "red"
        assert any("sleep" in r and "red" in r for r in reasons)

    def test_yellow_moderate_recovery(self):
        m = _make_metrics(recovery_score=50, readiness_score=85)
        s = _make_sleep()
        status, reasons = compute_status(m, s, 0, None)
        assert status == "yellow"
        assert any("recovery_score" in r and "yellow" in r for r in reasons)

    def test_yellow_moderate_readiness(self):
        m = _make_metrics(recovery_score=80, readiness_score=65)
        s = _make_sleep()
        status, reasons = compute_status(m, s, 0, None)
        assert status == "yellow"
        assert any("readiness_score" in r and "yellow" in r for r in reasons)

    def test_yellow_moderate_sleep(self):
        m = _make_metrics()
        s = _make_sleep(total_seconds=6 * 3600)
        status, reasons = compute_status(m, s, 0, None)
        assert status == "yellow"
        assert any("sleep" in r and "yellow" in r for r in reasons)

    def test_yellow_anomalies(self):
        m = _make_metrics()
        s = _make_sleep()
        status, reasons = compute_status(m, s, 2, None)
        assert status == "yellow"
        assert any("anomal" in r for r in reasons)

    def test_red_stale_metrics(self):
        m = _make_metrics(metric_date=date.today() - timedelta(days=5))
        status, reasons = compute_status(m, None, 0, None)
        assert status == "red"
        assert any("stale" in r for r in reasons)

    def test_multiple_reasons(self):
        m = _make_metrics(recovery_score=30, readiness_score=55)
        s = _make_sleep(total_seconds=4 * 3600)
        status, reasons = compute_status(m, s, 1, None)
        assert status == "red"
        assert len(reasons) >= 3

    def test_metrics_only_no_sleep(self):
        m = _make_metrics()
        status, reasons = compute_status(m, None, 0, None)
        assert status == "green"

    def test_sleep_only_no_metrics(self):
        s = _make_sleep()
        status, reasons = compute_status(None, s, 0, None)
        assert status == "green"

    def test_none_scores_treated_as_ok(self):
        m = _make_metrics(recovery_score=None, readiness_score=None)
        s = _make_sleep(total_seconds=None)
        status, reasons = compute_status(m, s, 0, None)
        assert status == "green"


# --- Schema tests ---


class TestSchemas:
    def test_athlete_create_minimal(self):
        ac = AthleteCreate(name="Jane Doe")
        assert ac.name == "Jane Doe"
        assert ac.sport is None

    def test_athlete_create_full(self):
        ac = AthleteCreate(
            name="Jane Doe",
            sport="Basketball",
            position="PG",
            jersey_number="23",
            catapult_athlete_id="cat-123",
        )
        assert ac.catapult_athlete_id == "cat-123"

    def test_athlete_out_from_model(self):
        athlete = _make_athlete(
            name="Test",
            sport="Soccer",
            position="GK",
            jersey_number="1",
            is_active=True,
        )
        out = AthleteOut.model_validate(athlete)
        assert out.name == "Test"
        assert out.sport == "Soccer"
        assert out.is_active is True

    def test_team_readiness_response_shape(self):
        resp = TeamReadinessResponse(
            team_size=5,
            reporting=3,
            green=2,
            yellow=1,
            red=0,
            no_data=2,
            as_of=datetime.now(timezone.utc),
            athletes=[],
        )
        assert resp.team_size == 5
        assert resp.reporting == 3


# --- Catapult mapping tests ---


class TestCatapultMapping:
    def test_athlete_has_catapult_id(self):
        athlete = _make_athlete(catapult_athlete_id="cat-456")
        assert athlete.catapult_athlete_id == "cat-456"

    def test_athlete_catapult_id_optional(self):
        athlete = _make_athlete()
        assert athlete.catapult_athlete_id is None

    def test_mapping_dict_construction(self):
        athletes = [
            _make_athlete(name="A", catapult_athlete_id="c1"),
            _make_athlete(name="B", catapult_athlete_id="c2"),
            _make_athlete(name="C"),  # no catapult ID
        ]
        mapping = {
            a.catapult_athlete_id: a.user_id
            for a in athletes
            if a.catapult_athlete_id
        }
        assert len(mapping) == 2
        assert "c1" in mapping
        assert "c2" in mapping
