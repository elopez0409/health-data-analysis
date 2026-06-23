import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field


# --- Request models ---


class AthleteCreate(BaseModel):
    name: str
    sport: str | None = None
    position: str | None = None
    jersey_number: str | None = None
    catapult_athlete_id: str | None = None


class AthleteUpdate(BaseModel):
    name: str | None = None
    sport: str | None = None
    position: str | None = None
    jersey_number: str | None = None
    catapult_athlete_id: str | None = None
    is_active: bool | None = None


# --- Response models ---


class AthleteOut(BaseModel):
    id: uuid.UUID
    name: str
    sport: str | None = None
    position: str | None = None
    jersey_number: str | None = None
    is_active: bool

    model_config = {"from_attributes": True}


class AthleteReadiness(BaseModel):
    id: uuid.UUID
    name: str
    sport: str | None = None
    position: str | None = None
    jersey_number: str | None = None
    status: str  # green / yellow / red / no_data
    status_reasons: list[str] = Field(default_factory=list)
    readiness_score: float | None = None
    recovery_score: float | None = None
    hrv_rmssd: float | None = None
    resting_hr: float | None = None
    sleep_hours: float | None = None
    sleep_score: float | None = None
    anomaly_count: int = 0
    hr_offset: float | None = None
    metrics_date: date | None = None
    sleep_date: date | None = None


class TeamReadinessResponse(BaseModel):
    team_size: int
    reporting: int
    green: int
    yellow: int
    red: int
    no_data: int
    as_of: datetime
    athletes: list[AthleteReadiness]


class DailyMetricPoint(BaseModel):
    date: date
    readiness_score: float | None = None
    recovery_score: float | None = None
    hrv_rmssd: float | None = None
    resting_heart_rate: float | None = None
    steps: int | None = None
    strain_score: float | None = None

    model_config = {"from_attributes": True}


class SleepPoint(BaseModel):
    date: date
    total_hours: float | None = None
    sleep_score: float | None = None
    deep_hours: float | None = None
    rem_hours: float | None = None

    model_config = {"from_attributes": True}


class RecentActivity(BaseModel):
    activity_type: str
    started_at: datetime
    duration_seconds: float | None = None
    title: str | None = None

    model_config = {"from_attributes": True}


class AnomalyOut(BaseModel):
    source: str
    detected_at: datetime
    shift_bpm: float
    severity: str
    resolved: bool

    model_config = {"from_attributes": True}


class HrStateOut(BaseModel):
    source: str
    offset_mean: float
    n_samples: int
    last_updated: datetime

    model_config = {"from_attributes": True}


class AthleteSummaryResponse(BaseModel):
    athlete: AthleteOut
    daily_metrics: list[DailyMetricPoint]
    sleep: list[SleepPoint]
    recent_activities: list[RecentActivity]
    anomalies: list[AnomalyOut]
    hr_state: list[HrStateOut]
