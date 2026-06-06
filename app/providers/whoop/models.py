from datetime import datetime

from pydantic import BaseModel, Field


class WhoopUserProfile(BaseModel):
    user_id: int
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None


class WhoopRecoveryScore(BaseModel):
    user_calibrating: bool = False
    recovery_score: float
    resting_heart_rate: float
    hrv_rmssd_milli: float
    spo2_percentage: float | None = None
    skin_temp_celsius: float | None = None


class WhoopRecovery(BaseModel):
    cycle_id: int
    sleep_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime | None = None
    score_state: str = "SCORED"
    score: WhoopRecoveryScore


class WhoopSleepStageSummary(BaseModel):
    total_in_bed_time_milli: int = 0
    total_awake_time_milli: int = 0
    total_no_data_time_milli: int = 0
    total_light_sleep_time_milli: int = 0
    total_slow_wave_sleep_time_milli: int = 0
    total_rem_sleep_time_milli: int = 0
    sleep_cycle_count: int = 0
    disturbance_count: int = 0


class WhoopSleepScore(BaseModel):
    stage_summary: WhoopSleepStageSummary
    sleep_needed: dict | None = None
    respiratory_rate: float | None = None
    sleep_performance_percentage: float | None = None
    sleep_consistency_percentage: float | None = None
    sleep_efficiency_percentage: float | None = None


class WhoopSleep(BaseModel):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime | None = None
    start: datetime
    end: datetime
    timezone_offset: str | None = None
    nap: bool = False
    score_state: str = "SCORED"
    score: WhoopSleepScore


class WhoopWorkoutScore(BaseModel):
    strain: float
    average_heart_rate: int | None = None
    max_heart_rate: int | None = None
    kilojoule: float
    percent_recorded: float | None = None
    distance_meter: float | None = None
    altitude_gain_meter: float | None = None
    altitude_change_meter: float | None = None
    zone_duration: dict | None = None


class WhoopWorkout(BaseModel):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime | None = None
    start: datetime
    end: datetime
    timezone_offset: str | None = None
    sport_id: int
    score_state: str = "SCORED"
    score: WhoopWorkoutScore


class WhoopTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
    scope: str | None = None


class WhoopPaginatedResponse(BaseModel):
    records: list[dict] = Field(default_factory=list)
    next_token: str | None = None
