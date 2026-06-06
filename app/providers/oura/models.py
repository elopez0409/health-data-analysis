from datetime import datetime

from pydantic import BaseModel, Field


class OuraPersonalInfo(BaseModel):
    id: str | None = None
    age: int | None = None
    weight: float | None = None
    height: float | None = None
    email: str | None = None


class OuraDailySleep(BaseModel):
    id: str
    day: str  # YYYY-MM-DD
    score: int | None = None
    timestamp: datetime | None = None
    contributors: dict | None = None
    total_sleep_duration: int | None = None  # seconds
    rem_sleep_duration: int | None = None  # seconds
    deep_sleep_duration: int | None = None  # seconds
    light_sleep_duration: int | None = None  # seconds
    awake_time: int | None = None  # seconds
    bedtime_start: datetime | None = None
    bedtime_end: datetime | None = None
    time_in_bed: int | None = None  # seconds
    efficiency: int | None = None
    restless_periods: int | None = None
    average_heart_rate: float | None = None
    lowest_heart_rate: int | None = None
    average_hrv: int | None = None
    latency: int | None = None  # seconds


class OuraDailyReadiness(BaseModel):
    id: str
    day: str  # YYYY-MM-DD
    score: int | None = None
    timestamp: datetime | None = None
    contributors: dict | None = None
    temperature_deviation: float | None = None  # Celsius deviation
    temperature_trend_deviation: float | None = None
    hrv_balance: dict | None = Field(
        default=None, description="HRV balance contributor details"
    )


class OuraDailyActivity(BaseModel):
    id: str
    day: str  # YYYY-MM-DD
    score: int | None = None
    timestamp: datetime | None = None
    active_calories: int | None = None
    total_calories: int | None = None
    steps: int | None = None
    equivalent_walking_distance: int | None = None  # meters
    high_activity_time: int | None = None  # seconds
    medium_activity_time: int | None = None  # seconds
    low_activity_time: int | None = None  # seconds
    sedentary_time: int | None = None  # seconds
    resting_time: int | None = None  # seconds
    non_wear_time: int | None = None  # seconds
    average_met_minutes: float | None = None
    contributors: dict | None = None
    met: dict | None = None


class OuraTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
