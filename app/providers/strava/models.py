from datetime import datetime

from pydantic import BaseModel, Field


class StravaAthlete(BaseModel):
    id: int
    firstname: str | None = None
    lastname: str | None = None
    username: str | None = None
    city: str | None = None
    country: str | None = None


class StravaActivity(BaseModel):
    id: int
    name: str | None = None
    type: str = Field(alias="type", default="Unknown")
    sport_type: str | None = None
    start_date: datetime
    start_date_local: datetime | None = None
    elapsed_time: int  # seconds
    moving_time: int | None = None  # seconds
    distance: float = 0.0  # meters
    total_elevation_gain: float | None = None  # meters
    average_heartrate: float | None = None  # bpm
    max_heartrate: float | None = None  # bpm
    kilojoules: float | None = None
    calories: float | None = None

    model_config = {"populate_by_name": True}


class StravaTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int
    token_type: str = "Bearer"
    athlete: StravaAthlete | None = None
