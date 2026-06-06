from pydantic import BaseModel, Field


class GarminActivitySummary(BaseModel):
    """Single activity record from a Garmin Health API webhook push."""

    activity_id: int = Field(alias="activityId")
    activity_type: str = Field(alias="activityType", default="Unknown")
    start_time_in_seconds: int = Field(alias="startTimeInSeconds")
    duration_in_seconds: int = Field(alias="durationInSeconds")
    distance_in_meters: float = Field(alias="distanceInMeters", default=0.0)
    active_kilocalories: float = Field(alias="activeKilocalories", default=0.0)
    average_heart_rate_in_bpm: float | None = Field(
        alias="averageHeartRateInBeatsPerMinute", default=None
    )
    max_heart_rate_in_bpm: float | None = Field(
        alias="maxHeartRateInBeatsPerMinute", default=None
    )

    model_config = {"populate_by_name": True}


class GarminSleepSummary(BaseModel):
    """Single sleep record from a Garmin Health API webhook push."""

    start_time_in_seconds: int = Field(alias="startTimeInSeconds")
    duration_in_seconds: int = Field(alias="durationInSeconds")
    deep_sleep_duration_in_seconds: int = Field(
        alias="deepSleepDurationInSeconds", default=0
    )
    light_sleep_duration_in_seconds: int = Field(
        alias="lightSleepDurationInSeconds", default=0
    )
    rem_sleep_duration_in_seconds: int = Field(
        alias="remSleepDurationInSeconds", default=0
    )
    awake_duration_in_seconds: int = Field(
        alias="awakeDurationInSeconds", default=0
    )

    model_config = {"populate_by_name": True}


class GarminActivityPush(BaseModel):
    """Top-level webhook payload for activity summaries."""

    activities: list[GarminActivitySummary] = Field(alias="activities", default=[])

    model_config = {"populate_by_name": True}


class GarminSleepPush(BaseModel):
    """Top-level webhook payload for sleep summaries."""

    sleeps: list[GarminSleepSummary] = Field(alias="sleeps", default=[])

    model_config = {"populate_by_name": True}
