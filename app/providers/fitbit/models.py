from datetime import date, datetime

from pydantic import BaseModel


class FitbitProfile(BaseModel):
    encodedId: str
    displayName: str | None = None
    fullName: str | None = None


class FitbitSleepStage(BaseModel):
    dateTime: str
    level: str
    seconds: int


class FitbitSleepSummary(BaseModel):
    totalMinutesAsleep: int | None = None
    totalTimeInBed: int | None = None
    stages: dict | None = None


class FitbitSleepLog(BaseModel):
    logId: int
    dateOfSleep: str
    startTime: str
    endTime: str
    duration: int  # milliseconds
    efficiency: int | None = None
    minutesAsleep: int | None = None
    minutesAwake: int | None = None
    levels: dict | None = None


class FitbitActivitySummary(BaseModel):
    steps: int = 0
    caloriesOut: int = 0
    activeMinutes: int | None = None
    sedentaryMinutes: int | None = None
    restingHeartRate: int | None = None


class FitbitHeartRateZone(BaseModel):
    name: str
    min: int
    max: int
    minutes: int | None = None
    caloriesOut: float | None = None


class FitbitHeartRateIntraday(BaseModel):
    time: str
    value: int


class FitbitTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
    user_id: str | None = None
    scope: str | None = None
