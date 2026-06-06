from enum import Enum

from pydantic import BaseModel


class Provider(str, Enum):
    STRAVA = "strava"
    FITBIT = "fitbit"
    OURA = "oura"
    WITHINGS = "withings"
    WHOOP = "whoop"
    GARMIN = "garmin"
    APPLE_HEALTH = "apple_health"


class ConnectionStatus(BaseModel):
    provider: Provider
    connected: bool
    latency_ms: float | None = None
    error: str | None = None
    username: str | None = None
