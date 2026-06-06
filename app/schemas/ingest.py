from datetime import date, datetime

from pydantic import BaseModel


class AppleHealthSample(BaseModel):
    """A single sample from Apple HealthKit, POSTed by a mobile companion app."""

    sample_type: str
    value: float | None = None
    unit: str | None = None
    start_date: datetime
    end_date: datetime | None = None
    source_name: str | None = None
    device: str | None = None
    metadata: dict | None = None


class AppleHealthIngestRequest(BaseModel):
    """Batch of Apple Health samples to ingest."""

    user_id: str
    samples: list[AppleHealthSample]


class AppleHealthIngestResponse(BaseModel):
    """Response from the Apple Health ingest endpoint."""

    accepted: int
    rejected: int
    message: str = "ok"
