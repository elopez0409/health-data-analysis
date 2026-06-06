from fastapi import APIRouter

from app.logging import get_logger
from app.schemas.ingest import (
    AppleHealthIngestRequest,
    AppleHealthIngestResponse,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = get_logger(__name__)


@router.post("/apple_health", response_model=AppleHealthIngestResponse)
async def ingest_apple_health(request: AppleHealthIngestRequest):
    """Stub endpoint for Apple HealthKit data ingestion.

    A mobile companion app will POST batches of HealthKit samples here.
    Currently accepts and validates the payload but does not persist.
    """
    logger.info(
        "apple_health_ingest_stub",
        user_id=request.user_id,
        sample_count=len(request.samples),
    )

    return AppleHealthIngestResponse(
        accepted=len(request.samples),
        rejected=0,
        message="accepted (stub - not yet persisted)",
    )
