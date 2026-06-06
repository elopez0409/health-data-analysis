import uuid
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.logging import get_logger

logger = get_logger(__name__)

scheduler = AsyncIOScheduler()


def setup_scheduler() -> None:
    """Configure and start the APScheduler instance.

    Jobs are registered for each enabled provider. Designed to be swappable
    with Airflow or any other orchestrator by replacing this module.
    """
    user_id = uuid.UUID(settings.default_user_id)

    # Strava: pull every 15 minutes
    scheduler.add_job(
        _run_strava_pull,
        trigger=IntervalTrigger(minutes=15),
        id="strava_pull",
        kwargs={"user_id": user_id},
        replace_existing=True,
    )

    # Fitbit: pull every 30 minutes
    scheduler.add_job(
        _run_fitbit_pull,
        trigger=IntervalTrigger(minutes=30),
        id="fitbit_pull",
        kwargs={"user_id": user_id},
        replace_existing=True,
    )

    logger.info("scheduler_configured", jobs=len(scheduler.get_jobs()))


async def _run_strava_pull(user_id: uuid.UUID) -> None:
    from app.jobs.strava_pull import strava_pull_job
    from app.rate_limit import RateLimiterRegistry
    from app.schemas.common import Provider

    await RateLimiterRegistry.get(Provider.STRAVA).acquire()
    try:
        count = await strava_pull_job(user_id)
        logger.info("scheduled_strava_pull_done", count=count)
    except Exception as e:
        logger.error("scheduled_strava_pull_failed", error=str(e))


async def _run_fitbit_pull(user_id: uuid.UUID) -> None:
    from app.jobs.fitbit_pull import fitbit_pull_job
    from app.rate_limit import RateLimiterRegistry
    from app.schemas.common import Provider

    await RateLimiterRegistry.get(Provider.FITBIT).acquire()
    try:
        count = await fitbit_pull_job(user_id)
        logger.info("scheduled_fitbit_pull_done", count=count)
    except Exception as e:
        logger.error("scheduled_fitbit_pull_failed", error=str(e))


def start_scheduler() -> None:
    setup_scheduler()
    scheduler.start()
    logger.info("scheduler_started")


def shutdown_scheduler() -> None:
    scheduler.shutdown(wait=False)
    logger.info("scheduler_stopped")
