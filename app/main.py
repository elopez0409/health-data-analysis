from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.logging import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.jobs.scheduler import start_scheduler, shutdown_scheduler
    start_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(title="Health Data Backend", version="0.1.0", lifespan=lifespan)

from app.routes.oauth_callback import router as oauth_router
from app.routes.ingest import router as ingest_router
from app.webhooks.garmin import router as garmin_webhook_router

app.include_router(oauth_router)
app.include_router(ingest_router)
app.include_router(garmin_webhook_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
