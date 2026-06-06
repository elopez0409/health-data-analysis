import pytest


@pytest.fixture(autouse=True)
def _env_setup(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://health:health@localhost:5432/health_test")
    monkeypatch.setenv("STRAVA_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("STRAVA_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("FITBIT_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("FITBIT_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("OURA_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("OURA_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("WITHINGS_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("WITHINGS_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("WHOOP_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("WHOOP_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("GARMIN_ENABLED", "true")
    monkeypatch.setenv("GARMIN_CONSUMER_KEY", "test-consumer-key")
    monkeypatch.setenv("GARMIN_CONSUMER_SECRET", "test-consumer-secret")
    monkeypatch.setenv("GARMIN_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("WHOOP_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("WHOOP_CLIENT_SECRET", "test-client-secret")
