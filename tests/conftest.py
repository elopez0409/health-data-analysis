import pytest


@pytest.fixture(scope="session")
def hr_synthetic_root(tmp_path_factory):
    """Generate the three synthetic HR datasets once into a temp dir.

    Returns the root containing ``bigideas/``, ``galaxyppg/``, ``ppg_dalia/``.
    """
    from hr_selection.synthetic import generate_all

    root = tmp_path_factory.mktemp("hr_raw")
    generate_all(root=root)
    return root


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
