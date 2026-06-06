from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://health:health@localhost:5432/health"
    secret_key: str = "change-me-in-production"
    log_level: str = "INFO"

    strava_client_id: str = ""
    strava_client_secret: str = ""
    strava_redirect_uri: str = "http://localhost:8000/auth/strava/callback"

    fitbit_client_id: str = ""
    fitbit_client_secret: str = ""
    fitbit_redirect_uri: str = "http://localhost:8000/auth/fitbit/callback"

    oura_client_id: str = ""
    oura_client_secret: str = ""
    oura_redirect_uri: str = "http://localhost:8000/auth/oura/callback"

    withings_client_id: str = ""
    withings_client_secret: str = ""
    withings_redirect_uri: str = "http://localhost:8000/auth/withings/callback"

    whoop_client_id: str = ""
    whoop_client_secret: str = ""
    whoop_redirect_uri: str = "http://localhost:8000/auth/whoop/callback"

    garmin_enabled: bool = False
    garmin_consumer_key: str = ""
    garmin_consumer_secret: str = ""
    garmin_webhook_secret: str = ""

    default_user_id: str = "00000000-0000-0000-0000-000000000001"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
