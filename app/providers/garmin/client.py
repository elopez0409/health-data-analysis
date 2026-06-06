import uuid
from datetime import datetime

from app.logging import get_logger
from app.providers.base import ProviderClient, RawRecord
from app.providers.garmin import auth
from app.providers.registry import ProviderRegistry
from app.providers.token_store import get_token
from app.schemas.common import ConnectionStatus, Provider

logger = get_logger(__name__)


@ProviderRegistry.register(Provider.GARMIN)
class GarminClient(ProviderClient):
    """Garmin Health API client.

    Garmin is push-based: data arrives via webhooks rather than polling.
    The pull() method returns an empty list; ingest happens in the webhook
    handler (app.webhooks.garmin).
    """

    provider_name = Provider.GARMIN

    def __init__(self, access_token: str | None = None):
        self._access_token = access_token

    async def verify_connection(self, user_id: uuid.UUID) -> ConnectionStatus:
        connected = self._access_token is not None
        return ConnectionStatus(
            provider=Provider.GARMIN,
            connected=connected,
            error=None if connected else "No access token stored",
        )

    async def pull(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> list[RawRecord]:
        logger.info("garmin_pull_noop", reason="push-based provider")
        return []

    async def get_authorize_url(self, state: str) -> str:
        request_token = await auth.get_request_token()
        return auth.get_authorize_url(request_token["oauth_token"])

    async def exchange_code(self, code: str) -> dict:
        raise NotImplementedError(
            "Garmin uses OAuth 1.0a verifier flow; "
            "call exchange_verifier() via auth module instead"
        )

    async def refresh_token(self, refresh_token_value: str) -> dict:
        raise NotImplementedError(
            "Garmin OAuth 1.0a tokens do not expire; no refresh needed"
        )
