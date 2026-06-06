from authlib.integrations.httpx_client import AsyncOAuth1Client

from app.config import settings

REQUEST_TOKEN_URL = (
    "https://connectapi.garmin.com/oauth-service/oauth/request_token"
)
AUTHORIZE_URL = "https://connect.garmin.com/oauthConfirm"
ACCESS_TOKEN_URL = (
    "https://connectapi.garmin.com/oauth-service/oauth/access_token"
)


def _make_client(
    token: str | None = None,
    token_secret: str | None = None,
) -> AsyncOAuth1Client:
    return AsyncOAuth1Client(
        client_id=settings.garmin_consumer_key,
        client_secret=settings.garmin_consumer_secret,
        token=token,
        token_secret=token_secret,
    )


async def get_request_token() -> dict:
    """Fetch an OAuth 1.0a request token from Garmin."""
    client = _make_client()
    token = await client.fetch_request_token(REQUEST_TOKEN_URL)
    await client.aclose()
    return {
        "oauth_token": token["oauth_token"],
        "oauth_token_secret": token["oauth_token_secret"],
    }


def get_authorize_url(oauth_token: str) -> str:
    """Build the Garmin authorization redirect URL."""
    return f"{AUTHORIZE_URL}?oauth_token={oauth_token}"


async def exchange_verifier(
    oauth_token: str,
    oauth_verifier: str,
    oauth_token_secret: str,
) -> dict:
    """Exchange verifier for an OAuth 1.0a access token."""
    client = _make_client(token=oauth_token, token_secret=oauth_token_secret)
    token = await client.fetch_access_token(
        ACCESS_TOKEN_URL, verifier=oauth_verifier
    )
    await client.aclose()
    return {
        "oauth_token": token["oauth_token"],
        "oauth_token_secret": token["oauth_token_secret"],
    }
