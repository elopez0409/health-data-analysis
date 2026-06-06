import httpx

from app.config import settings

WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"


def get_authorize_url(state: str) -> str:
    params = {
        "client_id": settings.whoop_client_id,
        "redirect_uri": settings.whoop_redirect_uri,
        "response_type": "code",
        "scope": "read:recovery read:sleep read:workout read:profile",
        "state": state,
    }
    return f"{WHOOP_AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"


async def exchange_code(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            WHOOP_TOKEN_URL,
            data={
                "client_id": settings.whoop_client_id,
                "client_secret": settings.whoop_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.whoop_redirect_uri,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(refresh_token_value: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            WHOOP_TOKEN_URL,
            data={
                "client_id": settings.whoop_client_id,
                "client_secret": settings.whoop_client_secret,
                "refresh_token": refresh_token_value,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        return resp.json()
