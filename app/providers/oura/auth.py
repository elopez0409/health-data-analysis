import httpx

from app.config import settings

OURA_AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
OURA_TOKEN_URL = "https://api.ouraring.com/oauth/token"


def get_authorize_url(state: str) -> str:
    params = {
        "client_id": settings.oura_client_id,
        "redirect_uri": settings.oura_redirect_uri,
        "response_type": "code",
        "scope": "daily",
        "state": state,
    }
    return f"{OURA_AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"


async def exchange_code(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OURA_TOKEN_URL,
            data={
                "client_id": settings.oura_client_id,
                "client_secret": settings.oura_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.oura_redirect_uri,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(refresh_token_value: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OURA_TOKEN_URL,
            data={
                "client_id": settings.oura_client_id,
                "client_secret": settings.oura_client_secret,
                "refresh_token": refresh_token_value,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        return resp.json()
