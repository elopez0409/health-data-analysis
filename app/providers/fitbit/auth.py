import base64

import httpx

from app.config import settings

FITBIT_AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"


def get_authorize_url(state: str) -> str:
    params = {
        "client_id": settings.fitbit_client_id,
        "redirect_uri": settings.fitbit_redirect_uri,
        "response_type": "code",
        "scope": "activity heartrate sleep profile",
        "state": state,
    }
    return f"{FITBIT_AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"


def _basic_auth_header() -> str:
    credentials = f"{settings.fitbit_client_id}:{settings.fitbit_client_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


async def exchange_code(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            FITBIT_TOKEN_URL,
            headers={
                "Authorization": _basic_auth_header(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.fitbit_redirect_uri,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(refresh_token_value: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            FITBIT_TOKEN_URL,
            headers={
                "Authorization": _basic_auth_header(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "refresh_token": refresh_token_value,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        return resp.json()
