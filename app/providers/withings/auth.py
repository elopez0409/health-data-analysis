import httpx

from app.config import settings

WITHINGS_AUTH_URL = "https://account.withings.com/oauth2_user/authorize2"
WITHINGS_TOKEN_URL = "https://wbsapi.withings.net/v2/oauth2"


def get_authorize_url(state: str) -> str:
    params = {
        "client_id": settings.withings_client_id,
        "redirect_uri": settings.withings_redirect_uri,
        "response_type": "code",
        "scope": "user.metrics,user.activity",
        "state": state,
    }
    return f"{WITHINGS_AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"


async def exchange_code(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            WITHINGS_TOKEN_URL,
            data={
                "action": "requesttoken",
                "client_id": settings.withings_client_id,
                "client_secret": settings.withings_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.withings_redirect_uri,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["body"]


async def refresh_access_token(refresh_token_value: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            WITHINGS_TOKEN_URL,
            data={
                "action": "requesttoken",
                "client_id": settings.withings_client_id,
                "client_secret": settings.withings_client_secret,
                "refresh_token": refresh_token_value,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["body"]
