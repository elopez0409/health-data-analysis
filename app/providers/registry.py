from app.config import settings
from app.providers.base import ProviderClient
from app.schemas.common import Provider


class ProviderRegistry:
    """Registry of enabled provider clients with feature-flag support."""

    _clients: dict[Provider, type[ProviderClient]] = {}

    @classmethod
    def register(cls, provider: Provider):
        def decorator(client_cls: type[ProviderClient]):
            cls._clients[provider] = client_cls
            return client_cls
        return decorator

    @classmethod
    def get_client(cls, provider: Provider) -> ProviderClient:
        if provider not in cls._clients:
            raise ValueError(f"Provider {provider} not registered")
        return cls._clients[provider]()

    @classmethod
    def enabled_providers(cls) -> list[Provider]:
        enabled = []
        for provider in cls._clients:
            if provider == Provider.GARMIN and not settings.garmin_enabled:
                continue
            enabled.append(provider)
        return enabled

    @classmethod
    def all_registered(cls) -> list[Provider]:
        return list(cls._clients.keys())
