import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import httpx

from app.schemas.common import ConnectionStatus, Provider


@dataclass
class RawRecord:
    """Intermediate representation of a fetched raw record before DB insert."""

    external_id: str
    payload: dict
    fetched_at: datetime | None = None

    @property
    def payload_hash(self) -> str:
        serialized = json.dumps(self.payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


class ProviderClient(ABC):
    """Base class for all provider API clients."""

    provider_name: Provider

    @abstractmethod
    async def verify_connection(self, user_id: uuid.UUID) -> ConnectionStatus:
        """Hit a lightweight endpoint to check connectivity."""
        ...

    @abstractmethod
    async def pull(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> list[RawRecord]:
        """Fetch new records since last cursor."""
        ...

    @abstractmethod
    async def get_authorize_url(self, state: str) -> str:
        """Generate the OAuth authorize redirect URL."""
        ...

    @abstractmethod
    async def exchange_code(self, code: str) -> dict:
        """Exchange authorization code for tokens."""
        ...

    @abstractmethod
    async def refresh_token(self, refresh_token_value: str) -> dict:
        """Refresh an expired access token."""
        ...
