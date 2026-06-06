import asyncio
import time
from dataclasses import dataclass, field

from app.schemas.common import Provider

# Rate limits per provider (requests per minute)
PROVIDER_RATE_LIMITS: dict[Provider, int] = {
    Provider.STRAVA: 100,       # 100 requests per 15 min ≈ 6.7/min, be conservative
    Provider.FITBIT: 150,       # 150 requests/hour
    Provider.OURA: 60,          # Undocumented, conservative
    Provider.WITHINGS: 60,      # 120 requests per minute for most endpoints
    Provider.WHOOP: 60,         # Beta API, conservative
    Provider.GARMIN: 30,        # Push-based, minimal polling needed
}


@dataclass
class TokenBucket:
    """Token-bucket rate limiter for API calls."""

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until enough tokens are available, then consume them."""
        while True:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            deficit = tokens - self.tokens
            wait_time = deficit / self.refill_rate
            await asyncio.sleep(wait_time)


class RateLimiterRegistry:
    """Per-provider rate limiter instances."""

    _buckets: dict[Provider, TokenBucket] = {}

    @classmethod
    def get(cls, provider: Provider) -> TokenBucket:
        if provider not in cls._buckets:
            rpm = PROVIDER_RATE_LIMITS.get(provider, 60)
            cls._buckets[provider] = TokenBucket(
                capacity=float(rpm),
                refill_rate=rpm / 60.0,
            )
        return cls._buckets[provider]

    @classmethod
    def reset(cls) -> None:
        cls._buckets.clear()
