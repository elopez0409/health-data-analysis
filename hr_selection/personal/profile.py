"""Habitual HR baseline profile: resting HR by context and circadian curve."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from hr_selection.priors.lookup import bucket_activity

DEFAULT_EWMA_ALPHA = 0.05


@dataclass
class BaselineProfile:
    """Incremental personal HR habits."""

    resting_by_context: dict[str, float] = field(default_factory=dict)
    context_counts: dict[str, int] = field(default_factory=dict)
    circadian: dict[str, float] = field(default_factory=dict)
    circadian_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict | None) -> BaselineProfile:
        if not data:
            return cls()
        return cls(
            resting_by_context=dict(data.get("resting_by_context", {})),
            context_counts=dict(data.get("context_counts", {})),
            circadian=dict(data.get("circadian", {})),
            circadian_counts=dict(data.get("circadian_counts", {})),
        )

    def to_dict(self) -> dict:
        return {
            "resting_by_context": self.resting_by_context,
            "context_counts": self.context_counts,
            "circadian": self.circadian,
            "circadian_counts": self.circadian_counts,
        }

    def resting_mean(self, context: str = "rest") -> float:
        """Return habitual resting HR for a context bucket (default ``rest``)."""
        bucket = bucket_activity(context)
        val = self.resting_by_context.get(bucket)
        if val is not None:
            return float(val)
        if self.resting_by_context:
            return float(sum(self.resting_by_context.values()) / len(self.resting_by_context))
        return float("nan")


def _ewma_update(prev: float | None, value: float, alpha: float) -> float:
    if prev is None or math.isnan(prev):
        return value
    return alpha * value + (1.0 - alpha) * prev


def update_profile(
    profile: BaselineProfile,
    *,
    bpm: float,
    context: str | None = None,
    hour: int | None = None,
    alpha: float = DEFAULT_EWMA_ALPHA,
) -> BaselineProfile:
    """Incrementally update habitual baseline from one HR observation."""
    if math.isnan(bpm):
        return profile

    resting = dict(profile.resting_by_context)
    ctx_counts = dict(profile.context_counts)
    circadian = dict(profile.circadian)
    circ_counts = dict(profile.circadian_counts)

    bucket = bucket_activity(context)
    prev_ctx = resting.get(bucket)
    resting[bucket] = _ewma_update(prev_ctx, bpm, alpha)
    ctx_counts[bucket] = ctx_counts.get(bucket, 0) + 1

    if hour is not None and 0 <= hour <= 23:
        hour_key = str(hour)
        prev_h = circadian.get(hour_key)
        circadian[hour_key] = _ewma_update(prev_h, bpm, alpha)
        circ_counts[hour_key] = circ_counts.get(hour_key, 0) + 1

    return BaselineProfile(
        resting_by_context=resting,
        context_counts=ctx_counts,
        circadian=circadian,
        circadian_counts=circ_counts,
    )


def batch_update_profile(
    profile: BaselineProfile,
    observations: list[dict],
    *,
    alpha: float = DEFAULT_EWMA_ALPHA,
) -> BaselineProfile:
    """Update profile from a list of ``{bpm, context?, hour?}`` dicts."""
    for obs in observations:
        profile = update_profile(
            profile,
            bpm=obs["bpm"],
            context=obs.get("context"),
            hour=obs.get("hour"),
            alpha=alpha,
        )
    return profile
