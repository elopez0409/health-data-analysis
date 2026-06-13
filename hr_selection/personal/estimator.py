"""Online offset estimator using Welford's algorithm + EWMA track."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


DEFAULT_Z = 1.96
DEFAULT_EWMA_ALPHA = 0.05
MIN_SAMPLES_FOR_CI = 5


@dataclass
class OffsetState:
    """Running statistics for ``offset = hr_source - hr_trusted``."""

    offset_mean: float = 0.0
    offset_var: float = 0.0
    n_samples: int = 0
    ewma_offset: float = 0.0
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    _m2: float = field(default=0.0, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> OffsetState:
        return cls(
            offset_mean=float(data.get("offset_mean", 0.0)),
            offset_var=float(data.get("offset_var", 0.0)),
            n_samples=int(data.get("n_samples", 0)),
            ewma_offset=float(data.get("ewma_offset", 0.0)),
            ci_low=float(data.get("ci_low", float("nan"))),
            ci_high=float(data.get("ci_high", float("nan"))),
        )

    def to_dict(self) -> dict:
        return {
            "offset_mean": self.offset_mean,
            "offset_var": self.offset_var,
            "n_samples": self.n_samples,
            "ewma_offset": self.ewma_offset,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def confidence_interval(
    mean: float,
    var: float,
    n: int,
    z: float = DEFAULT_Z,
) -> tuple[float, float]:
    """Compute mean ± z * std / sqrt(n). Returns (nan, nan) when n < MIN_SAMPLES_FOR_CI."""
    if n < MIN_SAMPLES_FOR_CI or var < 0:
        return float("nan"), float("nan")
    std = math.sqrt(max(var, 0.0))
    half_width = z * std / math.sqrt(n)
    return mean - half_width, mean + half_width


def update_offset(
    state: OffsetState,
    delta: float,
    *,
    ewma_alpha: float = DEFAULT_EWMA_ALPHA,
    z: float = DEFAULT_Z,
) -> OffsetState:
    """Incorporate one offset observation (hr_source - hr_trusted) into running stats."""
    if math.isnan(delta):
        return state

    n = state.n_samples + 1
    delta_n = delta - state.offset_mean
    new_mean = state.offset_mean + delta_n / n
    delta_n2 = delta - new_mean
    m2 = state._m2 + delta_n * delta_n2

    if n == 1:
        ewma = delta
    else:
        ewma = ewma_alpha * delta + (1.0 - ewma_alpha) * state.ewma_offset

    var = m2 / n if n > 1 else 0.0
    ci_low, ci_high = confidence_interval(new_mean, var, n, z=z)

    return OffsetState(
        offset_mean=new_mean,
        offset_var=var,
        n_samples=n,
        ewma_offset=ewma,
        ci_low=ci_low,
        ci_high=ci_high,
        _m2=m2,
    )


def batch_update_offset(
    state: OffsetState,
    deltas: list[float],
    *,
    ewma_alpha: float = DEFAULT_EWMA_ALPHA,
    z: float = DEFAULT_Z,
) -> OffsetState:
    """Apply a sequence of offset observations."""
    for d in deltas:
        state = update_offset(state, d, ewma_alpha=ewma_alpha, z=z)
    return state
