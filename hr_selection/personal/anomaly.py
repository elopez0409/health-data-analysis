"""Change detection for sudden device offset shifts (e.g. firmware updates)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from hr_selection.personal.estimator import MIN_SAMPLES_FOR_CI, OffsetState


DEFAULT_SHIFT_THRESHOLD_BPM = 3.0
DEFAULT_MIN_BATCH_SIZE = 10
DEFAULT_Z_THRESHOLD = 2.5


@dataclass
class AnomalyResult:
    """Outcome of an anomaly check."""

    is_anomaly: bool
    shift_bpm: float
    prev_offset: float
    new_offset: float
    severity: str = "warning"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "shift_bpm": self.shift_bpm,
            "prev_offset": self.prev_offset,
            "new_offset": self.new_offset,
            "severity": self.severity,
            "reason": self.reason,
        }


def detect_offset_anomaly(
    state: OffsetState,
    recent_deltas: list[float],
    *,
    shift_threshold_bpm: float = DEFAULT_SHIFT_THRESHOLD_BPM,
    min_batch_size: int = DEFAULT_MIN_BATCH_SIZE,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> AnomalyResult:
    """Compare recent batch mean offset to stored distribution.

    Fires when:
    1. Enough historical samples exist (``n_samples >= MIN_SAMPLES_FOR_CI``)
    2. Recent batch has at least ``min_batch_size`` observations
    3. Batch mean exceeds stored CI by ``shift_threshold_bpm`` OR z-score > ``z_threshold``
    """
    valid = [d for d in recent_deltas if not math.isnan(d)]
    if len(valid) < min_batch_size:
        return AnomalyResult(
            is_anomaly=False,
            shift_bpm=0.0,
            prev_offset=state.offset_mean,
            new_offset=float("nan"),
            reason="insufficient_batch",
        )

    if state.n_samples < MIN_SAMPLES_FOR_CI:
        return AnomalyResult(
            is_anomaly=False,
            shift_bpm=0.0,
            prev_offset=state.offset_mean,
            new_offset=float("nan"),
            reason="insufficient_history",
        )

    batch_mean = sum(valid) / len(valid)
    prev = state.offset_mean
    shift = batch_mean - prev

    ci_low = state.ci_low
    ci_high = state.ci_high
    outside_ci = False
    if not math.isnan(ci_low) and not math.isnan(ci_high):
        if batch_mean < ci_low - shift_threshold_bpm or batch_mean > ci_high + shift_threshold_bpm:
            outside_ci = True

    std = math.sqrt(max(state.offset_var, 1e-6))
    z_score = abs(batch_mean - prev) / (std / math.sqrt(len(valid)))
    z_anomaly = z_score > z_threshold and abs(shift) >= shift_threshold_bpm

    is_anomaly = outside_ci or z_anomaly
    severity = "critical" if abs(shift) >= 2 * shift_threshold_bpm else "warning"

    return AnomalyResult(
        is_anomaly=is_anomaly,
        shift_bpm=shift,
        prev_offset=prev,
        new_offset=batch_mean,
        severity=severity if is_anomaly else "info",
        reason="offset_shift" if is_anomaly else "normal",
    )


def detect_self_drift_anomaly(
    profile_resting_mean: float,
    recent_hr_values: list[float],
    *,
    shift_threshold_bpm: float = DEFAULT_SHIFT_THRESHOLD_BPM,
    min_batch_size: int = DEFAULT_MIN_BATCH_SIZE,
) -> AnomalyResult:
    """Single-source fallback: detect drift vs habitual resting HR."""
    valid = [v for v in recent_hr_values if not math.isnan(v)]
    if len(valid) < min_batch_size or math.isnan(profile_resting_mean):
        return AnomalyResult(
            is_anomaly=False,
            shift_bpm=0.0,
            prev_offset=profile_resting_mean,
            new_offset=float("nan"),
            reason="insufficient_data",
        )

    batch_mean = sum(valid) / len(valid)
    shift = batch_mean - profile_resting_mean
    is_anomaly = abs(shift) >= shift_threshold_bpm

    return AnomalyResult(
        is_anomaly=is_anomaly,
        shift_bpm=shift,
        prev_offset=profile_resting_mean,
        new_offset=batch_mean,
        severity="warning" if is_anomaly else "info",
        reason="self_drift" if is_anomaly else "normal",
    )
