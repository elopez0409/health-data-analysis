"""Tests for the personal living HR model."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from hr_selection.personal.align import align_heart_rate_windows, compute_deltas
from hr_selection.personal.anomaly import detect_offset_anomaly, detect_self_drift_anomaly
from hr_selection.personal.estimator import (
    OffsetState,
    batch_update_offset,
    confidence_interval,
    update_offset,
)
from hr_selection.personal.profile import BaselineProfile, update_profile
from hr_selection.personal.trusted import select_trusted_source


def test_ci_tightens_with_more_samples():
    state = OffsetState()
    ci_widths = []
    for i in range(50):
        delta = 2.0 + (i % 3) * 0.1  # stable ~2 bpm offset with tiny noise
        state = update_offset(state, delta)
        if state.n_samples >= 5:
            width = state.ci_high - state.ci_low
            ci_widths.append(width)

    assert len(ci_widths) >= 10
    assert ci_widths[-1] < ci_widths[0]


def test_confidence_interval_requires_min_samples():
    low, high = confidence_interval(2.0, 0.5, n=3)
    assert math.isnan(low)
    assert math.isnan(high)

    low, high = confidence_interval(2.0, 0.5, n=10)
    assert low < 2.0 < high


def test_anomaly_fires_on_3bpm_step():
    state = OffsetState()
    for _ in range(100):
        state = update_offset(state, 1.0 + (0.01 * (_ % 5)))

    recent = [5.0] * 20  # sudden +4 bpm shift
    result = detect_offset_anomaly(state, recent, shift_threshold_bpm=3.0)
    assert result.is_anomaly is True
    assert abs(result.shift_bpm) >= 3.0


def test_anomaly_does_not_fire_on_noise():
    state = OffsetState()
    for _ in range(100):
        state = update_offset(state, 1.0)

    recent = [1.0 + 0.2 * ((_ % 3) - 1) for _ in range(20)]
    result = detect_offset_anomaly(state, recent, shift_threshold_bpm=3.0)
    assert result.is_anomaly is False


def test_self_drift_anomaly_single_source():
    result = detect_self_drift_anomaly(
        profile_resting_mean=65.0,
        recent_hr_values=[68.5] * 15,
        shift_threshold_bpm=3.0,
    )
    assert result.is_anomaly is True
    assert result.reason == "self_drift"


def test_select_trusted_source_prefers_lower_mae():
    trusted = select_trusted_source(["fitbit", "garmin"], context="rest")
    assert trusted in ("fitbit", "garmin")


def test_select_trusted_source_keeps_existing():
    trusted = select_trusted_source(
        ["fitbit", "garmin"],
        existing_trusted="fitbit",
    )
    assert trusted == "fitbit"


def test_align_single_source():
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(60):
        rows.append(
            {
                "recorded_at": base + timedelta(seconds=i),
                "source": "fitbit",
                "bpm": 70 + (i % 5),
                "context": "resting",
            }
        )
    df = pd.DataFrame(rows)
    windows = align_heart_rate_windows(df)
    assert len(windows) > 0
    assert all("fitbit" in w["sources"] for w in windows)


def test_align_multi_source_deltas():
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(60):
        t = base + timedelta(seconds=i)
        rows.append({"recorded_at": t, "source": "garmin", "bpm": 72, "context": "resting"})
        rows.append({"recorded_at": t, "source": "fitbit", "bpm": 75, "context": "resting"})
    df = pd.DataFrame(rows)
    windows = align_heart_rate_windows(df)
    trusted = "garmin"
    deltas = compute_deltas(windows, trusted)
    assert "fitbit" in deltas
    assert len(deltas["fitbit"]) > 0
    assert abs(sum(deltas["fitbit"]) / len(deltas["fitbit"]) - 3.0) < 0.5


def test_profile_updates_resting_by_context():
    profile = BaselineProfile()
    profile = update_profile(profile, bpm=62.0, context="resting", hour=8)
    profile = update_profile(profile, bpm=64.0, context="resting", hour=9)
    assert "rest" in profile.resting_by_context
    assert 62.0 <= profile.resting_mean("rest") <= 64.0


def test_batch_update_offset():
    state = batch_update_offset(OffsetState(), [2.0, 2.1, 1.9, 2.0, 2.05])
    assert state.n_samples == 5
    assert 1.9 <= state.offset_mean <= 2.1


def test_single_source_degradation_graceful():
    """Single-source users get profile + self-drift, not cross-device offsets."""
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "recorded_at": base + timedelta(seconds=i),
            "source": "fitbit",
            "bpm": 70.0,
            "context": "resting",
        }
        for i in range(60)
    ]
    df = pd.DataFrame(rows)
    windows = align_heart_rate_windows(df)
    sources = set()
    for w in windows:
        sources.update(w["sources"])
    assert sources == {"fitbit"}

    deltas = compute_deltas(windows, "fitbit")
    assert deltas == {}  # no cross-source deltas when only trusted device present

    profile = BaselineProfile()
    for w in windows:
        for obs in [{"bpm": w["sources"]["fitbit"], "context": w["context"], "hour": w["hour"]}]:
            profile = update_profile(profile, **obs)
    assert not math.isnan(profile.resting_mean())
