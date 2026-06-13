"""Tests for HR calibration dataset generation and analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hr_selection.synthetic.calibration import (
    DEVICE_CHEST,
    DEVICE_WRIST,
    build_paired_nights,
    generate_calibration_data,
    write_calibration_dataset,
)


def test_generate_calibration_schema():
    readings, truth, manifest = generate_calibration_data(seed=42, n_users=5, n_nights=10)
    assert set(readings.columns) == {"user_id", "date", "device_type", "hr_reading", "is_resting"}
    assert set(truth.columns) == {"user_id", "true_baseline", "personal_offset"}
    assert manifest.n_users == 5
    assert manifest.n_nights == 10
    assert readings["is_resting"].all()
    assert set(readings["device_type"].unique()) <= {DEVICE_CHEST, DEVICE_WRIST}


def test_paired_nights_delta():
    readings, _, _ = generate_calibration_data(seed=123, n_users=3, n_nights=20)
    paired = build_paired_nights(readings)
    assert "delta" in paired.columns
    assert (paired["delta"] == paired["wrist"] - paired["chest"]).all()


def test_write_calibration_files(tmp_path):
    manifest = write_calibration_dataset(tmp_path, seed=42, n_users=5, n_nights=10)
    assert Path(manifest.paths["readings"]).exists()
    assert Path(manifest.paths["truth"]).exists()
    assert Path(manifest.paths["dictionary"]).exists()

    df = pd.read_csv(manifest.paths["readings"])
    assert len(df) > 0
