"""Synthetic files must load through the real-format adapters unchanged."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from hr_selection import config
from hr_selection.datasets import get_adapter


def test_bigideas_csv_schema(hr_synthetic_root):
    csv = hr_synthetic_root / "bigideas" / "BigIdeasLab_STEP.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    expected = ["ECG", "Apple Watch", "Empatica", "Garmin", "Fitbit", "Miband", "Biovotion", "ID", "Skin Tone", "Activity"]
    assert list(df.columns) == expected
    assert set(df["Activity"].unique()) <= {"Rest", "Activity", "Breathe", "Type"}
    assert df["Skin Tone"].between(1, 6).all()


def test_bigideas_adapter(hr_synthetic_root):
    adapter = get_adapter("bigideas", hr_synthetic_root / "bigideas")
    sessions = list(adapter.iter_sessions())
    assert len(sessions) >= 1
    s = sessions[0]
    assert s.dataset == "bigideas"
    assert set(s.sources) == set(config.DATASET_SOURCES["bigideas"])
    assert s.reference_hr.shape[0] > 0
    assert 1 <= s.metadata["skin_tone"] <= 6
    # HR-only: no raw waveforms
    assert all(src.raw == {} for src in s.sources.values())


def test_galaxyppg_tree_and_adapter(hr_synthetic_root):
    root = hr_synthetic_root / "galaxyppg"
    assert (root / "metadata.csv").exists()
    assert (root / "data" / "P01" / "PolarH10" / "ECG.csv").exists()
    assert (root / "data" / "P01" / "GalaxyWatch" / "PPG.csv").exists()
    assert (root / "data" / "P01" / "E4" / "BVP.csv").exists()

    adapter = get_adapter("galaxyppg", root)
    sessions = list(adapter.iter_sessions())
    assert len(sessions) >= 1
    s = sessions[0]
    assert set(s.sources) == {"galaxy_watch", "e4"}
    for src in s.sources.values():
        assert "ppg" in src.raw and src.raw["ppg"].data.shape[0] > 0
        assert "acc" in src.raw and src.raw["acc"].data.shape[1] == 3
    assert s.reference_hr.shape[0] > 0


def test_ppg_dalia_pickle_and_adapter(hr_synthetic_root):
    root = hr_synthetic_root / "ppg_dalia"
    pkls = sorted(root.glob("S*/S*.pkl"))
    assert pkls, "no DaLiA pickles written"
    # Must load with latin1 encoding like the real Python-2 pickles.
    with open(pkls[0], "rb") as f:
        data = pickle.load(f, encoding="latin1")
    assert "signal" in data and "label" in data
    assert "ECG" in data["signal"]["chest"]
    assert "BVP" in data["signal"]["wrist"]

    adapter = get_adapter("ppg_dalia", root)
    sessions = list(adapter.iter_sessions())
    assert len(sessions) >= 1
    s = sessions[0]
    assert set(s.sources) == {"dalia_wrist"}
    assert s.sources["dalia_wrist"].raw["ppg"].data.shape[0] > 0
    assert s.reference_hr.shape[0] > 0
    assert np.isfinite(s.reference_hr).any()
