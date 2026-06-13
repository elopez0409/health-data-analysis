"""Central configuration for the HR source-selection pipeline.

All windowing, sample-rate, canonical-source, and path constants live here so
the synthetic generators, dataset adapters, feature builder, and models stay
coherent.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Synthetic (and drop-in real) raw datasets live here, one sub-dir per format.
HR_RAW_DIR = DATA_DIR / "hr_raw"
HR_OUT_DIR = DATA_DIR / "hr_out"  # metrics / model artifacts

DATASET_DIRS = {
    "bigideas": HR_RAW_DIR / "bigideas",
    "galaxyppg": HR_RAW_DIR / "galaxyppg",
    "ppg_dalia": HR_RAW_DIR / "ppg_dalia",
}

# Real PhysioNet BigIdeasLab_STEP CSV (Bent et al. 2020; 6 devices + ECG + skin tone).
BIGIDEAS_REAL_CSV = (
    ROOT
    / "physionet.org"
    / "files"
    / "bigideaslab-step-hr-smartwatch"
    / "1.0"
    / "deidentified_data.csv"
)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Windowing (matches PPG-DaLiA's native label cadence)
# --------------------------------------------------------------------------
WINDOW_SEC = 8.0
SHIFT_SEC = 2.0

# HR series for all sources/reference are normalised to this rate before
# windowing (1 Hz: one HR value per second).
HR_FS = 1.0

# BigIdeasLab_STEP has no timestamp column; rows are assumed uniformly spaced.
BIGIDEAS_ROW_HZ = 1.0

# --------------------------------------------------------------------------
# Canonical source space (union across all datasets)
#
# The multiclass label = index into CANONICAL_SOURCES.  Per window, only the
# *available* sources can be the label; missing-source features are NaN.
# Reference signals (ECG / Polar H10) are NOT selectable sources.
# --------------------------------------------------------------------------
CANONICAL_SOURCES: list[str] = [
    # BigIdeasLab_STEP devices
    "apple_watch",
    "empatica",
    "garmin",
    "fitbit",
    "miband",
    "biovotion",
    # GalaxyPPG devices
    "galaxy_watch",
    "e4",
    # PPG-DaLiA (single wrist device)
    "dalia_wrist",
]

SOURCE_INDEX: dict[str, int] = {s: i for i, s in enumerate(CANONICAL_SOURCES)}

# Which canonical sources each dataset provides.
DATASET_SOURCES: dict[str, list[str]] = {
    "bigideas": [
        "apple_watch",
        "empatica",
        "garmin",
        "fitbit",
        "miband",
        "biovotion",
    ],
    "galaxyppg": ["galaxy_watch", "e4"],
    "ppg_dalia": ["dalia_wrist"],
}

# Datasets that carry raw waveforms (PPG/BVP + ACC) -> signal-quality features
# and the deep track apply. BigIdeas is HR-only.
RAW_SIGNAL_DATASETS = {"galaxyppg", "ppg_dalia"}

ALL_DATASETS = ["bigideas", "galaxyppg", "ppg_dalia"]

# --------------------------------------------------------------------------
# Native sample rates for synthesized raw waveforms (Hz)
# --------------------------------------------------------------------------
FS = {
    # GalaxyPPG
    "galaxy_ppg": 25.0,
    "galaxy_acc": 32.0,
    "galaxy_hr": 1.0,
    "e4_bvp": 64.0,
    "e4_acc": 32.0,
    "e4_hr": 1.0,
    "e4_temp": 4.0,
    "e4_eda": 4.0,
    "polar_ecg": 130.0,
    "polar_hr": 1.0,
    "polar_acc": 32.0,
    # PPG-DaLiA
    "dalia_ecg": 700.0,
    "dalia_bvp": 64.0,
    "dalia_acc": 32.0,
    "dalia_eda": 4.0,
    "dalia_temp": 4.0,
}

# PPG band for FFT-based HR estimation (Hz) -> 42-210 bpm
PPG_BAND = (0.7, 3.5)
