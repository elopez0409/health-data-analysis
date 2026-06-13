"""Feature-table correctness: best-source label + NaN handling."""

from __future__ import annotations

import numpy as np

from hr_selection import config
from hr_selection.features.build import all_feature_columns, build_feature_table


def test_bigideas_table_shape_and_columns(hr_synthetic_root):
    df = build_feature_table(["bigideas"], root=hr_synthetic_root / "bigideas")
    assert len(df) > 0
    for col in all_feature_columns():
        assert col in df.columns
    assert df["reference_hr"].notna().all()
    assert df["label"].between(0, len(config.CANONICAL_SOURCES) - 1).all()


def test_label_is_closest_available_source(hr_synthetic_root):
    df = build_feature_table(["bigideas"], root=hr_synthetic_root / "bigideas")
    sample = df.sample(min(50, len(df)), random_state=0)
    for _, row in sample.iterrows():
        ref = row["reference_hr"]
        best_src = None
        best_err = np.inf
        for src in config.CANONICAL_SOURCES:
            if row.get(f"{src}__missing", 1.0) == 0.0:
                err = abs(row[f"{src}__hr"] - ref)
                if err < best_err:
                    best_err = err
                    best_src = src
        assert config.SOURCE_INDEX[best_src] == int(row["label"])


def test_hr_only_dataset_has_nan_quality(hr_synthetic_root):
    df = build_feature_table(["bigideas"], root=hr_synthetic_root / "bigideas")
    # BigIdeas is HR-only -> signal-quality features are all NaN.
    assert df["fitbit__q_snr"].isna().all()
    # but the HR feature is present for available sources.
    assert df["fitbit__hr"].notna().any()


def test_raw_dataset_has_quality_features(hr_synthetic_root):
    df = build_feature_table(["galaxyppg"], root=hr_synthetic_root / "galaxyppg")
    assert df["galaxy_watch__q_snr"].notna().any()
    assert df["galaxy_watch__q_hr_est"].notna().any()
    assert df["e4__q_acc_mag"].notna().any()


def test_missing_flag_consistency(hr_synthetic_root):
    df = build_feature_table(["bigideas"], root=hr_synthetic_root / "bigideas")
    for src in config.DATASET_SOURCES["bigideas"]:
        hr_nan = df[f"{src}__hr"].isna()
        flagged = df[f"{src}__missing"] == 1.0
        assert (hr_nan == flagged).all()
