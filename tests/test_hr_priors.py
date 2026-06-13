"""Paper-derived prior knowledge base, lookup, features, and baseline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hr_selection import config
from hr_selection.evaluate import evaluate
from hr_selection.features.build import PRIOR_KEYS, build_feature_table, feature_columns
from hr_selection.models.prior_baseline import prior_proba
from hr_selection.priors.knowledge import DEVICE_FAMILY, PAPERS, PRIORS, SKIN_TONE_ADJ
from hr_selection.priors.lookup import bucket_activity, source_prior


def test_every_canonical_source_has_family():
    for src in config.CANONICAL_SOURCES:
        assert src in DEVICE_FAMILY, f"missing DEVICE_FAMILY for {src}"


def test_all_papers_have_trust_weight():
    for pid, meta in PAPERS.items():
        assert 0 < meta["trust_weight"] <= 1.0, pid
        assert "citation" in meta


def test_dalia_activities_bucket():
    for act in ["sitting", "walking", "stairs", "soccer", "cycling", "driving", "transient"]:
        ctx = bucket_activity(act)
        assert ctx in ("daily_living", "walking", "exercise", "unknown")


def test_bigideas_activities_bucket():
    assert bucket_activity("Rest") == "rest"
    assert bucket_activity("Breathe") == "rest"
    assert bucket_activity("Activity") == "exercise"
    assert bucket_activity("Type") == "daily_living"


def test_source_prior_finite_for_covered_families():
    for src in config.CANONICAL_SOURCES:
        p = source_prior(src, "rest", skin_tone=3.0)
        for key in PRIOR_KEYS:
            assert key in p
            assert math.isfinite(p[key])


def test_skin_tone_increases_prior_mae():
    base = source_prior("empatica", "rest", skin_tone=1.0)["prior_mae"]
    dark = source_prior("empatica", "rest", skin_tone=6.0)["prior_mae"]
    assert dark > base
    assert dark - base == pytest.approx(SKIN_TONE_ADJ[6] - SKIN_TONE_ADJ[1])


def test_feature_table_has_prior_columns(hr_synthetic_root):
    df = build_feature_table(["bigideas"], root=hr_synthetic_root / "bigideas")
    for key in PRIOR_KEYS:
        col = f"fitbit__{key}"
        assert col in df.columns
        assert df[col].notna().all()


def test_prior_constant_within_context_skin(hr_synthetic_root):
    df = build_feature_table(["bigideas"], root=hr_synthetic_root / "bigideas")
    sub = df[(df["activity"] == "Rest") & (df["skin_tone"] == 1.0)]
    if len(sub) > 1:
        vals = sub["apple_watch__prior_mae"].unique()
        assert len(vals) == 1


def test_feature_columns_ablation():
    with_priors = set(feature_columns(use_priors=True))
    without = set(feature_columns(use_priors=False))
    assert "apple_watch__prior_mae" in with_priors
    assert "apple_watch__prior_mae" not in without


def test_prior_baseline_proba_sums_to_one(hr_synthetic_root):
    df = build_feature_table(["galaxyppg"], root=hr_synthetic_root / "galaxyppg")
    proba = prior_proba(df)
    assert proba.shape == (len(df), len(config.CANONICAL_SOURCES))
    for i in range(len(df)):
        row_sum = proba[i].sum()
        if row_sum > 0:
            assert row_sum == pytest.approx(1.0, abs=1e-5)


def test_evaluate_exposes_paper_prior(hr_synthetic_root):
    df = build_feature_table(["galaxyppg"], root=hr_synthetic_root / "galaxyppg")
    proba = prior_proba(df)
    metrics = evaluate(df, proba)
    assert "paper_prior" in metrics["hr_mae"]
    assert math.isfinite(metrics["hr_mae"]["paper_prior"])


def test_kb_has_records_for_bigideas_families():
    families = {DEVICE_FAMILY[s] for s in config.DATASET_SOURCES["bigideas"]}
    covered = {r.family for r in PRIORS}
    assert families.issubset(covered)
