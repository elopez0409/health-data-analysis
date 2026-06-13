"""Smoke test: training runs and beats the single-device + median baselines."""

from __future__ import annotations

from hr_selection.evaluate import evaluate
from hr_selection.features.build import build_feature_table
from hr_selection.models.classical import train_classical


def test_classical_smoke_beats_baselines(hr_synthetic_root):
    df = build_feature_table(["bigideas"], root=hr_synthetic_root / "bigideas")
    res = train_classical(df, backend="hist", max_iter=50)
    metrics = evaluate(df, res["oof_proba"])

    sel = metrics["selection"]
    mae = metrics["hr_mae"]

    # Selection clearly better than random over the available label space.
    assert sel["top1_accuracy"] > 0.4
    assert sel["top2_accuracy"] >= sel["top1_accuracy"]

    # Downstream HR MAE beats both baselines and approaches the oracle.
    assert mae["model"] < mae["best_single_device"]
    assert mae["model"] < mae["cross_source_median"]
    assert mae["model"] >= mae["oracle"] - 1e-6
    assert metrics["beats_baselines"] is True


def test_degenerate_single_source(hr_synthetic_root):
    # PPG-DaLiA has one wrist source -> degenerate selection, must not crash.
    df = build_feature_table(["ppg_dalia"], root=hr_synthetic_root / "ppg_dalia")
    res = train_classical(df, backend="hist", max_iter=50)
    assert res["degenerate"] is True
    metrics = evaluate(df, res["oof_proba"])
    assert metrics["selection"]["top1_accuracy"] == 1.0
