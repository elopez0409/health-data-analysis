"""Evaluation metrics for HR source selection.

Given per-window class probabilities (masked to the sources actually available
in each window) and the feature table, compute:

- selection accuracy (top-1 / top-2) and macro-F1
- downstream HR MAE of the selected source vs:
    * oracle best source (lower bound)
    * always-pick-single-device baselines
    * cross-source median
- per-activity and per-dataset breakdowns + a confusion matrix
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from hr_selection import config
from hr_selection.priors.lookup import bucket_activity, source_prior


def _availability(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    n_classes = len(config.CANONICAL_SOURCES)
    avail = np.zeros((n, n_classes), dtype=bool)
    for j, src in enumerate(config.CANONICAL_SOURCES):
        col = f"{src}__missing"
        if col in df.columns:
            avail[:, j] = df[col].to_numpy() == 0
    return avail


def _hr_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [f"{src}__hr" for src in config.CANONICAL_SOURCES]
    return df[cols].to_numpy(dtype=float)


def _paper_prior_top1(df: pd.DataFrame, avail: np.ndarray) -> np.ndarray:
    """Pick the available source with lowest paper-derived prior MAE per window."""
    n = len(df)
    n_classes = len(config.CANONICAL_SOURCES)
    prior_mae = np.full((n, n_classes), np.inf, dtype=float)

    has_prior_cols = any(f"{src}__prior_mae" in df.columns for src in config.CANONICAL_SOURCES)
    if has_prior_cols:
        for j, src in enumerate(config.CANONICAL_SOURCES):
            col = f"{src}__prior_mae"
            if col in df.columns:
                prior_mae[:, j] = df[col].to_numpy(dtype=float)
    else:
        for i, row in df.iterrows():
            context = bucket_activity(row.get("activity"))
            skin = row.get("skin_tone", float("nan"))
            for j, src in enumerate(config.CANONICAL_SOURCES):
                prior_mae[i, j] = source_prior(src, context, skin)["prior_mae"]

    masked_prior = np.where(avail, prior_mae, np.inf)
    return np.argmin(masked_prior, axis=1)


def masked_predictions(oof_proba: np.ndarray, avail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 and top-2 predictions restricted to available sources."""
    masked = np.where(avail, oof_proba, -np.inf)
    top1 = np.argmax(masked, axis=1)
    order = np.argsort(masked, axis=1)[:, ::-1]
    top2 = order[:, :2]
    return top1, top2


def evaluate(df: pd.DataFrame, oof_proba: np.ndarray) -> dict:
    """Compute the full metric suite. Returns a JSON-serialisable dict."""
    n = len(df)
    idx = np.arange(n)
    avail = _availability(df)
    hr = _hr_matrix(df)
    ref = df["reference_hr"].to_numpy(dtype=float)
    y = df["label"].to_numpy()

    top1, top2 = masked_predictions(oof_proba, avail)

    top1_acc = float(np.mean(top1 == y))
    top2_acc = float(np.mean([yi in t2 for yi, t2 in zip(y, top2)]))
    present_labels = np.unique(y)
    macro_f1 = float(f1_score(y, top1, labels=present_labels, average="macro", zero_division=0))

    # Downstream HR MAE
    sel_hr = hr[idx, top1]
    oracle_hr = hr[idx, y]
    mae_model = float(np.nanmean(np.abs(sel_hr - ref)))
    mae_oracle = float(np.nanmean(np.abs(oracle_hr - ref)))

    masked_hr = np.where(avail, hr, np.nan)
    median_hr = np.nanmedian(masked_hr, axis=1)
    mae_median = float(np.nanmean(np.abs(median_hr - ref)))

    prior_top1 = _paper_prior_top1(df, avail)
    prior_hr = hr[idx, prior_top1]
    mae_paper_prior = float(np.nanmean(np.abs(prior_hr - ref)))

    single_device_mae: dict[str, float] = {}
    for j, src in enumerate(config.CANONICAL_SOURCES):
        mask = avail[:, j]
        if mask.sum() == 0:
            continue
        single_device_mae[src] = float(np.nanmean(np.abs(hr[mask, j] - ref[mask])))
    best_single = min(single_device_mae.values()) if single_device_mae else float("nan")
    best_single_src = (
        min(single_device_mae, key=single_device_mae.get) if single_device_mae else None
    )

    # Per-activity selection accuracy
    per_activity = {}
    if "activity" in df.columns:
        for act, sub in df.assign(_pred=top1).groupby("activity"):
            mask = sub.index.to_numpy()
            per_activity[str(act)] = {
                "n": int(len(sub)),
                "top1_acc": float(np.mean(top1[mask] == y[mask])),
            }

    # Per-dataset breakdown
    per_dataset = {}
    for ds, sub in df.assign(_pred=top1).groupby("dataset"):
        mask = sub.index.to_numpy()
        sel = hr[mask, top1[mask]]
        orc = hr[mask, y[mask]]
        prior_sel = hr[mask, prior_top1[mask]]
        per_dataset[str(ds)] = {
            "n": int(len(sub)),
            "top1_acc": float(np.mean(top1[mask] == y[mask])),
            "mae_model": float(np.nanmean(np.abs(sel - ref[mask]))),
            "mae_oracle": float(np.nanmean(np.abs(orc - ref[mask]))),
            "mae_median": float(np.nanmean(np.abs(median_hr[mask] - ref[mask]))),
            "mae_paper_prior": float(np.nanmean(np.abs(prior_sel - ref[mask]))),
        }

    cm = confusion_matrix(y, top1, labels=list(range(len(config.CANONICAL_SOURCES))))

    improvement_vs_single = best_single - mae_model
    improvement_vs_median = mae_median - mae_model

    return {
        "n_windows": int(n),
        "n_classes_present": int(present_labels.size),
        "selection": {
            "top1_accuracy": top1_acc,
            "top2_accuracy": top2_acc,
            "macro_f1": macro_f1,
        },
        "hr_mae": {
            "model": mae_model,
            "oracle": mae_oracle,
            "cross_source_median": mae_median,
            "paper_prior": mae_paper_prior,
            "best_single_device": best_single,
            "best_single_device_name": best_single_src,
            "single_device": single_device_mae,
            "improvement_vs_best_single": improvement_vs_single,
            "improvement_vs_median": improvement_vs_median,
        },
        "beats_baselines": bool(mae_model < best_single and mae_model < mae_median),
        "per_activity": per_activity,
        "per_dataset": per_dataset,
        "confusion_matrix": cm.tolist(),
        "canonical_sources": config.CANONICAL_SOURCES,
    }


def save_metrics(metrics: dict, out_dir: Path | str, tag: str = "classical") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{tag}.json"
    path.write_text(json.dumps(metrics, indent=2, default=str))
    return path
