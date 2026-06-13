"""Build the per-window multiclass feature table from dataset adapters.

One row per window. For every canonical source we emit HR + agreement +
stability + signal-quality features (NaN where the source is absent or HR-only).
The label is the canonical index of the *best available* source, i.e.
``argmin_s |hr_s - reference_hr|`` over sources present in the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hr_selection import config
from hr_selection.datasets.base import DatasetAdapter, get_adapter
from hr_selection.features.signal_quality import QUALITY_KEYS, window_quality
from hr_selection.priors.lookup import bucket_activity, source_prior

BASE_KEYS = ["hr", "missing", "dev_med", "roll_std"]
PRIOR_KEYS = ["prior_mae", "prior_bias_abs", "prior_trust", "prior_rank"]
PER_SOURCE_KEYS = BASE_KEYS + QUALITY_KEYS + PRIOR_KEYS
CONTEXT_FEATURES = ["activity", "skin_tone"]

META_COLUMNS = [
    "dataset",
    "subject_id",
    "group",
    "window_idx",
    "t_start",
    "reference_hr",
    "n_available",
    "label",
]


def feature_columns(use_priors: bool = True) -> list[str]:
    """Numeric per-source feature columns (consistent across datasets)."""
    keys = PER_SOURCE_KEYS if use_priors else BASE_KEYS + QUALITY_KEYS
    cols: list[str] = []
    for src in config.CANONICAL_SOURCES:
        for key in keys:
            cols.append(f"{src}__{key}")
    return cols


def all_feature_columns(use_priors: bool = True) -> list[str]:
    """All model input columns (per-source numeric + context)."""
    return feature_columns(use_priors=use_priors) + CONTEXT_FEATURES


def _source_window_hr(session, src_key, t0, t1, quality) -> float:
    src = session.sources.get(src_key)
    if src is None:
        return float("nan")
    if src.hr.size > 0:
        hr = src.hr_at(t0, t1)
        if not np.isnan(hr):
            return hr
    return quality.get("q_hr_est", float("nan"))


def _session_windows(session) -> list[dict]:
    win = config.WINDOW_SEC
    shift = config.SHIFT_SEC
    duration = session.duration_sec
    rows: list[dict] = []
    k = 0
    t0 = 0.0
    while t0 + win <= duration + 1e-6:
        t1 = t0 + win
        ref_hr = session.reference_hr_at(t0, t1)
        if np.isnan(ref_hr):
            t0 = (k + 1) * shift
            k += 1
            continue

        # Per-source quality (raw datasets only) + HR.
        quality_by_src: dict[str, dict] = {}
        hr_by_src: dict[str, float] = {}
        for src_key in session.canonical_sources:
            src = session.sources.get(src_key)
            q = {key: float("nan") for key in QUALITY_KEYS}
            if src is not None and "ppg" in src.raw:
                ppg_ch = src.raw["ppg"]
                acc_ch = src.raw.get("acc")
                ppg_seg = ppg_ch.slice_seconds(t0, t1)
                acc_seg = acc_ch.slice_seconds(t0, t1) if acc_ch is not None else np.array([])
                q = window_quality(ppg_seg, ppg_ch.fs, acc_seg, acc_ch.fs if acc_ch is not None else 1.0)
            quality_by_src[src_key] = q
            hr_by_src[src_key] = _source_window_hr(session, src_key, t0, t1, q)

        available = [s for s, hr in hr_by_src.items() if not np.isnan(hr)]
        if not available:
            t0 = (k + 1) * shift
            k += 1
            continue

        cross_med = float(np.nanmedian([hr_by_src[s] for s in available]))
        best = min(available, key=lambda s: abs(hr_by_src[s] - ref_hr))
        label = config.SOURCE_INDEX[best]

        activity = session.activity_at(t0, t1)
        skin_tone = float(session.metadata.get("skin_tone", float("nan")))
        context = bucket_activity(activity)

        row: dict = {
            "dataset": session.dataset,
            "subject_id": session.subject_id,
            "group": f"{session.dataset}:{session.subject_id}",
            "window_idx": k,
            "t_start": t0,
            "reference_hr": ref_hr,
            "n_available": len(available),
            "label": label,
            "activity": activity,
            "skin_tone": skin_tone,
        }
        for src_key in config.CANONICAL_SOURCES:
            hr = hr_by_src.get(src_key, float("nan"))
            q = quality_by_src.get(src_key, {key: float("nan") for key in QUALITY_KEYS})
            missing = 1.0 if np.isnan(hr) else 0.0
            dev = float("nan") if np.isnan(hr) else hr - cross_med
            row[f"{src_key}__hr"] = hr
            row[f"{src_key}__missing"] = missing
            row[f"{src_key}__dev_med"] = dev
            row[f"{src_key}__roll_std"] = float("nan")  # filled after assembly
            for key in QUALITY_KEYS:
                row[f"{src_key}__{key}"] = q.get(key, float("nan"))
            prior = source_prior(src_key, context, skin_tone)
            for key in PRIOR_KEYS:
                row[f"{src_key}__{key}"] = prior[key]
        rows.append(row)

        t0 = (k + 1) * shift
        k += 1
    return rows


def _add_rolling_std(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Temporal rolling std of each source's HR within a session."""
    if df.empty:
        return df
    df = df.sort_values(["group", "window_idx"]).reset_index(drop=True)
    for src in config.CANONICAL_SOURCES:
        hr_col = f"{src}__hr"
        std_col = f"{src}__roll_std"
        df[std_col] = (
            df.groupby("group")[hr_col]
            .transform(lambda s: s.rolling(window, min_periods=2).std())
        )
    return df


def build_feature_table(
    datasets: list[str] | str,
    source: str = "synthetic",
    root: str | None = None,
) -> pd.DataFrame:
    """Build the multiclass feature table for one or more datasets.

    ``root`` overrides the dataset directory (e.g. for real data). For synthetic
    data the default ``config.DATASET_DIRS`` are used.
    """
    if isinstance(datasets, str):
        datasets = config.ALL_DATASETS if datasets == "all" else [datasets]

    all_rows: list[dict] = []
    for ds in datasets:
        ds_root = root if root is not None else config.DATASET_DIRS[ds]
        adapter: DatasetAdapter = get_adapter(ds, ds_root)
        for session in adapter.iter_sessions():
            all_rows.extend(_session_windows(session))

    if not all_rows:
        raise RuntimeError(f"No windows produced for datasets={datasets} (source={source}).")

    df = pd.DataFrame(all_rows)
    df = _add_rolling_std(df)
    return df
