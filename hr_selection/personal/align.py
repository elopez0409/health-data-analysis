"""Align multi-source unified heart rate into time windows for offset learning."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from hr_selection import config


def _to_pandas(df: Any) -> pd.DataFrame:
    """Accept pandas or Polars DataFrame."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return pd.DataFrame(df)


def align_heart_rate_windows(
    hr_df: Any,
    *,
    window_sec: float = config.WINDOW_SEC,
    shift_sec: float = config.SHIFT_SEC,
    hr_fs: float = config.HR_FS,
) -> list[dict]:
    """Turn long-format HR rows into aligned multi-source windows.

    Input columns: ``recorded_at``, ``source``, ``bpm``, optional ``context``.

    Returns list of dicts::
        {
            "t_start": float (seconds since epoch start of session),
            "recorded_at": datetime (window start),
            "sources": {provider: mean_bpm, ...},
            "context": str,
            "hour": int,
        }
    """
    df = _to_pandas(hr_df)
    if df.empty:
        return []

    required = {"recorded_at", "source", "bpm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HR dataframe missing columns: {missing}")

    df = df.copy()
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    df = df.sort_values("recorded_at")

    t_min = df["recorded_at"].min()
    df["t_sec"] = (df["recorded_at"] - t_min).dt.total_seconds()

    sources = sorted(df["source"].unique())
    bucket_sec = 1.0 / hr_fs

    per_source: dict[str, pd.Series] = {}
    for src in sources:
        sub = df[df["source"] == src].copy()
        sub["bucket"] = (sub["t_sec"] / bucket_sec).astype(int) * bucket_sec
        grouped = sub.groupby("bucket")["bpm"].mean()
        per_source[src] = grouped

    if not per_source:
        return []

    all_buckets = sorted(set().union(*(s.index for s in per_source.values())))
    if not all_buckets:
        return []

    t_max = max(all_buckets) + bucket_sec
    windows: list[dict] = []

    t0 = 0.0
    while t0 + window_sec <= t_max + 1e-6:
        t1 = t0 + window_sec
        src_hr: dict[str, float] = {}
        contexts: list[str] = []

        for src in sources:
            series = per_source[src]
            mask = (series.index >= t0) & (series.index < t1)
            vals = series[mask]
            if len(vals) == 0:
                continue
            mean_bpm = float(vals.mean())
            if not np.isnan(mean_bpm):
                src_hr[src] = mean_bpm

        if src_hr:
            win_start_dt = t_min + pd.Timedelta(seconds=t0)
            ctx = "unknown"
            if "context" in df.columns:
                ctx_rows = df[
                    (df["t_sec"] >= t0)
                    & (df["t_sec"] < t1)
                    & (df["source"].isin(src_hr.keys()))
                ]["context"].dropna()
                if len(ctx_rows):
                    ctx = str(ctx_rows.mode().iloc[0])

            windows.append(
                {
                    "t_start": t0,
                    "recorded_at": win_start_dt.to_pydatetime(),
                    "sources": src_hr,
                    "context": ctx,
                    "hour": win_start_dt.hour,
                }
            )

        t0 += shift_sec

    return windows


def compute_deltas(
    windows: list[dict],
    trusted_source: str,
) -> dict[str, list[float]]:
    """Compute per-source offset deltas vs trusted device for each window.

    Returns ``{source: [delta, ...]}`` excluding the trusted source (offset=0).
    """
    deltas: dict[str, list[float]] = {}

    for win in windows:
        src_hr = win["sources"]
        if trusted_source not in src_hr:
            continue
        trusted_hr = src_hr[trusted_source]
        for src, hr in src_hr.items():
            if src == trusted_source:
                continue
            delta = hr - trusted_hr
            deltas.setdefault(src, []).append(delta)

    return deltas


def extract_profile_observations(
    windows: list[dict],
    source: str | None = None,
) -> list[dict]:
    """Extract profile update observations from aligned windows."""
    obs: list[dict] = []
    for win in windows:
        src_hr = win["sources"]
        if source is not None:
            if source not in src_hr:
                continue
            obs.append(
                {
                    "bpm": src_hr[source],
                    "context": win.get("context"),
                    "hour": win.get("hour"),
                }
            )
        else:
            for src, bpm in src_hr.items():
                obs.append(
                    {
                        "bpm": bpm,
                        "context": win.get("context"),
                        "hour": win.get("hour"),
                    }
                )
    return obs
