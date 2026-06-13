"""Normalize the real GalaxyPPG release into the canonical adapter layout.

The Zenodo GalaxyPPG dataset stores each participant under
``Dataset/Pxx/{GalaxyWatch,E4,PolarH10}/<SIGNAL>.csv`` but with device-specific
schemas that differ from the layout our ``GalaxyPPGAdapter`` reads:

- timestamps are epoch ms (Galaxy / Polar ``phoneTimestamp``) or epoch us (E4),
- value columns are named ``value`` / ``ppg`` / capitalised ``X,Y,Z``,
- the three devices use different clocks (we align on epoch seconds),
- missing Galaxy HR is encoded as ``hr == 0``.

This script resamples every stream onto a uniform per-signal grid sharing a
common time origin and writes the canonical CSVs the adapter expects::

    data/hr_real/galaxyppg/data/Pxx/GalaxyWatch/{HR,PPG,ACC}.csv
    data/hr_real/galaxyppg/data/Pxx/E4/{HR,BVP,ACC}.csv
    data/hr_real/galaxyppg/data/Pxx/PolarH10/HR.csv        (reference)

with columns ``timestamp`` (seconds from 0) + ``hr|ppg|bvp`` or ``x,y,z``.

Usage::

    python scripts/normalize_galaxyppg.py
    python scripts/normalize_galaxyppg.py --src data/hr_real/galaxyppg_extracted/Dataset
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "data" / "hr_real" / "galaxyppg_extracted" / "Dataset"
DEFAULT_DST = ROOT / "data" / "hr_real" / "galaxyppg"

# Canonical output sampling rates (Hz).
FS_GALAXY_PPG = 25.0
FS_GALAXY_ACC = 25.0
FS_E4_BVP = 64.0
FS_E4_ACC = 32.0
FS_HR = 1.0

# ACC unit conversions -> g (so sources are comparable for the model).
G = 9.80665
E4_ACC_COUNTS_PER_G = 64.0


def to_seconds(ts: np.ndarray) -> np.ndarray:
    """Convert an epoch timestamp array to seconds (auto-detect ms/us/ns)."""
    ts = np.asarray(ts, dtype=float)
    finite = ts[np.isfinite(ts)]
    if finite.size == 0:
        return ts
    m = np.median(np.abs(finite))
    if m > 1e17:  # nanoseconds
        return ts / 1e9
    if m > 1e14:  # microseconds
        return ts / 1e6
    if m > 1e11:  # milliseconds
        return ts / 1e3
    return ts  # already seconds


def _resample_wave(ts: np.ndarray, vals: np.ndarray, fs: float, t0: float, t1: float) -> np.ndarray:
    """Linear-interpolate a continuous waveform onto a uniform grid."""
    grid = np.arange(t0, t1, 1.0 / fs)
    ok = np.isfinite(ts) & np.isfinite(vals)
    if ok.sum() < 2:
        return np.full(grid.shape, np.nan)
    order = np.argsort(ts[ok])
    return np.interp(grid, ts[ok][order], vals[ok][order])


def _resample_hr(ts: np.ndarray, vals: np.ndarray, t0: float, t1: float, tol: float = 3.0) -> np.ndarray:
    """Nearest-sample (within ``tol`` s) resample of an HR series to 1 Hz, NaN in gaps."""
    grid = np.arange(t0, t1, 1.0 / FS_HR)
    ok = np.isfinite(ts) & np.isfinite(vals)
    if ok.sum() == 0:
        return np.full(grid.shape, np.nan)
    ts_ok = ts[ok]
    vals_ok = vals[ok]
    order = np.argsort(ts_ok)
    ts_ok, vals_ok = ts_ok[order], vals_ok[order]
    idx = np.searchsorted(ts_ok, grid)
    idx = np.clip(idx, 0, len(ts_ok) - 1)
    left = np.clip(idx - 1, 0, len(ts_ok) - 1)
    use_left = np.abs(ts_ok[left] - grid) < np.abs(ts_ok[idx] - grid)
    nearest = np.where(use_left, left, idx)
    out = vals_ok[nearest]
    out[np.abs(ts_ok[nearest] - grid) > tol] = np.nan
    return out


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _col(df: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _write(path: Path, **columns: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(path, index=False)


def _ts(df: pd.DataFrame, *cols: str) -> np.ndarray:
    return to_seconds(df[_col(df, *cols)].to_numpy())


def _vals(df: pd.DataFrame, *cols: str) -> np.ndarray:
    return pd.to_numeric(df[_col(df, *cols)], errors="coerce").to_numpy(dtype=float)


def normalize_participant(src_p: Path, dst_p: Path) -> dict:
    """Normalize one participant; returns a small summary dict.

    Clock handling: Galaxy + E4 device ``timestamp`` columns share one epoch
    clock (their start offset is a *real* difference). Polar ``phoneTimestamp``
    is a different phone clock with the same session duration but a large
    constant offset, so we map Polar by relative start onto the master clock,
    assuming all devices began recording at ~the same real instant.
    """
    summary = {"participant": src_p.name, "devices": []}

    polar_hr = _read(src_p / "PolarH10" / "HR.csv")
    if polar_hr is None or len(polar_hr) == 0:
        summary["error"] = "no PolarH10/HR.csv"
        return summary

    gw_ppg = _read(src_p / "GalaxyWatch" / "PPG.csv")
    gw_hr = _read(src_p / "GalaxyWatch" / "HR.csv")
    gw_acc = _read(src_p / "GalaxyWatch" / "ACC.csv")
    have_galaxy = gw_ppg is not None and len(gw_ppg) > 0

    e4_bvp = _read(src_p / "E4" / "BVP.csv")
    e4_hr = _read(src_p / "E4" / "HR.csv")
    e4_acc = _read(src_p / "E4" / "ACC.csv")
    have_e4 = e4_bvp is not None and len(e4_bvp) > 0

    # Master clock origin = earliest start among Galaxy/E4 (shared epoch clock).
    master_starts = []
    if have_galaxy:
        master_starts.append(_ts(gw_ppg, "timestamp", "dataReceived").min())
    if have_e4:
        master_starts.append(_ts(e4_bvp, "timestamp").min())
    if not master_starts:
        summary["error"] = "no Galaxy/E4 source device"
        return summary
    master_start = min(master_starts)

    # Map Polar onto the master clock by relative start.
    polar_ts_raw = _ts(polar_hr, "phoneTimestamp", "timestamp")
    polar_offset = polar_ts_raw.min() - master_start
    polar_ts = polar_ts_raw - polar_offset
    polar_v = _vals(polar_hr, "hr")
    polar_v[polar_v <= 0] = np.nan

    # Overlap window across present streams (all on master clock now).
    starts = [polar_ts.min()]
    ends = [polar_ts.max()]
    if have_galaxy:
        g = _ts(gw_ppg, "timestamp", "dataReceived"); starts.append(g.min()); ends.append(g.max())
    if have_e4:
        e = _ts(e4_bvp, "timestamp"); starts.append(e.min()); ends.append(e.max())
    abs_t0, abs_t1 = max(starts), min(ends)
    if not np.isfinite(abs_t0) or abs_t1 - abs_t0 < 30:
        summary["error"] = f"insufficient overlap ({abs_t1 - abs_t0:.1f}s)"
        return summary
    dur = abs_t1 - abs_t0
    summary["overlap_sec"] = round(dur, 1)

    def rel(ts: np.ndarray) -> np.ndarray:
        return ts - abs_t0

    # Reference: Polar H10 HR -> 1 Hz.
    ref = _resample_hr(rel(polar_ts), polar_v, 0.0, dur)
    _write(dst_p / "PolarH10" / "HR.csv", timestamp=np.arange(len(ref)) / FS_HR, hr=ref)
    summary["ref_hr_n"] = int(np.isfinite(ref).sum())

    if have_galaxy:
        ppg = _resample_wave(rel(_ts(gw_ppg, "timestamp", "dataReceived")), _vals(gw_ppg, "ppg", "value"), FS_GALAXY_PPG, 0.0, dur)
        _write(dst_p / "GalaxyWatch" / "PPG.csv", timestamp=np.arange(len(ppg)) / FS_GALAXY_PPG, ppg=ppg)
        if gw_acc is not None and len(gw_acc) > 0:
            a = rel(_ts(gw_acc, "timestamp", "dataReceived"))
            ax = _resample_wave(a, _vals(gw_acc, "x") / G, FS_GALAXY_ACC, 0.0, dur)
            ay = _resample_wave(a, _vals(gw_acc, "y") / G, FS_GALAXY_ACC, 0.0, dur)
            az = _resample_wave(a, _vals(gw_acc, "z") / G, FS_GALAXY_ACC, 0.0, dur)
            _write(dst_p / "GalaxyWatch" / "ACC.csv", timestamp=np.arange(len(ax)) / FS_GALAXY_ACC, x=ax, y=ay, z=az)
        if gw_hr is not None and len(gw_hr) > 0:
            hv = _vals(gw_hr, "hr"); hv[hv <= 0] = np.nan
            hr = _resample_hr(rel(_ts(gw_hr, "timestamp", "dataReceived")), hv, 0.0, dur)
        else:
            hr = np.full(len(ref), np.nan)
        _write(dst_p / "GalaxyWatch" / "HR.csv", timestamp=np.arange(len(hr)) / FS_HR, hr=hr)
        summary["devices"].append("GalaxyWatch")

    if have_e4:
        bvp = _resample_wave(rel(_ts(e4_bvp, "timestamp")), _vals(e4_bvp, "bvp", "value"), FS_E4_BVP, 0.0, dur)
        _write(dst_p / "E4" / "BVP.csv", timestamp=np.arange(len(bvp)) / FS_E4_BVP, bvp=bvp)
        if e4_acc is not None and len(e4_acc) > 0:
            a = rel(_ts(e4_acc, "timestamp"))
            ax = _resample_wave(a, _vals(e4_acc, "x") / E4_ACC_COUNTS_PER_G, FS_E4_ACC, 0.0, dur)
            ay = _resample_wave(a, _vals(e4_acc, "y") / E4_ACC_COUNTS_PER_G, FS_E4_ACC, 0.0, dur)
            az = _resample_wave(a, _vals(e4_acc, "z") / E4_ACC_COUNTS_PER_G, FS_E4_ACC, 0.0, dur)
            _write(dst_p / "E4" / "ACC.csv", timestamp=np.arange(len(ax)) / FS_E4_ACC, x=ax, y=ay, z=az)
        if e4_hr is not None and len(e4_hr) > 0:
            hv = _vals(e4_hr, "hr", "value"); hv[hv <= 0] = np.nan
            hr = _resample_hr(rel(_ts(e4_hr, "timestamp")), hv, 0.0, dur)
        else:
            hr = np.full(len(ref), np.nan)
        _write(dst_p / "E4" / "HR.csv", timestamp=np.arange(len(hr)) / FS_HR, hr=hr)
        summary["devices"].append("E4")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Real GalaxyPPG Dataset/ dir.")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="Canonical output root.")
    args = parser.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source not found: {args.src}\nRun scripts/fetch_hr_datasets.py first.")

    data_out = args.dst / "data"
    data_out.mkdir(parents=True, exist_ok=True)

    meta = args.src / "Meta.csv"
    if meta.exists():
        shutil.copy2(meta, args.dst / "metadata.csv")

    participants = sorted(p for p in args.src.glob("P*") if p.is_dir())
    print(f"Normalizing {len(participants)} participants -> {args.dst}")
    n_ok = 0
    for p in participants:
        summary = normalize_participant(p, data_out / p.name)
        if "error" in summary:
            print(f"  {p.name}: SKIP ({summary['error']})")
        else:
            print(f"  {p.name}: {summary['overlap_sec']}s, devices={summary['devices']}, ref_hr={summary['ref_hr_n']}")
            n_ok += 1
    print(f"\nDone. {n_ok}/{len(participants)} participants normalized.")


if __name__ == "__main__":
    main()
