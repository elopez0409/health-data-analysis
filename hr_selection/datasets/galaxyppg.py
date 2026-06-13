"""GalaxyPPG adapter.

Real format: directory tree ``data/Pxx/{E4,GalaxyWatch,PolarH10}/<SIGNAL>.csv``
plus a top-level metadata CSV. Each signal CSV has a ``timestamp`` column + one
or more value columns. PolarH10 is the reference (ECG + HR).

Canonical sources: ``galaxy_watch`` (PPG ~25 Hz + HR 1 Hz + ACC ~32 Hz) and
``e4`` (BVP 64 Hz + HR + ACC 32 Hz + TEMP + EDA). Raw PPG/BVP and ACC are
exposed as ``raw['ppg']`` / ``raw['acc']`` for the signal-quality features and
deep track.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from hr_selection import config
from hr_selection.datasets.base import DatasetAdapter
from hr_selection.datasets.schema import RawSession, SignalChannel, SourceSignal


def _infer_fs(ts: np.ndarray, default: float) -> float:
    if ts.shape[0] < 2:
        return default
    dt = np.median(np.diff(ts))
    if dt <= 0:
        return default
    return float(1.0 / dt)


def _read_signal(path: Path, value_cols: list[str], default_fs: float) -> tuple[np.ndarray, float]:
    """Read a signal CSV -> (data array (N,) or (N,k), fs)."""
    df = pd.read_csv(path)
    ts = df["timestamp"].to_numpy(dtype=float) if "timestamp" in df.columns else np.arange(len(df))
    fs = _infer_fs(ts, default_fs)
    cols = [c for c in value_cols if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if c != "timestamp"]
    arr = df[cols].to_numpy(dtype=float)
    if arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr, fs


class GalaxyPPGAdapter(DatasetAdapter):
    name = "galaxyppg"
    canonical_sources = config.DATASET_SOURCES["galaxyppg"]

    def _participant_dirs(self) -> list[Path]:
        base = self.root / "data" if (self.root / "data").is_dir() else self.root
        return sorted(p for p in base.glob("P*") if p.is_dir())

    def session_keys(self) -> list[str]:
        """Stable per-session keys (participant dirs) for parallel loading."""
        return [str(p) for p in self._participant_dirs()]

    def iter_sessions(self) -> Iterator[RawSession]:
        for pdir in self._participant_dirs():
            yield self.load_session(str(pdir))

    def load_session(self, key: str) -> RawSession:
        pdir = Path(key)
        sid = pdir.name

        # Reference: PolarH10 HR (1 Hz)
        polar_hr_path = pdir / "PolarH10" / "HR.csv"
        ref_hr, ref_fs = _read_signal(polar_hr_path, ["hr"], config.FS["polar_hr"])

        sources: dict[str, SourceSignal] = {}

        # GalaxyWatch
        gw_dir = pdir / "GalaxyWatch"
        if gw_dir.is_dir():
            hr, hr_fs = _read_signal(gw_dir / "HR.csv", ["hr"], config.FS["galaxy_hr"])
            ppg, ppg_fs = _read_signal(gw_dir / "PPG.csv", ["ppg"], config.FS["galaxy_ppg"])
            acc, acc_fs = _read_signal(gw_dir / "ACC.csv", ["x", "y", "z"], config.FS["galaxy_acc"])
            sources["galaxy_watch"] = SourceSignal(
                name="galaxy_watch",
                hr=hr,
                hr_fs=hr_fs,
                raw={
                    "ppg": SignalChannel("PPG", ppg_fs, ppg),
                    "acc": SignalChannel("ACC", acc_fs, acc),
                },
            )

        # E4
        e4_dir = pdir / "E4"
        if e4_dir.is_dir():
            hr, hr_fs = _read_signal(e4_dir / "HR.csv", ["hr"], config.FS["e4_hr"])
            bvp, bvp_fs = _read_signal(e4_dir / "BVP.csv", ["bvp"], config.FS["e4_bvp"])
            acc, acc_fs = _read_signal(e4_dir / "ACC.csv", ["x", "y", "z"], config.FS["e4_acc"])
            sources["e4"] = SourceSignal(
                name="e4",
                hr=hr,
                hr_fs=hr_fs,
                raw={
                    "ppg": SignalChannel("BVP", bvp_fs, bvp),
                    "acc": SignalChannel("ACC", acc_fs, acc),
                },
            )

        return RawSession(
            dataset=self.name,
            subject_id=str(sid),
            duration_sec=ref_hr.shape[0] / ref_fs,
            reference_hr=ref_hr,
            reference_hr_fs=ref_fs,
            sources=sources,
            activity=None,
            activity_fs=1.0,
            metadata={"skin_tone": float("nan")},
            canonical_sources=list(self.canonical_sources),
        )
