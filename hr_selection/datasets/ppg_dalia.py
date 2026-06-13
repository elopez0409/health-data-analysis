"""PPG-DaLiA adapter.

Real format: per-subject pickle ``S{n}.pkl`` (under ``S{n}/`` in the official
release) holding a dict::

    {signal: {chest: {ECG(700Hz), ACC, ...},
              wrist: {BVP(64Hz), ACC(32Hz), EDA, TEMP}},
     label (HR array, 8s window / 2s shift),
     activity, rpeaks, questionnaire}

Must be loaded with ``pickle.load(f, encoding='latin1')`` (Python-2 pickles).
ECG (chest) is the reference; the single selectable source is the wrist PPG
(``dalia_wrist``), whose HR is estimated from the BVP waveform downstream.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from hr_selection import config
from hr_selection.datasets.base import DatasetAdapter
from hr_selection.datasets.schema import RawSession, SignalChannel, SourceSignal

# Official DaLiA activity-id -> name.
DALIA_CODE_TO_ACT = {
    0: "transient",
    1: "sitting",
    2: "stairs",
    3: "soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}


def _as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] == 1:
            arr = arr[:, 0]
    return arr


class PPGDaLiAAdapter(DatasetAdapter):
    name = "ppg_dalia"
    canonical_sources = config.DATASET_SOURCES["ppg_dalia"]

    def _pickle_paths(self) -> list[Path]:
        paths = sorted(self.root.glob("S*/S*.pkl"))
        if not paths:
            paths = sorted(self.root.glob("S*.pkl"))
        return paths

    def session_keys(self) -> list[str]:
        """Stable per-session keys (pickle paths) for parallel loading."""
        return [str(p) for p in self._pickle_paths()]

    def iter_sessions(self) -> Iterator[RawSession]:
        for pkl_path in self._pickle_paths():
            yield self.load_session(str(pkl_path))

    def load_session(self, key: str) -> RawSession:
        pkl_path = Path(key)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        subject = str(data.get("subject", pkl_path.stem))
        sig = data["signal"]
        bvp = _as_1d(sig["wrist"]["BVP"])
        acc = np.asarray(sig["wrist"]["ACC"], dtype=float)

        label = _as_1d(data["label"])
        ref_fs = 1.0 / config.SHIFT_SEC  # one HR value per 2 s shift

        activity_raw = _as_1d(data.get("activity", np.array([])))
        if activity_raw.size:
            activity = np.array([DALIA_CODE_TO_ACT.get(int(round(c)), "unknown") for c in activity_raw])
        else:
            activity = None

        source = SourceSignal(
            name="dalia_wrist",
            hr=np.array([]),  # no device HR; estimated from BVP downstream
            hr_fs=config.FS["dalia_bvp"],
            raw={
                "ppg": SignalChannel("BVP", config.FS["dalia_bvp"], bvp),
                "acc": SignalChannel("ACC", config.FS["dalia_acc"], acc),
            },
        )

        duration = bvp.shape[0] / config.FS["dalia_bvp"]
        return RawSession(
            dataset=self.name,
            subject_id=subject,
            duration_sec=duration,
            reference_hr=label,
            reference_hr_fs=ref_fs,
            sources={"dalia_wrist": source},
            activity=activity,
            activity_fs=ref_fs,
            metadata={"skin_tone": float("nan")},
            canonical_sources=list(self.canonical_sources),
        )
