"""BigIdeasLab_STEP adapter.

Real format: a single CSV, one row per synced timepoint, columns exactly
``ECG, Apple Watch, Empatica, Garmin, Fitbit, Miband, Biovotion, ID, Skin Tone,
Activity`` (HR in bpm; ECG is the reference). HR-only -> no raw waveforms.
Rows are assumed uniformly spaced at ``config.BIGIDEAS_ROW_HZ``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from hr_selection import config
from hr_selection.datasets.base import DatasetAdapter
from hr_selection.datasets.schema import RawSession, SourceSignal

# CSV column header -> canonical source key.
_COLUMN_TO_SOURCE = {
    "Apple Watch": "apple_watch",
    "Empatica": "empatica",
    "Garmin": "garmin",
    "Fitbit": "fitbit",
    "Miband": "miband",
    "Biovotion": "biovotion",
}


class BigIdeasAdapter(DatasetAdapter):
    name = "bigideas"
    canonical_sources = config.DATASET_SOURCES["bigideas"]

    def _find_csv(self) -> Path:
        if self.root.is_file():
            return self.root
        candidates = sorted(self.root.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No BigIdeasLab CSV found under {self.root}")
        return candidates[0]

    def iter_sessions(self) -> Iterator[RawSession]:
        csv_path = self._find_csv()
        df = pd.read_csv(csv_path)
        row_fs = config.BIGIDEAS_ROW_HZ

        for sid, sub in df.groupby("ID", sort=True):
            sub = sub.reset_index(drop=True)
            n = len(sub)
            reference = sub["ECG"].to_numpy(dtype=float)
            activity = sub["Activity"].astype(str).to_numpy()
            skin_tone = float(sub["Skin Tone"].iloc[0]) if "Skin Tone" in sub else float("nan")

            sources: dict[str, SourceSignal] = {}
            for col, key in _COLUMN_TO_SOURCE.items():
                if col not in sub.columns:
                    continue
                hr = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                sources[key] = SourceSignal(name=key, hr=hr, hr_fs=row_fs)

            yield RawSession(
                dataset=self.name,
                subject_id=str(sid),
                duration_sec=n / row_fs,
                reference_hr=reference,
                reference_hr_fs=row_fs,
                sources=sources,
                activity=activity,
                activity_fs=row_fs,
                metadata={"skin_tone": skin_tone},
                canonical_sources=list(self.canonical_sources),
            )
