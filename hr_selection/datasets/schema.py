"""Unified in-memory representation shared by every adapter.

A ``RawSession`` is one participant's recording: a reference HR series, a set of
device ``SourceSignal``s (each with an HR series and optionally raw waveforms),
an activity track, and metadata (e.g. skin tone).  Windowing turns a session
into ``WindowRecord``s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalChannel:
    """A raw waveform channel (e.g. PPG, BVP, ECG, ACC)."""

    name: str
    fs: float
    data: np.ndarray  # shape (N,) for 1-D signals or (N, 3) for ACC

    def slice_seconds(self, t0: float, t1: float) -> np.ndarray:
        i0 = int(round(t0 * self.fs))
        i1 = int(round(t1 * self.fs))
        i0 = max(0, i0)
        i1 = min(self.data.shape[0], i1)
        if i1 <= i0:
            return self.data[0:0]
        return self.data[i0:i1]


@dataclass
class SourceSignal:
    """A selectable device source."""

    name: str  # canonical source key (see config.CANONICAL_SOURCES)
    hr: np.ndarray  # reported HR series; NaN where dropped out
    hr_fs: float = 1.0
    raw: dict[str, SignalChannel] = field(default_factory=dict)

    def hr_at(self, t0: float, t1: float) -> float:
        i0 = int(round(t0 * self.hr_fs))
        i1 = int(round(t1 * self.hr_fs))
        i0 = max(0, i0)
        i1 = min(self.hr.shape[0], i1)
        if i1 <= i0:
            return float("nan")
        seg = self.hr[i0:i1]
        if np.all(np.isnan(seg)):
            return float("nan")
        return float(np.nanmean(seg))


@dataclass
class RawSession:
    """One participant's multi-source recording."""

    dataset: str
    subject_id: str
    duration_sec: float
    reference_hr: np.ndarray
    reference_hr_fs: float = 1.0
    sources: dict[str, SourceSignal] = field(default_factory=dict)
    activity: np.ndarray | None = None  # string array
    activity_fs: float = 1.0
    metadata: dict = field(default_factory=dict)
    canonical_sources: list[str] = field(default_factory=list)

    def reference_hr_at(self, t0: float, t1: float) -> float:
        i0 = int(round(t0 * self.reference_hr_fs))
        i1 = int(round(t1 * self.reference_hr_fs))
        i0 = max(0, i0)
        i1 = min(self.reference_hr.shape[0], i1)
        if i1 <= i0:
            return float("nan")
        seg = self.reference_hr[i0:i1]
        if np.all(np.isnan(seg)):
            return float("nan")
        return float(np.nanmean(seg))

    def activity_at(self, t0: float, t1: float) -> str:
        if self.activity is None or len(self.activity) == 0:
            return "unknown"
        i0 = int(round(t0 * self.activity_fs))
        i1 = int(round(t1 * self.activity_fs))
        i0 = max(0, i0)
        i1 = min(len(self.activity), i1)
        if i1 <= i0:
            return "unknown"
        seg = self.activity[i0:i1]
        vals, counts = np.unique(seg, return_counts=True)
        return str(vals[int(np.argmax(counts))])


@dataclass
class WindowRecord:
    """A single time window after windowing a RawSession."""

    dataset: str
    subject_id: str
    window_idx: int
    t_start: float
    reference_hr: float
    activity: str
    skin_tone: float  # NaN when unknown
    per_source: dict[str, dict] = field(default_factory=dict)
    available_sources: list[str] = field(default_factory=list)
    label: int = -1  # canonical index of best available source
