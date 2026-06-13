"""Dataset adapters: read each real on-disk format into a unified RawSession."""

from hr_selection.datasets.base import DatasetAdapter, get_adapter
from hr_selection.datasets.schema import RawSession, SignalChannel, SourceSignal, WindowRecord

__all__ = [
    "DatasetAdapter",
    "get_adapter",
    "RawSession",
    "SignalChannel",
    "SourceSignal",
    "WindowRecord",
]
