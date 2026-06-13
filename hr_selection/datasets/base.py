"""Abstract dataset adapter + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from hr_selection.datasets.schema import RawSession


class DatasetAdapter(ABC):
    """Read a dataset (real or synthetic) into ``RawSession`` objects.

    The same adapter reads synthetic output and real data: synthetic generators
    write files in the exact real on-disk format, so swapping in real data only
    means pointing ``root`` at the real download.
    """

    #: short dataset key, e.g. "bigideas"
    name: str = ""
    #: canonical source keys this dataset provides
    canonical_sources: list[str] = []

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @abstractmethod
    def iter_sessions(self) -> Iterator[RawSession]:
        """Yield one ``RawSession`` per participant."""
        raise NotImplementedError


def get_adapter(name: str, root: str | Path) -> DatasetAdapter:
    """Factory: return the adapter for a dataset key."""
    from hr_selection.datasets.bigideas import BigIdeasAdapter
    from hr_selection.datasets.galaxyppg import GalaxyPPGAdapter
    from hr_selection.datasets.ppg_dalia import PPGDaLiAAdapter

    adapters: dict[str, type[DatasetAdapter]] = {
        "bigideas": BigIdeasAdapter,
        "galaxyppg": GalaxyPPGAdapter,
        "ppg_dalia": PPGDaLiAAdapter,
    }
    if name not in adapters:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(adapters)}.")
    return adapters[name](root)
