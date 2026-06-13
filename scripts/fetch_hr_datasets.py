"""Download real wearable HR datasets for the hr_selection pipeline.

Currently supports the two openly-downloadable datasets:

- **PPG-DaLiA** (UCI ML Repository) -> per-subject pickles ``S{n}/S{n}.pkl``.
- **GalaxyPPG** (Zenodo 10.5281/zenodo.14635823) -> per-participant device tree.

BigIdeasLab_STEP is intentionally NOT handled here: it is a credentialed-access
PhysioNet dataset behind a Data Use Agreement and must be obtained manually.

Real files land under ``data/hr_real/`` (kept separate from the synthetic
``data/hr_raw/`` tree). Point training at them with ``--root`` per dataset, e.g.::

    python -m hr_selection.train --dataset ppg_dalia --root data/hr_real/ppg_dalia

Layout produced::

    data/hr_real/ppg_dalia/S1/S1.pkl ...
    data/hr_real/galaxyppg_extracted/<raw extraction>   (normalized in a 2nd step)

Usage::

    python scripts/fetch_hr_datasets.py                 # both datasets
    python scripts/fetch_hr_datasets.py --only ppg_dalia
    python scripts/fetch_hr_datasets.py --only galaxyppg --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HR_REAL = ROOT / "data" / "hr_real"
DOWNLOADS = HR_REAL / "_downloads"

SOURCES = {
    "ppg_dalia": {
        "url": "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip",
        "zip_name": "ppg_dalia.zip",
        "dest": HR_REAL / "ppg_dalia",
    },
    "galaxyppg": {
        "url": "https://zenodo.org/records/14635823/files/GalaxyPPG.zip?download=1",
        "zip_name": "GalaxyPPG.zip",
        "dest": HR_REAL / "galaxyppg_extracted",
    },
}


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Download a URL to ``dest`` using resumable ``curl`` (robust for GB files)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and dest.stat().st_size > 0:
        print(f"  [skip] already downloaded: {dest.name} ({_human(dest.stat().st_size)})")
        return dest

    print(f"  downloading {url}")
    # -L follow redirects, -C - resume a partial file, --retry for transient errors.
    cmd = ["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3",
           "-o", str(dest), url]
    proc = subprocess.run(cmd)
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(f"curl failed (exit {proc.returncode}) for {url}")
    print(f"  saved {dest.name} ({_human(dest.stat().st_size)})")
    return dest


def extract_zip(zip_path: Path, dest: Path, force: bool = False) -> Path:
    if dest.exists() and any(dest.iterdir()) and not force:
        print(f"  [skip] already extracted: {dest}")
        return dest
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {zip_path.name} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def arrange_ppg_dalia(extracted: Path, dest: Path) -> None:
    """Expose the UCI subjects as ``dest/S{n}/S{n}.pkl`` via symlinks.

    The UCI archive nests the subjects under a ``PPG_FieldStudy`` directory.
    The adapter globs ``S*/S*.pkl`` or ``S*.pkl``. We symlink (not copy) because
    the uncompressed pickles total ~12 GB.
    """
    pkls = sorted(extracted.rglob("S*.pkl"))
    if not pkls:
        print(f"  WARNING: no S*.pkl found under {extracted}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    for pkl in pkls:
        subj_dir = dest / pkl.stem
        subj_dir.mkdir(exist_ok=True)
        target = subj_dir / pkl.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(pkl.resolve())
    print(f"  arranged {len(pkls)} subject pickles under {dest} (symlinks)")


def fetch(name: str, force: bool = False) -> None:
    spec = SOURCES[name]
    print(f"[{name}]")
    zip_path = download(spec["url"], DOWNLOADS / spec["zip_name"], force=force)

    if name == "ppg_dalia":
        extracted = extract_zip(zip_path, DOWNLOADS / "ppg_dalia_extracted", force=force)
        # UCI nests the actual subjects inside an inner ``data.zip``.
        inner = extracted / "data.zip"
        if inner.exists() and not list(extracted.rglob("S*.pkl")):
            print(f"  extracting nested {inner.name}")
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(extracted)
        arrange_ppg_dalia(extracted, spec["dest"])
    elif name == "galaxyppg":
        extract_zip(zip_path, spec["dest"], force=force)
        print(f"  GalaxyPPG extracted to {spec['dest']}.")
        print("  Run scripts/normalize_galaxyppg.py to produce data/hr_raw/galaxyppg/.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=list(SOURCES), default=None, help="Fetch a single dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download / re-extract even if present.")
    args = parser.parse_args()

    names = [args.only] if args.only else list(SOURCES)
    for name in names:
        try:
            fetch(name, force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR fetching {name}: {exc}", file=sys.stderr)
    print("\nDone.")


if __name__ == "__main__":
    main()
