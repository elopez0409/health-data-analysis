"""
Generate synthetic HR source-selection datasets in the exact real on-disk
formats of BigIdeasLab_STEP, GalaxyPPG, and PPG-DaLiA.

A shared physiological model produces ground-truth HR + motion per subject; the
reference (ECG/Polar) tracks truth, while each device source adds bias,
motion-correlated error, lag, dropouts, and (BigIdeas) skin-tone degradation.
Raw-signal datasets get synthesized ECG/PPG/ACC waveforms.

Output: data/hr_raw/{bigideas,galaxyppg,ppg_dalia}/ + data/hr_raw/manifest.json

Usage:
    python scripts/gen_synthetic_hr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hr_selection import config  # noqa: E402
from hr_selection.synthetic import generate_all  # noqa: E402


def main() -> None:
    print("Generating synthetic HR source-selection datasets")
    print(f"  Output root: {config.HR_RAW_DIR}")
    print(f"  Seed: {config.SEED}, window {config.WINDOW_SEC}s / shift {config.SHIFT_SEC}s")
    print()

    manifest = generate_all()

    for name, info in manifest["datasets"].items():
        n_sub = info.get("n_subjects")
        print(f"  [{name}] {info['format']}")
        print(f"      reference={info['reference']}  subjects={n_sub}  sources={info['sources']}")

    print(f"\n  Manifest: {config.HR_RAW_DIR / 'manifest.json'}")
    print("  Done!")


if __name__ == "__main__":
    main()
