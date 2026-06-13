#!/usr/bin/env python3
"""Download the BigIdeasLab_STEP smartwatch HR dataset from PhysioNet.

This is a **credentialed-access** dataset published under the *PhysioNet
Restricted Health Data License*. Before running this script you must:

1. Hold a credentialed PhysioNet account
   (https://physionet.org/settings/credentialing/).
2. Sign the Data Use Agreement for the project page:
   https://physionet.org/content/bigideaslab-step-hr-smartwatch/1.0/

The files land under
``physionet.org/files/bigideaslab-step-hr-smartwatch/1.0/`` (relative to the
repo root) which is exactly where ``hr_selection.config.BIGIDEAS_REAL_CSV``
expects to find ``deidentified_data.csv``.

Credentials are read from the environment, or prompted interactively::

    export PHYSIONET_USERNAME=yourname
    export PHYSIONET_PASSWORD=...      # optional; prompted (hidden) if absent
    python scripts/fetch_bigideas_step.py

Options::

    python scripts/fetch_bigideas_step.py            # download + verify
    python scripts/fetch_bigideas_step.py --force     # re-download everything
    python scripts/fetch_bigideas_step.py --no-verify # skip checksum check

IMPORTANT: This data is RESTRICTED. Do not commit it to git or share it with
anyone. It is git-ignored on purpose; keep it that way.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SLUG = "bigideaslab-step-hr-smartwatch"
VERSION = "1.0"
BASE_URL = f"https://physionet.org/files/{SLUG}/{VERSION}"

# Where the files live, mirroring PhysioNet's own path layout so config.py works.
DEST_DIR = ROOT / "physionet.org" / "files" / SLUG / VERSION

# Files that make up the published record. SHA256SUMS is fetched first and used
# to verify the rest; the data/protocol are the parts the pipeline actually needs.
FILES = [
    "SHA256SUMS.txt",
    "LICENSE.txt",
    "README.md",
    "deidentified_data.csv",
    "protocol.pdf",
]


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _credentials() -> tuple[str, str]:
    user = os.environ.get("PHYSIONET_USERNAME")
    if not user:
        user = input("PhysioNet username: ").strip()
    if not user:
        print("ERROR: a PhysioNet username is required.", file=sys.stderr)
        sys.exit(2)
    password = os.environ.get("PHYSIONET_PASSWORD")
    if not password:
        password = getpass.getpass(f"PhysioNet password for {user}: ")
    if not password:
        print("ERROR: a PhysioNet password is required.", file=sys.stderr)
        sys.exit(2)
    return user, password


def download(name: str, user: str, password: str, force: bool = False) -> Path:
    """Download a single dataset file via authenticated, resumable ``curl``."""
    dest = DEST_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [skip] {name} already present ({_human(dest.stat().st_size)})")
        return dest

    url = f"{BASE_URL}/{name}"
    print(f"  downloading {name}")
    # -L follow redirects, -C - resume partial files, --retry transient errors,
    # -f fail (non-zero exit) on HTTP errors so bad auth doesn't write an HTML page.
    cmd = [
        "curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "3",
        "--user", f"{user}:{password}", "-o", str(dest), url,
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        # Remove any partial/garbage file so a retry starts clean.
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        raise RuntimeError(
            f"curl failed (exit {proc.returncode}) for {url}. "
            "Check your PhysioNet credentials and that you have signed the "
            "Data Use Agreement for this project."
        )
    print(f"  saved {name} ({_human(dest.stat().st_size)})")
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(checksums_file: Path) -> bool:
    """Verify downloaded files against the published SHA256SUMS.txt."""
    expected: dict[str, str] = {}
    for line in checksums_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition(" ")
        expected[name.strip()] = digest.strip()

    ok = True
    print("\nVerifying checksums:")
    for name, want in expected.items():
        path = DEST_DIR / name
        if not path.exists():
            continue  # we only download a subset; ignore files we skipped
        got = _sha256(path)
        status = "OK" if got == want else "MISMATCH"
        if got != want:
            ok = False
        print(f"  [{status}] {name}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--no-verify", action="store_true", help="Skip SHA256 verification.")
    args = parser.parse_args()

    print(f"BigIdeasLab_STEP -> {DEST_DIR}")
    user, password = _credentials()

    for name in FILES:
        try:
            download(name, user, password, force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    if not args.no_verify:
        checksums = DEST_DIR / "SHA256SUMS.txt"
        if checksums.exists():
            if not verify(checksums):
                print("\nChecksum verification FAILED.", file=sys.stderr)
                sys.exit(1)
            print("\nAll checksums verified.")
        else:
            print("\nWARNING: SHA256SUMS.txt missing; skipped verification.")

    print(f"\nDone. CSV ready at: {DEST_DIR / 'deidentified_data.csv'}")


if __name__ == "__main__":
    main()
