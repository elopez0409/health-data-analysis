"""
Generate synthetic HR calibration dataset for baseline + personal offset demos.

Output:
    data/hr_calibration_dataset.csv
    data/hr_calibration_truth.csv
    data/hr_calibration_dataset.md

Usage:
    python scripts/gen_calibration_dataset.py
    python scripts/gen_calibration_dataset.py --seed 42 --users 20 --nights 90
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from hr_selection import config
from hr_selection.synthetic.calibration import write_calibration_dataset

console = Console()
app = typer.Typer(help="Generate HR calibration dataset")


@app.command()
def main(
    seed: int = typer.Option(config.SEED, help="Random seed"),
    users: int = typer.Option(20, help="Number of simulated users"),
    nights: int = typer.Option(90, help="Nights per user"),
    out_dir: str = typer.Option(None, help="Output directory (default: data/)"),
) -> None:
    """Generate calibration CSV + truth sidecar + data dictionary."""
    root = Path(out_dir) if out_dir else config.DATA_DIR
    console.print(f"[bold]Generating HR calibration dataset[/bold]  seed={seed}  users={users}  nights={nights}")
    console.print(f"  Output: {root}")

    manifest = write_calibration_dataset(root, seed=seed, n_users=users, n_nights=nights)

    table = Table(title="Calibration Dataset Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("users", str(manifest.n_users))
    table.add_row("nights/user", str(manifest.n_nights))
    table.add_row("rows", str(manifest.n_rows))
    table.add_row("expected (no missing)", str(manifest.n_expected))
    table.add_row("missing rate", f"{manifest.missing_rate:.1%}")
    table.add_row("pop offset mean", f"{manifest.population_offset_mean:.2f} bpm")
    table.add_row("pop offset std", f"{manifest.population_offset_std:.2f} bpm")
    console.print(table)

    for label, path in manifest.paths.items():
        console.print(f"  [{label}] {path}")
    console.print("[green]Done![/green]")


if __name__ == "__main__":
    app()
