"""
Walk-forward out-of-sample backtest for HR calibration strategies.

Compares three strategies per night (no look-ahead):
  - uncorrected: raw wrist reading
  - population_prior: wrist - pooled population mean offset
  - personal: wrist - user's walk-forward personal offset estimate

Usage:
    python scripts/backtest_calibration.py
    python scripts/backtest_calibration.py --warmup 7 --holdout-start 60
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from hr_selection import config
from hr_selection.personal.backtest import (
    STRATEGIES,
    aggregate_backtest_metrics,
    walk_forward_backtest,
)
from hr_selection.synthetic.calibration import build_paired_nights

console = Console()
app = typer.Typer(help="Walk-forward calibration backtest")

STRATEGIES = ("uncorrected", "population_prior", "personal")


def _load_paired(data_path: Path, truth_path: Path):
    import pandas as pd

    readings = pd.read_csv(data_path, parse_dates=["date"])
    truth = pd.read_csv(truth_path)
    paired = build_paired_nights(readings)
    paired = paired.sort_values(["user_id", "date"]).reset_index(drop=True)
    return paired, truth


def _aggregate_metrics(nights: list[dict], holdout_start: int) -> dict:
    return aggregate_backtest_metrics(nights, holdout_start=holdout_start)


@app.command()
def main(
    data: str = typer.Option(None, help="Path to hr_calibration_dataset.csv"),
    truth: str = typer.Option(None, help="Path to hr_calibration_truth.csv"),
    warmup: int = typer.Option(7, help="Warmup nights before OOS predictions"),
    holdout_start: int = typer.Option(60, help="First night of fixed holdout window"),
    out: str = typer.Option(None, help="JSON output path"),
) -> None:
    """Run walk-forward backtest and save results."""
    data_path = Path(data) if data else config.DATA_DIR / "hr_calibration_dataset.csv"
    truth_path = Path(truth) if truth else config.DATA_DIR / "hr_calibration_truth.csv"
    out_path = Path(out) if out else config.HR_OUT_DIR / "calibration_backtest.json"

    if not data_path.exists():
        console.print(f"[red]Missing {data_path}. Run: python scripts/gen_calibration_dataset.py[/red]")
        raise typer.Exit(1)

    paired, _ = _load_paired(data_path, truth_path)
    pop_mean = float(paired["delta"].mean())

    nights, meta = walk_forward_backtest(paired, warmup=warmup, population_mean=pop_mean)
    metrics = _aggregate_metrics(nights, holdout_start=holdout_start)
    metrics["meta"] = meta
    metrics["population_prior"] = {"mean": round(pop_mean, 4), "std": round(float(paired["delta"].std()), 4)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    table = Table(title=f"Walk-Forward Backtest (warmup={warmup}, holdout≥{holdout_start})")
    table.add_column("Strategy", style="bold")
    table.add_column("Cumulative OOS MAE", justify="right")
    table.add_column("Holdout MAE", justify="right")
    table.add_column("Rolling 14n MAE", justify="right")

    for strat in STRATEGIES:
        s = metrics["strategies"][strat]
        table.add_row(
            strat,
            f"{s['cumulative_oos_mae']:.3f}",
            f"{s['holdout_mae']:.3f}",
            f"{s['rolling_14night_mae']:.3f}",
        )
    console.print(table)
    console.print(f"Skill (personal vs uncorrected): {metrics['skill_pct_vs_uncorrected']:.1f}%")
    console.print(f"Results saved to {out_path}")

    personal_mae = metrics["strategies"]["personal"]["cumulative_oos_mae"]
    uncorr_mae = metrics["strategies"]["uncorrected"]["cumulative_oos_mae"]
    prior_mae = metrics["strategies"]["population_prior"]["cumulative_oos_mae"]
    if personal_mae < uncorr_mae and personal_mae <= prior_mae:
        console.print("[green]Backtest passed: personal beats uncorrected and population prior.[/green]")
    else:
        console.print("[yellow]Backtest warning: personal did not beat all baselines.[/yellow]")


if __name__ == "__main__":
    app()
