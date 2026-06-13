"""
Demonstrate population prior fit and personal offset convergence at 30/60/90 nights.

Usage:
    python scripts/demo_calibration_convergence.py
    python scripts/demo_calibration_convergence.py --data data/hr_calibration_dataset.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from hr_selection import config
from hr_selection.personal.estimator import OffsetState, batch_update_offset
from hr_selection.synthetic.calibration import build_paired_nights

console = Console()
app = typer.Typer(help="Calibration convergence demo")

SNAPSHOT_NIGHTS = [30, 60, 90]


def _load_data(data_path: Path, truth_path: Path):
    import pandas as pd

    readings = pd.read_csv(data_path, parse_dates=["date"])
    truth = pd.read_csv(truth_path)
    paired = build_paired_nights(readings)
    paired = paired.sort_values(["user_id", "date"]).reset_index(drop=True)
    return readings, truth, paired


def _population_prior(paired):
    """Pool all users' nightly wrist-chest deltas."""
    deltas = paired["delta"].to_numpy()
    mean = float(deltas.mean())
    std = float(deltas.std())
    mae_chest = 0.0  # anchor by definition
    mae_wrist = float((paired["wrist"] - paired["chest"]).abs().mean())
    return {"mean": mean, "std": std, "mae_wrist_vs_chest": mae_wrist, "n_pairs": len(paired)}


def _convergence_at_nights(paired, truth, checkpoints: list[int]):
    """Per-user offset estimate error and CI width at each checkpoint."""
    truth_map = truth.set_index("user_id")
    results: list[dict] = []

    for user_id, grp in paired.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        true_offset = float(truth_map.loc[user_id, "personal_offset"])
        state = OffsetState()
        n_pairs = len(grp)

        for i, row in grp.iterrows():
            state = batch_update_offset(state, [float(row["delta"])])
            night_num = i + 1

            for cp in checkpoints:
                target_night = min(cp, n_pairs)
                if night_num == target_night:
                    ci_width = (
                        state.ci_high - state.ci_low
                        if state.ci_high == state.ci_high
                        else float("nan")
                    )
                    results.append(
                        {
                            "user_id": user_id,
                            "night": cp,
                            "actual_pairs": night_num,
                            "true_offset": true_offset,
                            "estimated_offset": state.offset_mean,
                            "abs_error": abs(state.offset_mean - true_offset),
                            "ci_width": ci_width,
                            "n_samples": state.n_samples,
                        }
                    )

    import pandas as pd

    return pd.DataFrame(results)


@app.command()
def main(
    data: str = typer.Option(None, help="Path to hr_calibration_dataset.csv"),
    truth: str = typer.Option(None, help="Path to hr_calibration_truth.csv"),
) -> None:
    """Run population prior + personal offset convergence demo."""
    data_path = Path(data) if data else config.DATA_DIR / "hr_calibration_dataset.csv"
    truth_path = Path(truth) if truth else config.DATA_DIR / "hr_calibration_truth.csv"

    if not data_path.exists():
        console.print(f"[red]Missing {data_path}. Run: python scripts/gen_calibration_dataset.py[/red]")
        raise typer.Exit(1)

    _, truth_df, paired = _load_data(data_path, truth_path)

    # (a) Population prior
    pop = _population_prior(paired)
    pop_table = Table(title="(a) Population Prior (pooled wrist - chest_strap)")
    pop_table.add_column("Metric", style="bold")
    pop_table.add_column("Value", justify="right")
    pop_table.add_row("paired nights", str(pop["n_pairs"]))
    pop_table.add_row("offset mean", f"{pop['mean']:.3f} bpm")
    pop_table.add_row("offset std", f"{pop['std']:.3f} bpm")
    pop_table.add_row("wrist MAE vs chest", f"{pop['mae_wrist_vs_chest']:.3f} bpm")
    console.print(pop_table)

    # (b) Personal convergence
    conv = _convergence_at_nights(paired, truth_df, SNAPSHOT_NIGHTS)

    conv_table = Table(title="(b) Personal Offset Convergence (mean across users)")
    conv_table.add_column("Night", justify="right")
    conv_table.add_column("|est - true|", justify="right")
    conv_table.add_column("CI width", justify="right")
    conv_table.add_column("n_samples", justify="right")

    for night in SNAPSHOT_NIGHTS:
        sub = conv[conv["night"] == night]
        conv_table.add_row(
            str(night),
            f"{sub['abs_error'].mean():.3f}",
            f"{sub['ci_width'].mean():.3f}",
            f"{sub['n_samples'].mean():.0f}",
        )
    console.print(conv_table)

    # Monotonicity check
    errors = [conv[conv["night"] == n]["abs_error"].mean() for n in SNAPSHOT_NIGHTS]
    ci_widths = [conv[conv["night"] == n]["ci_width"].mean() for n in SNAPSHOT_NIGHTS]
    if errors[0] > errors[-1] and ci_widths[0] > ci_widths[-1]:
        console.print("[green]Convergence confirmed: error and CI width decrease 30→90 nights.[/green]")
    else:
        console.print("[yellow]Warning: expected monotonic decrease in error/CI width.[/yellow]")


if __name__ == "__main__":
    app()
