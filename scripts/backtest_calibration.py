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
from hr_selection.personal.estimator import OffsetState, update_offset
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


def walk_forward_backtest(
    paired,
    *,
    warmup: int = 7,
    population_mean: float | None = None,
) -> tuple[list[dict], dict]:
    """Expanding-window walk-forward backtest per user."""
    if population_mean is None:
        population_mean = float(paired["delta"].mean())

    all_nights: list[dict] = []

    for user_id, grp in paired.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        state = OffsetState()

        for t in range(len(grp)):
            row = grp.iloc[t]
            chest = float(row["chest"])
            wrist = float(row["wrist"])
            night = t + 1

            # Predict using only history [0..t-1]
            if t >= warmup:
                offset_est = state.offset_mean if state.n_samples > 0 else population_mean
                preds = {
                    "uncorrected": wrist,
                    "population_prior": wrist - population_mean,
                    "personal": wrist - offset_est,
                }
                for strat, pred in preds.items():
                    all_nights.append(
                        {
                            "user_id": user_id,
                            "night": night,
                            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                            "strategy": strat,
                            "predicted_hr": pred,
                            "target_chest": chest,
                            "error": abs(pred - chest),
                            "in_sample": False,
                        }
                    )

            # Update state with night t (for future predictions)
            state = update_offset(state, float(row["delta"]))

    return all_nights, {"population_mean": population_mean, "warmup": warmup}


def _aggregate_metrics(nights: list[dict], holdout_start: int) -> dict:
    import pandas as pd

    df = pd.DataFrame(nights)
    out: dict = {"strategies": {}, "holdout_start": holdout_start}

    for strat in STRATEGIES:
        sub = df[df["strategy"] == strat]
        cum_mae = float(sub["error"].mean())
        holdout = sub[sub["night"] >= holdout_start]
        holdout_mae = float(holdout["error"].mean()) if len(holdout) else float("nan")

        # Rolling 14-night MAE (last 14 OOS nights per user, then average)
        rolling = []
        for _, grp in sub.groupby("user_id"):
            grp = grp.sort_values("night")
            if len(grp) >= 14:
                rolling.append(float(grp.tail(14)["error"].mean()))
        rolling_mae = float(np.mean(rolling)) if rolling else float("nan")

        out["strategies"][strat] = {
            "cumulative_oos_mae": round(cum_mae, 4),
            "holdout_mae": round(holdout_mae, 4),
            "rolling_14night_mae": round(rolling_mae, 4),
            "n_predictions": int(len(sub)),
        }

    uncorr = out["strategies"]["uncorrected"]["cumulative_oos_mae"]
    personal = out["strategies"]["personal"]["cumulative_oos_mae"]
    if uncorr > 0:
        out["skill_pct_vs_uncorrected"] = round(100 * (uncorr - personal) / uncorr, 2)
    else:
        out["skill_pct_vs_uncorrected"] = 0.0

    return out


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
