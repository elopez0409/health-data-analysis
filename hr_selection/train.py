"""HR source-selection training CLI (typer, matching the repo's app/cli.py style).

Examples
--------
    python -m hr_selection.train --dataset all --source synthetic --model classical
    python -m hr_selection.train --dataset galaxyppg --model deep
    python -m hr_selection.train --dataset bigideas --backend lightgbm
"""

from __future__ import annotations

from pathlib import Path

import joblib
import typer
from rich.console import Console
from rich.table import Table

from hr_selection import config
from hr_selection.evaluate import evaluate, save_metrics

console = Console()


def _print_metrics(metrics: dict, model: str) -> None:
    sel = metrics["selection"]
    mae = metrics["hr_mae"]

    table = Table(title=f"HR Source Selection — {model}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("windows", str(metrics["n_windows"]))
    table.add_row("top-1 accuracy", f"{sel['top1_accuracy']:.3f}")
    table.add_row("top-2 accuracy", f"{sel['top2_accuracy']:.3f}")
    table.add_row("macro-F1", f"{sel['macro_f1']:.3f}")
    table.add_row("HR MAE — model", f"{mae['model']:.3f}")
    table.add_row("HR MAE — oracle", f"{mae['oracle']:.3f}")
    table.add_row("HR MAE — cross-source median", f"{mae['cross_source_median']:.3f}")
    if "paper_prior" in mae:
        table.add_row("HR MAE — paper prior", f"{mae['paper_prior']:.3f}")
    table.add_row(
        f"HR MAE — best single ({mae['best_single_device_name']})",
        f"{mae['best_single_device']:.3f}",
    )
    table.add_row("improvement vs best single", f"{mae['improvement_vs_best_single']:+.3f}")
    table.add_row("improvement vs median", f"{mae['improvement_vs_median']:+.3f}")
    console.print(table)

    ds_table = Table(title="Per-dataset")
    ds_table.add_column("dataset")
    ds_table.add_column("n", justify="right")
    ds_table.add_column("top-1", justify="right")
    ds_table.add_column("MAE model", justify="right")
    ds_table.add_column("MAE oracle", justify="right")
    ds_table.add_column("MAE median", justify="right")
    for ds, d in metrics["per_dataset"].items():
        ds_table.add_row(
            ds, str(d["n"]), f"{d['top1_acc']:.3f}", f"{d['mae_model']:.3f}", f"{d['mae_oracle']:.3f}", f"{d['mae_median']:.3f}"
        )
    console.print(ds_table)

    verdict = "[green]beats baselines[/green]" if metrics["beats_baselines"] else "[red]does NOT beat baselines[/red]"
    console.print(f"Model {verdict} (single-device + cross-source median).")


def train(
    dataset: str = typer.Option("all", help="bigideas | galaxyppg | ppg_dalia | all"),
    source: str = typer.Option("synthetic", help="synthetic | real"),
    root: str = typer.Option(None, help="Override dataset root path (e.g. for real data)."),
    model: str = typer.Option("classical", help="classical | deep"),
    backend: str = typer.Option("hist", help="classical backend: hist | lightgbm"),
    max_iter: int = typer.Option(300, help="boosting iterations (classical)."),
    epochs: int = typer.Option(15, help="epochs (deep)."),
    n_jobs: int = typer.Option(-1, help="parallel workers for deep window-building (-1 = all cores)."),
    no_priors: bool = typer.Option(False, help="Ablation: omit paper-derived prior_* features."),
    save_model: bool = typer.Option(True, help="Persist fitted classical model as joblib artifact."),
    out_dir: str = typer.Option(None, help="Where to save metrics JSON."),
) -> None:
    """Train + evaluate a source-selection model on synthetic or real data."""
    datasets = config.ALL_DATASETS if dataset == "all" else [dataset]
    out_path = Path(out_dir) if out_dir else config.HR_OUT_DIR

    priors_note = " priors=off" if no_priors else ""
    console.print(
        f"[bold]Training[/bold] model={model} dataset={dataset} source={source} backend={backend}{priors_note}"
    )

    if model == "classical":
        from hr_selection.features.build import build_feature_table, feature_columns
        from hr_selection.models.classical import train_classical

        console.print("Building feature table…")
        df = build_feature_table(datasets, source=source, root=root)
        console.print(f"  {len(df)} windows, {df['group'].nunique()} subjects.")
        console.print("Cross-validated training (subject-wise GroupKFold)…")
        res = train_classical(df, backend=backend, max_iter=max_iter, use_priors=not no_priors)
        if res.get("degenerate"):
            console.print("[yellow]Degenerate selection (single source / subject) — predicting the only source.[/yellow]")
        metrics = evaluate(df, res["oof_proba"])
        tag = f"classical_{dataset}_{backend}" + ("_nopriors" if no_priors else "")

    elif model == "deep":
        from hr_selection.models.deep import train_deep

        console.print("Building raw windows + training 1D-CNN (subject-wise GroupKFold)…")
        res = train_deep(datasets, source=source, root=root, epochs=epochs, n_jobs=n_jobs)
        console.print(f"  device: [cyan]{res.get('device', 'cpu')}[/cyan]")
        if res["skipped_datasets"]:
            console.print(f"[yellow]Skipped (HR-only, no raw signals): {res['skipped_datasets']}[/yellow]")
        metrics = evaluate(res["df"], res["oof_proba"])
        tag = f"deep_{dataset}"

    else:
        raise typer.BadParameter(f"Unknown model '{model}'. Choose classical | deep.")

    _print_metrics(metrics, model)
    saved = save_metrics(metrics, out_path, tag=tag)
    console.print(f"Metrics saved to {saved}")

    if model == "classical" and save_model and res.get("model") is not None:
        artifact = {
            "model": res["model"],
            "feature_columns": feature_columns(use_priors=not no_priors),
            "canonical_sources": config.CANONICAL_SOURCES,
            "use_priors": not no_priors,
            "dataset": dataset,
            "backend": backend,
        }
        model_path = out_path / f"model_{tag}.joblib"
        joblib.dump(artifact, model_path)
        console.print(f"Model saved to {model_path}")


def _cli_entrypoint() -> None:
    typer.run(train)


if __name__ == "__main__":
    _cli_entrypoint()
