"""Run HR source-selection training on Modal cloud GPUs.

This mirrors ``python -m hr_selection.train`` but executes on a remote GPU
(CUDA), and adds a parallel hyperparameter ``sweep`` that fans many configs out
across containers at once. The remote image is built from this same project, so
the training code is identical to local.

Prereqs
-------
1. Install the launcher deps locally:    pip install -e '.[cloud]'
2. Authenticate once:                    modal setup
3. Upload your (normalized) real data to the ``hr-data`` Volume, one dir per
   dataset, matching the layout the adapters expect:

       modal volume create hr-data            # (auto-created on first use too)
       modal volume put hr-data ./data/hr_real/galaxyppg  /galaxyppg
       modal volume put hr-data ./data/hr_real/ppg_dalia  /ppg_dalia

   (PPG-DaLiA is large; the upload is the main one-time cost.)

Usage
-----
    # one training run on an A10G GPU
    modal run scripts/modal_train.py --dataset galaxyppg --model deep --epochs 15

    # classical baseline (still runs remotely; no GPU needed but fine on one)
    modal run scripts/modal_train.py --dataset galaxyppg --model classical

    # parallel sweep over epochs/lr (deep) — each combo gets its own container
    modal run scripts/modal_train.py::sweep --dataset galaxyppg

Results print locally and are also written to the ``hr-out`` Volume:
    modal volume get hr-out / ./data/hr_out/modal
"""

from __future__ import annotations

import json

import modal

GPU = "A10G"  # cheap + plenty for the 1D-CNN; bump to "A100" for bigger models.
DATA_DIR = "/data"
OUT_DIR = "/out"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.14.0",
        "joblib>=1.3.0",
        "torch>=2.2.0",
        "typer>=0.12.0",
        "rich>=13.7.0",
    )
    # Ship the training code itself so the remote runs the exact same logic.
    .add_local_python_source("hr_selection")
)

app = modal.App("hr-source-selection")

# Persistent storage: inputs in hr-data, metrics JSON out in hr-out.
data_vol = modal.Volume.from_name("hr-data", create_if_missing=True)
out_vol = modal.Volume.from_name("hr-out", create_if_missing=True)


def _resolve_root(dataset: str, root: str | None) -> str:
    """Default each dataset to ``/data/<dataset>`` inside the Volume."""
    return root if root else f"{DATA_DIR}/{dataset}"


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_DIR: data_vol, OUT_DIR: out_vol},
    timeout=60 * 60,
)
def run_training(cfg: dict) -> dict:
    """Train + evaluate one config (a dict) on the GPU; returns the metrics dict.

    Taking a single dict keeps ``.map`` (parallel sweeps) and ``.remote``
    (single run) on one code path.
    """
    from pathlib import Path

    from hr_selection.evaluate import evaluate, save_metrics

    dataset = cfg.get("dataset", "galaxyppg")
    model = cfg.get("model", "deep")
    source = cfg.get("source", "real")
    root = cfg.get("root")
    backend = cfg.get("backend", "hist")
    max_iter = int(cfg.get("max_iter", 300))
    epochs = int(cfg.get("epochs", 15))
    lr = float(cfg.get("lr", 1e-3))
    n_jobs = int(cfg.get("n_jobs", -1))

    ds_root = _resolve_root(dataset, root)
    datasets = [dataset]

    if model == "classical":
        from hr_selection.features.build import build_feature_table
        from hr_selection.models.classical import train_classical

        df = build_feature_table(datasets, source=source, root=ds_root)
        res = train_classical(df, backend=backend, max_iter=max_iter)
        metrics = evaluate(df, res["oof_proba"])
        tag = f"classical_{dataset}_{backend}"
        device = "cpu"
    elif model == "deep":
        from hr_selection.models.deep import train_deep

        res = train_deep(
            datasets, source=source, root=ds_root, epochs=epochs, lr=lr, n_jobs=n_jobs
        )
        metrics = evaluate(res["df"], res["oof_proba"])
        tag = f"deep_{dataset}"
        device = res.get("device", "cpu")
    else:
        raise ValueError(f"Unknown model '{model}'. Choose classical | deep.")

    metrics["_run"] = {
        "dataset": dataset,
        "model": model,
        "source": source,
        "epochs": epochs,
        "lr": lr,
        "device": device,
    }
    saved = save_metrics(metrics, Path(OUT_DIR), tag=f"modal_{tag}")
    out_vol.commit()  # persist the JSON to the Volume
    print(f"[remote] device={device} saved={saved}")
    return metrics


def _summarize(metrics: dict) -> str:
    sel = metrics["selection"]
    mae = metrics["hr_mae"]
    run = metrics.get("_run", {})
    return (
        f"{run.get('model','?')}/{run.get('dataset','?')} "
        f"[{run.get('device','?')}] "
        f"n={metrics['n_windows']} top1={sel['top1_accuracy']:.3f} "
        f"MAE model={mae['model']:.3f} vs median={mae['cross_source_median']:.3f} "
        f"({'beats' if metrics['beats_baselines'] else 'below'} baselines)"
    )


@app.local_entrypoint()
def main(
    dataset: str = "galaxyppg",
    model: str = "deep",
    source: str = "real",
    root: str = "",
    backend: str = "hist",
    max_iter: int = 300,
    epochs: int = 15,
    lr: float = 1e-3,
    n_jobs: int = -1,
):
    """Launch a single remote training run and print the metrics."""
    metrics = run_training.remote(
        {
            "dataset": dataset,
            "model": model,
            "source": source,
            "root": (root or None),
            "backend": backend,
            "max_iter": max_iter,
            "epochs": epochs,
            "lr": lr,
            "n_jobs": n_jobs,
        }
    )
    print(_summarize(metrics))
    print(json.dumps(metrics["hr_mae"], indent=2))


@app.local_entrypoint()
def sweep(dataset: str = "galaxyppg", source: str = "real", root: str = ""):
    """Fan a small deep-model grid out across containers in parallel."""
    grid = [
        {"epochs": e, "lr": lr}
        for e in (5, 15, 30)
        for lr in (1e-3, 3e-4)
    ]
    args = [
        {
            "dataset": dataset,
            "model": "deep",
            "source": source,
            "root": (root or None),
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
        }
        for cfg in grid
    ]
    print(f"Launching {len(args)} configs in parallel on {GPU}…")
    results = list(run_training.map(args))
    results.sort(key=lambda m: m["hr_mae"]["model"])
    print("\n=== sweep results (best HR MAE first) ===")
    for m in results:
        print(" ", _summarize(m))
