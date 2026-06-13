# ML System: HR Source Selection

Given multiple wearable HR sources in a time window, predict which source's HR reading is closest to the reference signal (ECG or Polar chest strap). The goal is to minimize downstream HR error by dynamically selecting the best source per window.

## Problem Definition

- **Input**: A time window (default 8 s, shifted every 2 s) with HR readings from N available wearable sources, plus context (activity, skin tone) and optional raw PPG/ACC signals.
- **Output**: Which canonical source to trust for that window.
- **Label**: The available source whose HR is closest to the reference: `argmin_s |hr_s - reference_hr|`.
- **Evaluation**: Selection accuracy (top-1/top-2, macro-F1) and downstream HR MAE vs oracle, cross-source median, paper-prior baseline, and best single device.

## Pipeline Overview

```
Dataset (synthetic or real)
        │
        ▼
  DatasetAdapter  ──►  RawSession per subject
        │
        ▼
  Feature Builder  ──►  Per-window feature table (one row per window)
        │                    │
        │                    ├── Per-source: HR, missing, deviation, rolling std
        │                    ├── Signal quality (PPG datasets only)
        │                    └── Paper priors (activity + skin tone context)
        │
        ├──► Classical track (HistGradientBoosting / LightGBM)
        │         GroupKFold by subject → OOF probabilities
        │
        └──► Deep track (1D-CNN on raw PPG+ACC, GalaxyPPG + PPG-DaLiA only)
                  GroupKFold by subject → OOF probabilities
        │
        ▼
  evaluate()  ──►  Metrics JSON + optional joblib artifact
```

## Configuration (`hr_selection/config.py`)

Central constants shared across the pipeline:

| Setting | Default | Purpose |
|---------|---------|---------|
| `WINDOW_SEC` | 8.0 | Window length |
| `SHIFT_SEC` | 2.0 | Window shift (matches PPG-DaLiA label cadence) |
| `HR_FS` | 1.0 | HR series normalized to 1 Hz |
| `SEED` | 42 | Reproducibility |
| `CANONICAL_SOURCES` | 9 devices | Unified label space across datasets |
| `DATASET_DIRS` | `data/hr_raw/*` | On-disk dataset paths |
| `HR_OUT_DIR` | `data/hr_out/` | Metrics and model artifacts |

## Datasets

Three public research datasets, each with a `DatasetAdapter` in `hr_selection/datasets/`:

| Dataset | Adapter | Sources | Raw Signals |
|---------|---------|---------|-------------|
| BigIdeasLab STEP | `BigIdeasAdapter` | 6 wrist devices | HR only |
| GalaxyPPG | `GalaxyPPGAdapter` | Galaxy Watch, E4 | PPG, ACC, ECG |
| PPG-DaLiA | `PPGDaLiAAdapter` | Dalia wrist | PPG, ACC, ECG |

Adapters yield `RawSession` objects (`hr_selection/datasets/schema.py`) — one per participant with time-aligned HR series, optional raw waveforms, activity labels, and reference HR.

Synthetic data (`hr_selection/synthetic/generate.py`) writes files in the exact real on-disk format, so swapping in real downloads only requires pointing `root` at the real data directory.

## Feature Engineering (`hr_selection/features/`)

### Per-Window Features (`build.py`)

For each canonical source in each window:

| Feature | Column pattern | Description |
|---------|---------------|-------------|
| HR value | `{src}__hr` | Mean HR in window |
| Missing flag | `{src}__missing` | 1.0 if source absent |
| Deviation from median | `{src}__dev_med` | HR minus cross-source median |
| Rolling std | `{src}__roll_std` | Temporal HR stability (5-window rolling) |

Context columns: `activity`, `skin_tone`.

### Signal Quality (`signal_quality.py`)

For raw-signal datasets (GalaxyPPG, PPG-DaLiA), computed from PPG/BVP + ACC windows:

| Feature | Description |
|---------|-------------|
| `q_hr_est` | FFT peak HR in cardiac band |
| `q_snr` | Spectral SNR |
| `q_entropy` | Spectral entropy (lower = cleaner) |
| `q_acc_mag` | RMS accelerometer magnitude |
| `q_perfusion` | AC/DC amplitude ratio |
| `q_template` | Autocorrelation periodicity score |

### Paper-Derived Priors (`priors/`)

Knowledge-based features from published device accuracy studies (`priors/knowledge.py`):

| Feature | Description |
|---------|-------------|
| `prior_mae` | Expected MAE for device family × activity context |
| `prior_bias_abs` | Expected absolute bias |
| `prior_trust` | Coverage/confidence score |
| `prior_rank` | Rank among all sources for this context |

Lookup in `priors/lookup.py` maps activity strings to context buckets (rest, daily_living, walking, exercise, sleep) and adjusts for skin tone (BigIdeasLab).

## Models (`hr_selection/models/`)

### Prior Baseline (`prior_baseline.py`)

No training. Ranks available sources by `-prior_mae` with masked softmax. Cold-start baseline.

### Classical (`classical.py`)

Multiclass classifier over canonical sources:
- Default: `HistGradientBoostingClassifier` (native NaN handling)
- Optional: LightGBM (`backend="lightgbm"`)
- Subject-wise `GroupKFold` cross-validation (no subject leakage)
- Categorical context features (`activity`, `skin_tone`)

### Deep (`deep.py`)

1D-CNN over resampled raw PPG + ACC channels (256 samples). Hybrid model concatenates CNN embedding with HR-agreement auxiliary features. Requires PyTorch (`pip install -e '.[deep]'`). Skips HR-only datasets (BigIdeas).

## Training & Evaluation

### CLI (`hr_selection/train.py`)

```bash
# Classical model on all synthetic datasets
python -m hr_selection.train --dataset all --source synthetic --model classical

# LightGBM backend
python -m hr_selection.train --dataset galaxyppg --backend lightgbm

# Deep track (requires torch)
python -m hr_selection.train --dataset galaxyppg --model deep

# Ablation: no paper priors
python -m hr_selection.train --dataset all --no-priors
```

Options: `--dataset`, `--source` (synthetic|real), `--model` (classical|deep), `--backend` (hist|lightgbm), `--root` (override data path), `--out-dir`.

Artifacts saved to `data/hr_out/`:
- `metrics_{tag}.json` — full evaluation metrics
- `model_{tag}.joblib` — fitted classical model + feature column list (when `--save-model`)

### Metrics (`hr_selection/evaluate.py`)

| Metric | Description |
|--------|-------------|
| Top-1 / top-2 accuracy | Source selection correctness (masked to available sources) |
| Macro-F1 | Per-class F1 averaged |
| HR MAE (model) | Error of selected source vs reference |
| HR MAE (oracle) | Best possible source (lower bound) |
| HR MAE (cross-source median) | Naive baseline |
| HR MAE (paper prior) | Prior-only baseline |
| HR MAE (best single device) | Always-pick-one baseline |
| Per-activity / per-dataset breakdowns | Stratified analysis |
| Confusion matrix | Source selection errors |

Success criterion: `beats_baselines` = model MAE < best single device AND < cross-source median.

## Personal ("Living") Layer

The population classical model serves as a **baseline prior**. On top of it, the personal layer learns each user's device habits from ingested `unified_heart_rate` data.

### Two-layer architecture

| Layer | What it learns | Data source |
|-------|----------------|-------------|
| Population baseline | Which device family is generally most trustworthy in context X | Research datasets (BigIdeas, GalaxyPPG, PPG-DaLiA) with ECG reference |
| Personal living model | Per-device offset vs a trusted anchor, tightening CIs, habitual resting HR | User's `unified_heart_rate` in Postgres |

### Trusted device

Each user gets one **trusted source** (auto-selected by lowest paper prior MAE, stable once set). All other devices learn `offset = hr_source - hr_trusted`. The trusted device's offset is pinned at 0.

### Nightly job

`app/jobs/personal_hr.py` runs daily (02:00 UTC via scheduler, or manually via `python -m app personal-hr`):

1. Incrementally reads new `unified_heart_rate` since the `personal_hr` cursor
2. Aligns multi-source readings into time windows (`hr_selection/personal/align.py`)
3. Updates per-device offset stats (Welford + EWMA) and tightens confidence intervals
4. Updates habitual baseline profile (resting HR by context, circadian curve)
5. Detects anomalies (e.g. firmware shift of ~3 bpm) and writes to `hr_anomalies`

### Anomaly detection

Compares the most recent batch of offsets to the stored distribution. Fires when the batch mean exceeds the stored CI by a configurable threshold (default 3 bpm) or z-score > 2.5. Single-source users fall back to self-referential drift detection against their habitual resting HR.

### Inference

`hr_selection/personal/infer.py` combines:

- Population baseline probabilities from the saved classical joblib artifact
- Per-user personal offset corrections and confidence bands

Returns a corrected personal HR plus `confidence_low` / `confidence_high`.

### Package layout (personal)

```
hr_selection/personal/
├── estimator.py    # Welford + EWMA offset learning, CI tightening
├── anomaly.py      # Batch z-test / CI breach detection
├── profile.py      # Habitual resting HR by context + circadian
├── align.py        # Multi-source window alignment from unified data
├── trusted.py      # Trusted device selection
└── infer.py        # Baseline + personal offset inference
```

State persisted in Postgres: `personal_hr_state`, `hr_anomalies` (see [SCHEMA.md](SCHEMA.md)).

## Integration with Backend

The personal layer is wired to the backend:

1. Pull unified HR data via `app/explore.load_unified("heart_rate")`
2. Nightly job in `app/jobs/personal_hr.py` updates per-user state
3. Inference via `hr_selection/personal/infer.py` with saved joblib artifact from `data/hr_out/`

The population training pipeline (`features/build.py`) remains on research datasets; the personal layer consumes real provider data.

## Package Layout

```
hr_selection/
├── config.py           # Central constants
├── train.py            # Training CLI
├── evaluate.py         # Metrics computation
├── datasets/           # Adapters (bigideas, galaxyppg, ppg_dalia)
├── features/           # Feature builder + signal quality
├── priors/             # Paper-derived knowledge + lookup
├── models/             # prior_baseline, classical, deep
├── personal/           # Living model: offset learning, anomaly, inference
└── synthetic/          # Synthetic data generators
```
