"""Deep source-selection track: a small 1D-CNN over raw per-source signals.

Secondary / runnable-but-minimal. Consumes per-window raw PPG + ACC-magnitude
channels (one source pair of channels per canonical source in the dataset) and
predicts source logits. Subject-wise ``GroupKFold`` gives leave-one-subject-out
out-of-fold predictions, fed to the same ``evaluate`` as the classical track.

The torch import is guarded so ``import hr_selection`` works without torch.
Only the raw-signal datasets (GalaxyPPG, PPG-DaLiA) are supported; BigIdeas is
HR-only and is skipped for the deep track.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hr_selection import config
from hr_selection.datasets.base import get_adapter
from hr_selection.features.signal_quality import QUALITY_KEYS

try:  # guarded optional dependency
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    nn = None
    TORCH_AVAILABLE = False

RESAMPLE_LEN = 256


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "The deep track requires PyTorch. Install it with: pip install -e '.[deep]'"
        )


def select_device():
    """Pick the fastest available torch device (CUDA > Apple MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resample(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(length, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=x.shape[0])
    dst = np.linspace(0.0, 1.0, num=length)
    return np.interp(dst, src, x).astype(np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd + 1e-6)


def _windows_for_session(session, ds_sources: list[str]) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """Window one ``RawSession`` -> (meta_rows, raw_channel_stacks, aux_features).

    ``aux_features`` is the HR-agreement signal the classical track relies on:
    per source [hr/100, missing, (hr-median)/20, |hr-median|/20]. It never sees
    the reference, so it stays a fair feature. This is concatenated with the CNN
    embedding to form a hybrid model.
    """
    from hr_selection.features.signal_quality import ppg_quality

    win = config.WINDOW_SEC
    shift = config.SHIFT_SEC
    duration = session.duration_sec

    meta_rows: list[dict] = []
    raw_list: list[np.ndarray] = []
    feat_list: list[np.ndarray] = []

    k = 0
    t0 = 0.0
    while t0 + win <= duration + 1e-6:
        t1 = t0 + win
        ref_hr = session.reference_hr_at(t0, t1)
        if np.isnan(ref_hr):
            k += 1
            t0 = k * shift
            continue

        hr_by_src: dict[str, float] = {}
        channels: list[np.ndarray] = []
        for src_key in ds_sources:
            src = session.sources.get(src_key)
            ppg_arr = np.zeros(RESAMPLE_LEN, dtype=np.float32)
            accmag_arr = np.zeros(RESAMPLE_LEN, dtype=np.float32)
            hr_val = float("nan")
            q_hr = float("nan")
            if src is not None and "ppg" in src.raw:
                ppg_ch = src.raw["ppg"]
                ppg_seg = ppg_ch.slice_seconds(t0, t1)
                if ppg_seg.size:
                    ppg_arr = _normalize(_resample(ppg_seg, RESAMPLE_LEN))
                acc_ch = src.raw.get("acc")
                if acc_ch is not None:
                    acc_seg = acc_ch.slice_seconds(t0, t1)
                    if acc_seg.size:
                        acc_seg = np.asarray(acc_seg, dtype=float)
                        if acc_seg.ndim == 2:
                            mag = np.sqrt(np.sum((acc_seg - acc_seg.mean(0)) ** 2, axis=1))
                        else:
                            mag = np.abs(acc_seg - acc_seg.mean())
                        accmag_arr = _normalize(_resample(mag, RESAMPLE_LEN))
                q_hr = ppg_quality(ppg_seg, ppg_ch.fs).get("q_hr_est", float("nan"))
            if src is not None and src.hr.size > 0:
                v = src.hr_at(t0, t1)
                hr_val = v if not np.isnan(v) else q_hr
            else:
                hr_val = q_hr
            hr_by_src[src_key] = hr_val
            channels.append(ppg_arr)
            channels.append(accmag_arr)

        available = [s for s, hr in hr_by_src.items() if not np.isnan(hr)]
        if not available:
            k += 1
            t0 = k * shift
            continue

        best = min(available, key=lambda s: abs(hr_by_src[s] - ref_hr))
        label = config.SOURCE_INDEX[best]

        cross_median = float(np.median([hr_by_src[s] for s in available]))
        aux: list[float] = []
        for src_key in ds_sources:
            hr = hr_by_src.get(src_key, float("nan"))
            if np.isnan(hr):
                aux.extend([0.0, 1.0, 0.0, 0.0])
            else:
                dev = hr - cross_median
                aux.extend([hr / 100.0, 0.0, dev / 20.0, abs(dev) / 20.0])

        row = {
            "dataset": session.dataset,
            "subject_id": session.subject_id,
            "group": f"{session.dataset}:{session.subject_id}",
            "t_start": t0,
            "reference_hr": ref_hr,
            "activity": session.activity_at(t0, t1),
            "label": label,
        }
        for src_key in config.CANONICAL_SOURCES:
            hr = hr_by_src.get(src_key, float("nan"))
            row[f"{src_key}__hr"] = hr
            row[f"{src_key}__missing"] = 1.0 if np.isnan(hr) else 0.0
        meta_rows.append(row)
        raw_list.append(np.stack(channels, axis=0))
        feat_list.append(np.asarray(aux, dtype=np.float32))

        k += 1
        t0 = k * shift

    return meta_rows, raw_list, feat_list


def _build_windows_for_key(dataset: str, root, key: str, ds_sources: list[str]):
    """Worker: load one session from disk and window it (bounded memory)."""
    adapter = get_adapter(dataset, root)
    session = adapter.load_session(key)
    return _windows_for_session(session, ds_sources)


def build_deep_windows(dataset: str, root=None, n_jobs: int = -1):
    """Build (meta_df, X_raw, X_feat, labels, groups) for one raw-signal dataset.

    Window-building is parallelized across sessions: each worker loads a single
    session from disk so memory stays bounded even for the ~12 GB PPG-DaLiA set.

    ``meta_df`` carries exactly the columns ``evaluate`` needs (reference_hr,
    label, activity, dataset, and per-source ``__hr`` / ``__missing``).
    ``X_raw`` has shape (n_windows, n_channels, RESAMPLE_LEN); ``X_feat`` has the
    per-window HR-agreement features for the hybrid head.
    """
    if dataset not in config.RAW_SIGNAL_DATASETS:
        raise ValueError(f"Deep track only supports raw-signal datasets {config.RAW_SIGNAL_DATASETS}, got '{dataset}'.")

    ds_sources = config.DATASET_SOURCES[dataset]
    ds_root = root if root is not None else config.DATASET_DIRS[dataset]
    adapter = get_adapter(dataset, ds_root)

    keys = adapter.session_keys() if hasattr(adapter, "session_keys") else None

    meta_rows: list[dict] = []
    raw_list: list[np.ndarray] = []
    feat_list: list[np.ndarray] = []

    if keys and n_jobs != 1 and len(keys) > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_build_windows_for_key)(dataset, str(ds_root), k, ds_sources) for k in keys
        )
        for m, r, f in results:
            meta_rows.extend(m)
            raw_list.extend(r)
            feat_list.extend(f)
    else:
        for session in adapter.iter_sessions():
            m, r, f = _windows_for_session(session, ds_sources)
            meta_rows.extend(m)
            raw_list.extend(r)
            feat_list.extend(f)

    meta_df = pd.DataFrame(meta_rows)
    X_raw = np.stack(raw_list, axis=0).astype(np.float32)
    X_feat = np.stack(feat_list, axis=0).astype(np.float32)
    labels = meta_df["label"].to_numpy()
    groups = meta_df["group"].to_numpy()
    return meta_df, X_raw, X_feat, labels, groups


if TORCH_AVAILABLE:

    class SourceCNN(nn.Module):
        """Hybrid 1D-CNN: raw per-source channels + HR-agreement features -> logits."""

        def __init__(self, n_channels: int, n_feat: int, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(64 + n_feat, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, n_classes),
            )

        def forward(self, x, feat):
            emb = self.features(x)
            return self.head(torch.cat([emb, feat], dim=1))


def _class_weights(y_tr, n_classes, device):
    """Balanced class weights so rare best-sources aren't ignored (fixes macro-F1)."""
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    counts[counts == 0] = np.nan
    w = np.nansum(counts) / (np.count_nonzero(~np.isnan(counts)) * counts)
    w = np.nan_to_num(w, nan=0.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _train_one_fold(X_tr, F_tr, y_tr, n_channels, n_feat, n_classes, epochs, lr, seed, device, batch_size=128):
    torch.manual_seed(seed)
    model = SourceCNN(n_channels, n_feat, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y_tr, n_classes, device))
    Xt = torch.tensor(X_tr)
    Ft = torch.tensor(F_tr)
    yt = torch.tensor(y_tr, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(Xt, Ft, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, fb, yb in dl:
            xb = xb.to(device)
            fb = fb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            opt.step()
    return model


def _oof_for_dataset(X_raw, X_feat, labels, groups, n_classes, epochs, lr, seed, device):
    from sklearn.model_selection import GroupKFold

    n = X_raw.shape[0]
    oof = np.zeros((n, n_classes), dtype=float)
    n_groups = len(np.unique(groups))
    if np.unique(labels).size < 2 or n_groups < 2:
        # Degenerate single-source dataset: predict the only source.
        for lbl in np.unique(labels):
            oof[:, lbl] = 1.0
        return oof

    splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=splits)
    for tr, te in gkf.split(X_raw, labels, groups):
        model = _train_one_fold(
            X_raw[tr], X_feat[tr], labels[tr], X_raw.shape[1], X_feat.shape[1],
            n_classes, epochs, lr, seed, device,
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_raw[te]).to(device), torch.tensor(X_feat[te]).to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        oof[te] = proba
    return oof


def train_deep(
    datasets: list[str] | str,
    source: str = "synthetic",
    root=None,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = config.SEED,
    n_jobs: int = -1,
) -> dict:
    """Train the deep track on the raw-signal datasets; returns df + oof_proba.

    BigIdeas (HR-only) is skipped. For multiple datasets a separate CNN is
    trained per dataset (channel counts differ) and the out-of-fold predictions
    are stitched together for a unified evaluation.
    """
    _require_torch()
    if isinstance(datasets, str):
        datasets = config.ALL_DATASETS if datasets == "all" else [datasets]

    raw_datasets = [d for d in datasets if d in config.RAW_SIGNAL_DATASETS]
    skipped = [d for d in datasets if d not in config.RAW_SIGNAL_DATASETS]
    if not raw_datasets:
        raise ValueError(
            f"Deep track needs a raw-signal dataset {sorted(config.RAW_SIGNAL_DATASETS)}; got {datasets}."
        )

    device = select_device()
    n_classes = len(config.CANONICAL_SOURCES)
    metas: list[pd.DataFrame] = []
    oofs: list[np.ndarray] = []
    for ds in raw_datasets:
        meta_df, X_raw, X_feat, labels, groups = build_deep_windows(ds, root, n_jobs=n_jobs)
        oof = _oof_for_dataset(X_raw, X_feat, labels, groups, n_classes, epochs, lr, seed, device)
        metas.append(meta_df)
        oofs.append(oof)

    df = pd.concat(metas, ignore_index=True)
    oof_proba = np.concatenate(oofs, axis=0)
    return {
        "df": df,
        "oof_proba": oof_proba,
        "skipped_datasets": skipped,
        "datasets": raw_datasets,
        "device": str(device),
    }
