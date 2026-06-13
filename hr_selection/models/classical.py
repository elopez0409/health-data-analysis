"""Classical source-selection baseline.

Multiclass classifier over canonical sources using
``sklearn.ensemble.HistGradientBoostingClassifier`` (native NaN handling, and
native categorical support on sklearn >= 1.4). Optional LightGBM backend behind
``backend="lightgbm"`` (import guarded).

Evaluation uses **subject-wise** ``GroupKFold`` (leave-group-out when there are
few subjects) so no subject leaks between train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import __version__ as _sklearn_version
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from hr_selection import config
from hr_selection.features.build import CONTEXT_FEATURES, feature_columns

_SK_MAJOR, _SK_MINOR = (int(p) for p in _sklearn_version.split(".")[:2])
_NATIVE_CATEGORICAL = (_SK_MAJOR, _SK_MINOR) >= (1, 4)


def build_design_matrix(df: pd.DataFrame, use_priors: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the model input matrix and the list of categorical columns.

    Context features (``activity``, ``skin_tone``) are exposed as native pandas
    categoricals when the installed sklearn supports it, otherwise ordinal-encoded
    to NaN-aware numeric columns (HGB still splits on them fine).
    """
    X = df[feature_columns(use_priors=use_priors)].copy()

    if _NATIVE_CATEGORICAL:
        X["activity"] = df["activity"].astype("category")
        X["skin_tone"] = df["skin_tone"].astype("category")
        cat_cols = list(CONTEXT_FEATURES)
    else:
        act = pd.Categorical(df["activity"].astype("object"))
        act_codes = act.codes.astype(float)
        act_codes[act_codes < 0] = np.nan
        X["activity"] = act_codes
        X["skin_tone"] = df["skin_tone"].astype(float)
        cat_cols = []
    return X, cat_cols


def make_model(
    backend: str = "hist",
    categorical: list[str] | None = None,
    random_state: int = config.SEED,
    max_iter: int = 300,
):
    """Construct the classifier for the chosen backend."""
    if backend == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "backend='lightgbm' requires lightgbm. Install with: pip install -e '.[ml-lightgbm]'"
            ) from exc
        return LGBMClassifier(
            n_estimators=max_iter,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbose=-1,
        )

    kwargs: dict = dict(
        max_iter=max_iter,
        learning_rate=0.08,
        l2_regularization=1.0,
        max_leaf_nodes=31,
        random_state=random_state,
    )
    if _NATIVE_CATEGORICAL and categorical:
        kwargs["categorical_features"] = categorical
    return HistGradientBoostingClassifier(**kwargs)


def train_classical(
    df: pd.DataFrame,
    backend: str = "hist",
    n_splits: int = 5,
    random_state: int = config.SEED,
    max_iter: int = 300,
    use_priors: bool = True,
) -> dict:
    """Subject-wise cross-validated training.

    Returns out-of-fold class probabilities aligned to the full canonical label
    space plus a model fit on all data.
    """
    X, cat_cols = build_design_matrix(df, use_priors=use_priors)
    y = df["label"].to_numpy()
    groups = df["group"].to_numpy()

    n = len(df)
    n_classes = len(config.CANONICAL_SOURCES)
    oof_proba = np.zeros((n, n_classes), dtype=float)

    unique_labels = np.unique(y)
    n_groups = len(np.unique(groups))

    if unique_labels.size < 2 or n_groups < 2:
        # Degenerate selection (e.g. PPG-DaLiA alone): predict the only source.
        for lbl in unique_labels:
            oof_proba[:, lbl] = 1.0 / unique_labels.size
        model = None
        if unique_labels.size >= 1:
            oof_proba[:] = 0.0
            oof_proba[:, unique_labels[0]] = 1.0
        return {
            "oof_proba": oof_proba,
            "model": model,
            "X": X,
            "y": y,
            "groups": groups,
            "categorical": cat_cols,
            "degenerate": True,
            "n_splits": 0,
        }

    splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=splits)
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = make_model(backend, cat_cols, random_state, max_iter)
        clf.fit(X.iloc[train_idx], y[train_idx])
        proba = clf.predict_proba(X.iloc[test_idx])
        for j, cls in enumerate(clf.classes_):
            oof_proba[test_idx, cls] = proba[:, j]

    final = make_model(backend, cat_cols, random_state, max_iter)
    final.fit(X, y)

    return {
        "oof_proba": oof_proba,
        "model": final,
        "X": X,
        "y": y,
        "groups": groups,
        "categorical": cat_cols,
        "degenerate": False,
        "n_splits": splits,
    }
