"""Cold-start paper-prior ranker: no training, ranks sources by prior MAE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hr_selection import config
from hr_selection.priors.lookup import bucket_activity, source_prior


def prior_proba(df: pd.DataFrame, temperature: float = 1.0) -> np.ndarray:
    """Per-window pseudo-probabilities from paper priors.

    For each window, score available sources with ``-prior_mae`` (lower expected
    error -> higher score) and apply a masked softmax. Unavailable sources get 0.
    """
    n = len(df)
    n_classes = len(config.CANONICAL_SOURCES)
    scores = np.full((n, n_classes), -np.inf, dtype=float)

    for i, row in df.iterrows():
        context = bucket_activity(row.get("activity"))
        skin = row.get("skin_tone", float("nan"))
        for j, src in enumerate(config.CANONICAL_SOURCES):
            missing_col = f"{src}__missing"
            if missing_col in df.columns and row.get(missing_col, 1.0) == 1.0:
                continue
            prior = source_prior(src, context, skin)
            scores[i, j] = -prior["prior_mae"] / max(temperature, 1e-6)

    # Stable softmax per row (only over finite scores).
    proba = np.zeros((n, n_classes), dtype=float)
    for i in range(n):
        finite = np.isfinite(scores[i])
        if not finite.any():
            continue
        s = scores[i, finite]
        s = s - s.max()
        exp = np.exp(s)
        proba[i, finite] = exp / exp.sum()
    return proba
