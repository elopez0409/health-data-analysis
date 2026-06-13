"""Lookup paper-derived priors for a canonical source in a given context."""

from __future__ import annotations

import math
from functools import lru_cache

from hr_selection import config
from hr_selection.priors.knowledge import (
    CONTEXTS,
    DEVICE_FAMILY,
    GLOBAL_DEFAULT_BIAS,
    GLOBAL_DEFAULT_MAE,
    PAPERS,
    PRIORS,
    SKIN_TONE_ADJ,
    TIER_TO_MAE,
    PriorRecord,
)

# Activity string -> context bucket (DaLiA + BigIdeas + generic).
_ACTIVITY_TO_CONTEXT: dict[str, str] = {
    # BigIdeasLab_STEP
    "rest": "rest",
    "breathe": "rest",
    "activity": "exercise",
    "type": "daily_living",
    # PPG-DaLiA
    "sitting": "daily_living",
    "driving": "daily_living",
    "lunch": "daily_living",
    "working": "daily_living",
    "walking": "walking",
    "stairs": "walking",
    "soccer": "exercise",
    "cycling": "exercise",
    "transient": "unknown",
    # Generic
    "sleep": "sleep",
    "resting": "rest",
    "run": "exercise",
    "running": "exercise",
    "exercise": "exercise",
    "unknown": "unknown",
    "nan": "unknown",
    "none": "unknown",
}


def bucket_activity(activity: str | None) -> str:
    """Map a window activity label to a canonical context bucket."""
    if activity is None:
        return "unknown"
    key = str(activity).strip().lower()
    if not key or key in ("nan", "none", ""):
        return "unknown"
    return _ACTIVITY_TO_CONTEXT.get(key, "unknown")


def _skin_bin(skin_tone: float) -> int | None:
    if skin_tone is None or (isinstance(skin_tone, float) and math.isnan(skin_tone)):
        return None
    try:
        v = int(round(float(skin_tone)))
    except (TypeError, ValueError):
        return None
    if 1 <= v <= 6:
        return v
    return None


def _record_to_mae(rec: PriorRecord) -> float:
    if rec.metric == "mae_bpm":
        return float(rec.value)
    if rec.metric == "tier":
        return TIER_TO_MAE.get(int(rec.value), GLOBAL_DEFAULT_MAE)
    return GLOBAL_DEFAULT_MAE


def _record_to_bias(rec: PriorRecord) -> float:
    if rec.metric == "bias_bpm":
        return abs(float(rec.value))
    return GLOBAL_DEFAULT_BIAS


def _weighted_aggregate(records: list[PriorRecord]) -> tuple[float, float, float]:
    """Trust-weighted MAE, bias, and coverage score from matching records."""
    if not records:
        return GLOBAL_DEFAULT_MAE, GLOBAL_DEFAULT_BIAS, 0.0

    mae_num = bias_num = trust_sum = 0.0
    for rec in records:
        w = PAPERS.get(rec.paper, {}).get("trust_weight", 0.5)
        mae_num += w * _record_to_mae(rec)
        bias_num += w * _record_to_bias(rec)
        trust_sum += w
    if trust_sum <= 0:
        return GLOBAL_DEFAULT_MAE, GLOBAL_DEFAULT_BIAS, 0.0
    return mae_num / trust_sum, bias_num / trust_sum, min(1.0, trust_sum / 3.0)


def _records_for(family: str, context: str) -> list[PriorRecord]:
    exact = [r for r in PRIORS if r.family == family and r.context == context]
    if exact:
        return exact
    # family x any context -> family overall
    family_any = [r for r in PRIORS if r.family == family]
    if family_any:
        return family_any
    return [r for r in PRIORS]


@lru_cache(maxsize=4096)
def _family_prior(family: str, context: str, skin_bin: int | None) -> tuple[float, float, float]:
    """Cached (mae, bias_abs, trust) for a device family x context."""
    mae, bias, trust = _weighted_aggregate(_records_for(family, context))
    if skin_bin is not None:
        mae += SKIN_TONE_ADJ.get(skin_bin, 0.0)
    return mae, bias, trust


def _prior_rank(prior_mae: float, context: str, skin_bin: int | None) -> float:
    """Rank of this source's prior_mae among all canonical sources (1=best)."""
    maes = []
    for src in config.CANONICAL_SOURCES:
        fam = DEVICE_FAMILY.get(src)
        if fam is None:
            continue
        m, _, _ = _family_prior(fam, context, skin_bin)
        maes.append(m)
    if not maes:
        return float("nan")
    sorted_maes = sorted(maes)
    try:
        rank = sorted_maes.index(prior_mae) + 1
    except ValueError:
        rank = len(sorted_maes)
    return float(rank)


def source_prior(
    source: str,
    context: str,
    skin_tone: float | None = None,
) -> dict[str, float]:
    """Return prior features for one canonical source in a context.

    Keys: ``prior_mae``, ``prior_bias_abs``, ``prior_trust``, ``prior_rank``.
    """
    if context not in CONTEXTS:
        context = "unknown"
    family = DEVICE_FAMILY.get(source)
    skin_bin = _skin_bin(skin_tone) if skin_tone is not None else None

    if family is None:
        return {
            "prior_mae": GLOBAL_DEFAULT_MAE,
            "prior_bias_abs": GLOBAL_DEFAULT_BIAS,
            "prior_trust": 0.0,
            "prior_rank": float(len(config.CANONICAL_SOURCES)),
        }

    mae, bias, trust = _family_prior(family, context, skin_bin)
    rank = _prior_rank(mae, context, skin_bin)
    return {
        "prior_mae": float(mae),
        "prior_bias_abs": float(bias),
        "prior_trust": float(trust),
        "prior_rank": float(rank),
    }
