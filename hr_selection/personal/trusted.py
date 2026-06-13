"""Trusted device selection for the personal HR layer."""

from __future__ import annotations

from hr_selection.priors.knowledge import GLOBAL_DEFAULT_MAE
from hr_selection.priors.lookup import source_prior

# Provider names in unified_heart_rate.source -> device family for priors.
PROVIDER_TO_FAMILY: dict[str, str] = {
    "fitbit": "fitbit",
    "garmin": "garmin",
    "oura": "oura",
    "whoop": "whoop",
    "withings": "withings",
    "strava": "garmin",
    "apple": "apple",
    "polar": "polar",
}

# Provider -> canonical source key for source_prior lookup (when available).
PROVIDER_TO_CANONICAL: dict[str, str] = {
    "fitbit": "fitbit",
    "garmin": "garmin",
    "apple": "apple_watch",
    "strava": "garmin",
}

# Chest-strap / ECG-class sources preferred as trusted anchor.
REFERENCE_PRIORITY: list[str] = ["polar", "apple"]


def provider_prior_mae(provider: str, context: str = "rest") -> float:
    """Expected MAE for a provider via paper priors."""
    canonical = PROVIDER_TO_CANONICAL.get(provider)
    if canonical:
        return source_prior(canonical, context)["prior_mae"]

    family = PROVIDER_TO_FAMILY.get(provider)
    if family is None:
        return GLOBAL_DEFAULT_MAE

    from hr_selection.priors.knowledge import PRIORS, PAPERS, TIER_TO_MAE

    records = [r for r in PRIORS if r.family == family and r.context == context]
    if not records:
        records = [r for r in PRIORS if r.family == family]
    if not records:
        return GLOBAL_DEFAULT_MAE

    mae_num = 0.0
    trust_sum = 0.0
    for rec in records:
        w = PAPERS.get(rec.paper, {}).get("trust_weight", 0.5)
        if rec.metric == "mae_bpm":
            mae_num += w * float(rec.value)
        elif rec.metric == "tier":
            mae_num += w * TIER_TO_MAE.get(int(rec.value), GLOBAL_DEFAULT_MAE)
        trust_sum += w
    return mae_num / trust_sum if trust_sum > 0 else GLOBAL_DEFAULT_MAE


def select_trusted_source(
    available_sources: list[str],
    *,
    existing_trusted: str | None = None,
    context: str = "rest",
) -> str | None:
    """Pick the per-user trusted anchor device.

    If ``existing_trusted`` is still available, keep it stable. Otherwise choose
    the source with lowest population prior MAE, preferring reference-class devices.
    """
    if not available_sources:
        return None

    if existing_trusted and existing_trusted in available_sources:
        return existing_trusted

    def sort_key(src: str) -> tuple[int, float, str]:
        ref_rank = (
            REFERENCE_PRIORITY.index(PROVIDER_TO_FAMILY.get(src, ""))
            if PROVIDER_TO_FAMILY.get(src) in REFERENCE_PRIORITY
            else 999
        )
        mae = provider_prior_mae(src, context)
        return (ref_rank, mae, src)

    return min(available_sources, key=sort_key)
