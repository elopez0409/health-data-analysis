"""Curated knowledge base of device x context accuracy priors from validation papers.

Each record is tagged with a per-paper ``trust_weight`` so preprints and
systematic reviews contribute less than direct ECG-referenced studies. Numeric
MAE/bias values are best-effort extractions; ordinal ``tier`` records (1=best,
5=worst) are converted to MAE via ``TIER_TO_MAE`` when no numeric value exists.

Papers encoded
--------------
- Bent2020   : optical HR inaccuracy; activity + skin tone + missingness (ECG ref)
- Pasadyn2019: athlete / high-intensity exercise (ECG ref)
- Wang2017   : wrist-worn monitors at rest (ECG ref)
- Nelson2019 : 24 h ecologically valid daily living (ambulatory ECG ref)
- Helmer2022 : postoperative resting / low-motion (clinical ECG ref)
- Fuller2020 : systematic review of commercial wearables (multiple refs)
- Dial2025   : nocturnal resting HR / HRV (ECG ref)
- Kostrna2026: diverse populations preprint (Polar H10 ref, lower trust)

Many device families (TomTom, Mio, Withings, WHOOP, Oura, Polar) are stored for
future canonical sources but are dormant until those devices appear in a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Context buckets (shared across datasets via bucket_activity in lookup.py)
# ---------------------------------------------------------------------------
CONTEXTS: list[str] = [
    "rest",
    "sleep",
    "daily_living",
    "walking",
    "exercise",
    "unknown",
]

# Ordinal tier -> expected MAE (bpm) when a paper reports only qualitative rank.
TIER_TO_MAE: dict[int, float] = {1: 3.0, 2: 5.0, 3: 8.0, 4: 12.0, 5: 18.0}

# Global fallback when no family-specific record exists.
GLOBAL_DEFAULT_MAE = 10.0
GLOBAL_DEFAULT_BIAS = 0.0

# ---------------------------------------------------------------------------
# Paper registry
# ---------------------------------------------------------------------------
PAPERS: dict[str, dict] = {
    "Bent2020": {
        "citation": "Bent et al. 2020 — Investigating sources of inaccuracy in wearable optical HR sensors",
        "reference_type": "ecg",
        "trust_weight": 1.0,
    },
    "Pasadyn2019": {
        "citation": "Pasadyn et al. 2019 — Accuracy of commercially available HR monitors in athletes",
        "reference_type": "ecg",
        "trust_weight": 0.95,
    },
    "Wang2017": {
        "citation": "Wang et al. 2017 — Accuracy of Wrist-Worn Heart Rate Monitors (JAMA Cardiology)",
        "reference_type": "ecg",
        "trust_weight": 0.9,
    },
    "Nelson2019": {
        "citation": "Nelson et al. 2019 — Consumer wearable HR during ecologically valid 24 h period",
        "reference_type": "ambulatory_ecg",
        "trust_weight": 0.9,
    },
    "Helmer2022": {
        "citation": "Helmer et al. 2022 — HR accuracy in postoperative patients (clinical ECG)",
        "reference_type": "clinical_ecg",
        "trust_weight": 0.95,
    },
    "Fuller2020": {
        "citation": "Fuller et al. 2020 — Systematic review of commercial wearable validity",
        "reference_type": "review",
        "trust_weight": 0.6,
    },
    "Dial2025": {
        "citation": "Dial et al. 2025 — Nocturnal resting HR and HRV validation",
        "reference_type": "ecg",
        "trust_weight": 0.85,
    },
    "Kostrna2026": {
        "citation": "Kostrna et al. 2026 preprint — PPG HR accuracy in diverse populations",
        "reference_type": "polar_h10",
        "trust_weight": 0.4,
    },
}

# ---------------------------------------------------------------------------
# Canonical source -> device family (brand line)
# ---------------------------------------------------------------------------
DEVICE_FAMILY: dict[str, str] = {
    "apple_watch": "apple",
    "fitbit": "fitbit",
    "garmin": "garmin",
    "miband": "xiaomi",
    "empatica": "empatica",
    "e4": "empatica",
    "dalia_wrist": "empatica",
    "biovotion": "biovotion",
    "galaxy_watch": "samsung",
}

# Dormant families (no canonical source yet) kept for KB completeness.
DORMANT_FAMILIES = frozenset(
    {"tomtom", "mio", "basis", "withings", "whoop", "oura", "polar", "samsung"}
)

# ---------------------------------------------------------------------------
# Skin-tone additive MAE penalty (Fitzpatrick 1-6) from Bent et al. 2020.
# Darker skin tones show higher optical-HR error in controlled studies.
# ---------------------------------------------------------------------------
SKIN_TONE_ADJ: dict[int, float] = {
    1: 0.0,
    2: 0.5,
    3: 1.0,
    4: 2.0,
    5: 3.5,
    6: 5.0,
}

MetricKind = Literal["mae_bpm", "bias_bpm", "tier"]


@dataclass(frozen=True)
class PriorRecord:
    paper: str
    family: str
    context: str
    metric: MetricKind
    value: float


def _r(paper: str, family: str, context: str, metric: MetricKind, value: float) -> PriorRecord:
    return PriorRecord(paper=paper, family=family, context=context, metric=metric, value=value)


# Hand-encoded priors (numeric where reported/estimated; tier otherwise).
# Values are approximate and auditable via ``paper`` + ``trust_weight``.
PRIORS: list[PriorRecord] = [
    # --- Bent 2020 (primary optical HR study; BigIdeas devices) ---
    _r("Bent2020", "apple", "rest", "mae_bpm", 4.5),
    _r("Bent2020", "apple", "exercise", "mae_bpm", 14.0),
    _r("Bent2020", "apple", "daily_living", "mae_bpm", 7.0),
    _r("Bent2020", "empatica", "rest", "mae_bpm", 4.0),
    _r("Bent2020", "empatica", "exercise", "mae_bpm", 18.0),
    _r("Bent2020", "empatica", "daily_living", "mae_bpm", 8.0),
    _r("Bent2020", "fitbit", "rest", "mae_bpm", 5.5),
    _r("Bent2020", "fitbit", "exercise", "mae_bpm", 12.0),
    _r("Bent2020", "fitbit", "daily_living", "mae_bpm", 7.5),
    _r("Bent2020", "garmin", "rest", "mae_bpm", 5.0),
    _r("Bent2020", "garmin", "exercise", "mae_bpm", 11.0),
    _r("Bent2020", "garmin", "daily_living", "mae_bpm", 7.0),
    _r("Bent2020", "xiaomi", "rest", "mae_bpm", 6.5),
    _r("Bent2020", "xiaomi", "exercise", "mae_bpm", 15.0),
    _r("Bent2020", "biovotion", "rest", "mae_bpm", 4.5),
    _r("Bent2020", "biovotion", "exercise", "mae_bpm", 13.0),
    _r("Bent2020", "biovotion", "daily_living", "mae_bpm", 7.0),
    # --- Pasadyn 2019 (athletes / high intensity) ---
    _r("Pasadyn2019", "apple", "exercise", "mae_bpm", 16.0),
    _r("Pasadyn2019", "fitbit", "exercise", "mae_bpm", 14.0),
    _r("Pasadyn2019", "garmin", "exercise", "mae_bpm", 10.0),
    _r("Pasadyn2019", "polar", "exercise", "mae_bpm", 3.5),
    # --- Wang 2017 (resting wrist PPG) ---
    _r("Wang2017", "apple", "rest", "mae_bpm", 5.0),
    _r("Wang2017", "fitbit", "rest", "mae_bpm", 6.0),
    _r("Wang2017", "mio", "rest", "mae_bpm", 8.0),
    _r("Wang2017", "basis", "rest", "mae_bpm", 9.0),
    # --- Nelson 2019 (24 h daily living) ---
    _r("Nelson2019", "apple", "daily_living", "mae_bpm", 6.5),
    _r("Nelson2019", "apple", "walking", "mae_bpm", 8.0),
    _r("Nelson2019", "apple", "exercise", "mae_bpm", 12.0),
    _r("Nelson2019", "fitbit", "daily_living", "mae_bpm", 7.0),
    _r("Nelson2019", "fitbit", "walking", "mae_bpm", 9.0),
    _r("Nelson2019", "fitbit", "sleep", "mae_bpm", 5.5),
    # --- Helmer 2022 (clinical resting / low motion) ---
    _r("Helmer2022", "apple", "rest", "mae_bpm", 3.5),
    _r("Helmer2022", "garmin", "rest", "mae_bpm", 3.0),
    _r("Helmer2022", "fitbit", "rest", "mae_bpm", 4.0),
    _r("Helmer2022", "withings", "rest", "mae_bpm", 4.5),
    # --- Fuller 2020 (systematic review; tier-based) ---
    _r("Fuller2020", "apple", "rest", "tier", 2),
    _r("Fuller2020", "fitbit", "rest", "tier", 2),
    _r("Fuller2020", "garmin", "rest", "tier", 2),
    _r("Fuller2020", "samsung", "rest", "tier", 3),
    _r("Fuller2020", "apple", "exercise", "tier", 4),
    _r("Fuller2020", "fitbit", "exercise", "tier", 4),
    # --- Dial 2025 (sleep / nocturnal rest) ---
    _r("Dial2025", "garmin", "sleep", "mae_bpm", 3.5),
    _r("Dial2025", "oura", "sleep", "mae_bpm", 2.5),
    _r("Dial2025", "whoop", "sleep", "mae_bpm", 3.0),
    _r("Dial2025", "polar", "sleep", "mae_bpm", 2.0),
    _r("Dial2025", "garmin", "rest", "mae_bpm", 3.0),
    # --- Kostrna 2026 preprint (diverse populations; lower trust) ---
    _r("Kostrna2026", "apple", "daily_living", "tier", 3),
    _r("Kostrna2026", "fitbit", "daily_living", "tier", 3),
    _r("Kostrna2026", "samsung", "daily_living", "tier", 3),
    _r("Kostrna2026", "garmin", "daily_living", "tier", 2),
    # --- GalaxyPPG: Samsung watch (limited direct citations; Fuller + Kostrna) ---
    _r("Fuller2020", "samsung", "exercise", "tier", 4),
    _r("Fuller2020", "samsung", "walking", "tier", 3),
    _r("Kostrna2026", "samsung", "exercise", "tier", 4),
    # --- PPG-DaLiA wrist = Empatica E4 (Bent primary) ---
    _r("Bent2020", "empatica", "walking", "mae_bpm", 10.0),
    _r("Bent2020", "empatica", "exercise", "mae_bpm", 18.0),
]
