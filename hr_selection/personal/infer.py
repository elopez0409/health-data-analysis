"""Inference: combine population baseline with per-user personal offsets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hr_selection import config
from hr_selection.models.classical import build_design_matrix
from hr_selection.personal.estimator import OffsetState
from hr_selection.personal.profile import BaselineProfile
from hr_selection.personal.trusted import PROVIDER_TO_CANONICAL, select_trusted_source


@dataclass
class PersonalHrResult:
    """Corrected personal HR for one time window."""

    corrected_hr: float
    selected_source: str
    raw_hr: float
    confidence_low: float
    confidence_high: float
    population_weight: float
    personal_offset: float

    def to_dict(self) -> dict:
        return {
            "corrected_hr": self.corrected_hr,
            "selected_source": self.selected_source,
            "raw_hr": self.raw_hr,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "population_weight": self.population_weight,
            "personal_offset": self.personal_offset,
        }


def load_baseline_artifact(path: Path | str) -> dict:
    """Load saved classical model joblib artifact."""
    return joblib.load(path)


def _provider_to_canonical(provider: str) -> str | None:
    return PROVIDER_TO_CANONICAL.get(provider)


def _build_inference_row(
    sources_hr: dict[str, float],
    *,
    activity: str = "unknown",
    skin_tone: float = float("nan"),
    use_priors: bool = True,
) -> pd.DataFrame:
    """Build a single-row feature table for baseline model scoring."""
    from hr_selection.features.build import feature_columns
    from hr_selection.priors.lookup import bucket_activity, source_prior

    row: dict = {
        "activity": activity,
        "skin_tone": skin_tone,
    }
    available = [s for s, hr in sources_hr.items() if not math.isnan(hr)]
    cross_med = float(np.nanmedian([sources_hr[s] for s in available])) if available else float("nan")
    context = bucket_activity(activity)

    for canonical in config.CANONICAL_SOURCES:
        provider_match = None
        for prov, canon in PROVIDER_TO_CANONICAL.items():
            if canon == canonical and prov in sources_hr:
                provider_match = prov
                break

        hr = sources_hr.get(provider_match, float("nan")) if provider_match else float("nan")
        missing = 1.0 if math.isnan(hr) else 0.0
        dev = float("nan") if math.isnan(hr) else hr - cross_med
        row[f"{canonical}__hr"] = hr
        row[f"{canonical}__missing"] = missing
        row[f"{canonical}__dev_med"] = dev
        row[f"{canonical}__roll_std"] = float("nan")
        for key in ["q_hr_est", "q_snr", "q_entropy", "q_acc_mag", "q_perfusion", "q_template"]:
            row[f"{canonical}__{key}"] = float("nan")
        if use_priors:
            prior = source_prior(canonical, context, skin_tone)
            for pk, pv in prior.items():
                row[f"{canonical}__{pk}"] = pv

    cols = feature_columns(use_priors=use_priors) + ["activity", "skin_tone"]
    return pd.DataFrame([{c: row.get(c, float("nan")) for c in cols}])


def _population_source_scores(
    artifact: dict,
    sources_hr: dict[str, float],
    *,
    activity: str = "unknown",
) -> dict[str, float]:
    """Map provider -> population trust score from baseline model."""
    use_priors = artifact.get("use_priors", True)
    df = _build_inference_row(sources_hr, activity=activity, use_priors=use_priors)
    model = artifact["model"]
    X, _ = build_design_matrix(df, use_priors=use_priors)
    proba = model.predict_proba(X)[0]

    scores: dict[str, float] = {}
    for j, cls_idx in enumerate(model.classes_):
        if cls_idx >= len(config.CANONICAL_SOURCES):
            continue
        canonical = config.CANONICAL_SOURCES[cls_idx]
        for prov, canon in PROVIDER_TO_CANONICAL.items():
            if canon == canonical and prov in sources_hr:
                scores[prov] = float(proba[j])
    return scores


def _personal_scores(
    sources_hr: dict[str, float],
    personal_state: dict[str, OffsetState],
    trusted_source: str,
) -> dict[str, float]:
    """Score sources by inverse absolute personal offset magnitude."""
    scores: dict[str, float] = {}
    for src, hr in sources_hr.items():
        if math.isnan(hr):
            continue
        if src == trusted_source:
            scores[src] = 1.0
            continue
        state = personal_state.get(src)
        if state is None or state.n_samples < 5:
            scores[src] = 0.5
            continue
        corrected = hr - state.offset_mean
        trusted_hr = sources_hr.get(trusted_source, hr)
        residual = abs(corrected - trusted_hr)
        scores[src] = 1.0 / (1.0 + residual)
    return scores


def infer_personal_hr(
    sources_hr: dict[str, float],
    personal_state: dict[str, Any],
    *,
    baseline_artifact: dict | None = None,
    trusted_source: str | None = None,
    activity: str = "unknown",
    population_weight: float = 0.4,
) -> PersonalHrResult:
    """Select best source and return corrected personal HR with confidence band.

    Combines population baseline probabilities (when artifact provided) with
    per-user offset learning. Applies personal offset correction to selected reading.
    """
    available = {s: hr for s, hr in sources_hr.items() if not math.isnan(hr)}
    if not available:
        return PersonalHrResult(
            corrected_hr=float("nan"),
            selected_source="",
            raw_hr=float("nan"),
            confidence_low=float("nan"),
            confidence_high=float("nan"),
            population_weight=population_weight,
            personal_offset=0.0,
        )

    existing_trusted = None
    if personal_state:
        first = next(iter(personal_state.values()), None)
        if first and hasattr(first, "trusted_source"):
            existing_trusted = getattr(first, "trusted_source", None)
        elif isinstance(first, dict):
            existing_trusted = first.get("trusted_source")

    trusted = trusted_source or select_trusted_source(
        list(available.keys()), existing_trusted=existing_trusted, context=activity
    )
    if trusted is None:
        trusted = list(available.keys())[0]

    offset_states: dict[str, OffsetState] = {}
    profiles: dict[str, BaselineProfile] = {}
    for src, data in (personal_state or {}).items():
        if isinstance(data, OffsetState):
            offset_states[src] = data
        elif isinstance(data, dict):
            offset_states[src] = OffsetState.from_dict(data)
            if "baseline_profile" in data:
                profiles[src] = BaselineProfile.from_dict(data["baseline_profile"])

    pop_scores = (
        _population_source_scores(baseline_artifact, available, activity=activity)
        if baseline_artifact
        else {}
    )
    pers_scores = _personal_scores(available, offset_states, trusted)

    combined: dict[str, float] = {}
    for src in available:
        pop = pop_scores.get(src, 0.5)
        pers = pers_scores.get(src, 0.5)
        combined[src] = (1.0 - population_weight) * pers + population_weight * pop

    selected = max(combined, key=combined.get)
    raw_hr = available[selected]

    offset = 0.0
    if selected != trusted and selected in offset_states:
        offset = offset_states[selected].offset_mean
        raw_hr = raw_hr - offset

    ci_half = 0.0
    if selected in offset_states and offset_states[selected].n_samples >= 5:
        state = offset_states[selected]
        if not math.isnan(state.ci_low) and not math.isnan(state.ci_high):
            ci_half = (state.ci_high - state.ci_low) / 2.0

    return PersonalHrResult(
        corrected_hr=raw_hr,
        selected_source=selected,
        raw_hr=available[selected],
        confidence_low=raw_hr - ci_half,
        confidence_high=raw_hr + ci_half,
        population_weight=population_weight,
        personal_offset=offset,
    )
