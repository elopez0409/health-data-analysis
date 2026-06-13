"""Synthetic HR-source datasets written in the EXACT real on-disk formats.

A single physiological model produces a ground-truth HR trajectory and a motion
level per subject.  The reference (ECG / Polar) tracks truth almost perfectly;
each device source = truth + device bias + motion-correlated error (worse during
activity) + lag + dropouts + (for BigIdeas) skin-tone degradation.  This makes
*which* source is closest genuinely context-dependent, so source selection is a
non-trivial supervised problem.

For raw-signal datasets (GalaxyPPG, PPG-DaLiA) we synthesize actual waveforms
(ECG PQRST, PPG pulse + motion artifacts, ACC) so the signal-quality features
and the deep track see real signals.

Everything is reproducible from ``config.SEED``.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hr_selection import config
from hr_selection.synthetic import waveforms as wf

# --------------------------------------------------------------------------
# Activity -> (target HR bpm, target motion level 0..~1)
# --------------------------------------------------------------------------
ACTIVITY_PROFILE: dict[str, tuple[float, float]] = {
    # BigIdeasLab_STEP activities
    "Rest": (66.0, 0.05),
    "Breathe": (62.0, 0.05),
    "Type": (76.0, 0.30),
    "Activity": (122.0, 1.00),
    # PPG-DaLiA activities
    "sitting": (70.0, 0.05),
    "working": (78.0, 0.25),
    "driving": (74.0, 0.15),
    "lunch": (82.0, 0.20),
    "walking": (106.0, 0.80),
    "stairs": (120.0, 1.00),
    "cycling": (126.0, 0.90),
    "soccer": (140.0, 1.10),
}

BIGIDEAS_ACTIVITIES = ["Rest", "Type", "Activity", "Breathe"]
DALIA_ACTIVITIES = ["sitting", "working", "walking", "cycling", "stairs", "driving", "lunch"]
# Activity name -> DaLiA integer code (mirrors the official transient labels).
DALIA_ACT_CODES = {
    "transient": 0,
    "sitting": 1,
    "stairs": 2,
    "soccer": 3,
    "cycling": 4,
    "driving": 5,
    "lunch": 6,
    "walking": 7,
    "working": 8,
}
DALIA_CODE_TO_ACT = {v: k for k, v in DALIA_ACT_CODES.items()}

# Galaxy is generated with motion variation but no explicit activity track.
GALAXY_ACTIVITIES = ["Rest", "Type", "Activity"]

# --------------------------------------------------------------------------
# Per-device error profiles.  Tuned so no single device wins across all
# activities & skin tones (forces real source selection).
#   bias        : constant offset (bpm)
#   motion_sens : how much motion corrupts the reading (0..1)
#   motion_bias : systematic component of the motion error (overestimation)
#   skin_sens   : optical degradation with darker skin (BigIdeas only)
#   lag_sec     : reporting lag (s)
#   dropout     : dropout event rate
#   noise       : baseline gaussian noise (bpm)
# --------------------------------------------------------------------------
BIGIDEAS_PROFILES: dict[str, dict] = {
    "apple_watch": dict(bias=1.0, motion_sens=0.40, motion_bias=1.0, skin_sens=0.5, lag_sec=1, dropout=0.02, noise=1.5),
    "empatica": dict(bias=-1.0, motion_sens=0.60, motion_bias=1.0, skin_sens=1.2, lag_sec=2, dropout=0.05, noise=2.0),
    "garmin": dict(bias=2.0, motion_sens=0.12, motion_bias=0.5, skin_sens=0.6, lag_sec=2, dropout=0.03, noise=1.8),
    "fitbit": dict(bias=0.5, motion_sens=0.90, motion_bias=1.2, skin_sens=0.7, lag_sec=1, dropout=0.03, noise=1.5),
    "miband": dict(bias=3.0, motion_sens=0.70, motion_bias=1.0, skin_sens=0.9, lag_sec=1, dropout=0.04, noise=2.0),
    "biovotion": dict(bias=1.5, motion_sens=0.50, motion_bias=0.8, skin_sens=0.2, lag_sec=2, dropout=0.04, noise=1.8),
}

GALAXY_PROFILES: dict[str, dict] = {
    "galaxy_watch": dict(bias=1.0, motion_sens=0.50, motion_bias=1.0, skin_sens=0.0, lag_sec=1, dropout=0.03, noise=1.5),
    "e4": dict(bias=-1.5, motion_sens=0.85, motion_bias=1.1, skin_sens=0.0, lag_sec=2, dropout=0.05, noise=2.0),
}

DALIA_PROFILE = dict(bias=0.5, motion_sens=0.70, motion_bias=1.0, skin_sens=0.0, lag_sec=1, dropout=0.04, noise=1.8)

# BigIdeas canonical-source -> CSV column name (exact real header).
BIGIDEAS_COLUMNS = {
    "apple_watch": "Apple Watch",
    "empatica": "Empatica",
    "garmin": "Garmin",
    "fitbit": "Fitbit",
    "miband": "Miband",
    "biovotion": "Biovotion",
}

# --------------------------------------------------------------------------
# Generation sizes (kept modest for fast, reproducible runs)
# --------------------------------------------------------------------------
N_BIGIDEAS = 8
BIGIDEAS_DURATION = 600  # seconds (1 row / second)

N_GALAXY = 4
GALAXY_DURATION = 240

N_DALIA = 4
DALIA_DURATION = 180


# --------------------------------------------------------------------------
# Physiological ground-truth model
# --------------------------------------------------------------------------
def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    pad = win // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(win) / win
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad : pad + len(x)]


def build_timeline(
    rng: np.random.Generator,
    duration_sec: int,
    activities: list[str],
    seg_range: tuple[int, int] = (40, 90),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-second (activity, target_hr, target_motion) arrays."""
    acts: list[str] = []
    i = 0
    k = 0
    while i < duration_sec:
        a = activities[k % len(activities)] if k == 0 else str(rng.choice(activities))
        seg = int(rng.integers(seg_range[0], seg_range[1] + 1))
        seg = min(seg, duration_sec - i)
        acts.extend([a] * seg)
        i += seg
        k += 1
    act_arr = np.array(acts[:duration_sec])
    hr_t = np.array([ACTIVITY_PROFILE[a][0] for a in act_arr], dtype=float)
    mo_t = np.array([ACTIVITY_PROFILE[a][1] for a in act_arr], dtype=float)
    return act_arr, hr_t, mo_t


def make_ground_truth(
    rng: np.random.Generator,
    duration_sec: int,
    activities: list[str],
) -> dict:
    """Generate a subject's 1 Hz ground-truth HR, motion, activity, reference."""
    act_arr, hr_t, mo_t = build_timeline(rng, duration_sec, activities)
    baseline = rng.normal(0, 4)
    t = np.arange(duration_sec)
    drift = 2.5 * np.sin(2 * np.pi * t / 300 + rng.uniform(0, 2 * np.pi))
    hr_truth = _smooth(hr_t, 9) + baseline + drift + rng.normal(0, 0.8, duration_sec)
    hr_truth = np.clip(hr_truth, 40, 200)
    motion = np.clip(_smooth(mo_t, 5) + rng.normal(0, 0.03, duration_sec), 0, None)
    reference = hr_truth + rng.normal(0, 0.4, duration_sec)
    return {
        "activity": act_arr,
        "hr_truth": hr_truth,
        "motion": motion,
        "reference": reference,
        "fs": 1.0,
    }


def _apply_dropouts(rng: np.random.Generator, series: np.ndarray, rate: float) -> np.ndarray:
    out = series.copy()
    n = len(series)
    n_events = rng.poisson(rate * n / 10.0)
    for _ in range(int(n_events)):
        start = int(rng.integers(0, n))
        length = int(rng.integers(3, 11))
        out[start : start + length] = np.nan
    return out


def apply_source_error(
    rng: np.random.Generator,
    truth: np.ndarray,
    motion: np.ndarray,
    profile: dict,
    skin_tone: float | None = None,
) -> np.ndarray:
    """Produce a device's reported HR series (NaN where dropped out)."""
    n = len(truth)
    rep = truth.astype(float).copy()

    lag = int(round(profile["lag_sec"]))
    if lag > 0:
        rep = np.concatenate([np.full(lag, rep[0]), rep[:-lag]])

    rep = rep + profile["bias"]

    motion_amp = profile["motion_sens"] * motion
    rep = rep + motion_amp * (15.0 * profile["motion_bias"] + rng.normal(0, 8.0, n))

    if skin_tone is not None and profile["skin_sens"] > 0:
        skin_factor = (skin_tone - 1.0) / 5.0 * profile["skin_sens"]
        rep = rep + skin_factor * (8.0 + rng.normal(0, 5.0, n)) * (0.3 + motion)

    rep = rep + rng.normal(0, profile["noise"], n)
    rep = np.clip(rep, 35, 220)
    rep = _apply_dropouts(rng, rep, profile["dropout"])
    return rep


def _resample_1hz(series_1hz: np.ndarray, out_fs: float, n_out: int) -> np.ndarray:
    """Resample a 1 Hz series to ``out_fs`` (nearest-hold style via interp)."""
    src_t = np.arange(len(series_1hz))
    dst_t = np.arange(n_out) / out_fs
    return np.interp(dst_t, src_t, series_1hz)


# --------------------------------------------------------------------------
# BigIdeasLab_STEP writer (single CSV, HR-only)
# --------------------------------------------------------------------------
def gen_bigideas(out_dir: Path, rng: np.random.Generator) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    subjects = []
    for s in range(N_BIGIDEAS):
        sid = f"BIG{s + 1:02d}"
        skin_tone = int((s % 6) + 1)  # spread skin tones 1..6
        gt = make_ground_truth(rng, BIGIDEAS_DURATION, BIGIDEAS_ACTIVITIES)
        reported = {
            src: apply_source_error(rng, gt["hr_truth"], gt["motion"], prof, skin_tone)
            for src, prof in BIGIDEAS_PROFILES.items()
        }
        for i in range(BIGIDEAS_DURATION):
            row = {"ECG": round(float(gt["reference"][i]), 1)}
            for src, col in BIGIDEAS_COLUMNS.items():
                v = reported[src][i]
                row[col] = "" if np.isnan(v) else round(float(v), 1)
            row["ID"] = sid
            row["Skin Tone"] = skin_tone
            row["Activity"] = str(gt["activity"][i])
            rows.append(row)
        subjects.append({"id": sid, "skin_tone": skin_tone, "rows": BIGIDEAS_DURATION})

    cols = ["ECG", "Apple Watch", "Empatica", "Garmin", "Fitbit", "Miband", "Biovotion", "ID", "Skin Tone", "Activity"]
    df = pd.DataFrame(rows)[cols]
    csv_path = out_dir / "BigIdeasLab_STEP.csv"
    df.to_csv(csv_path, index=False)
    return {
        "format": "BigIdeasLab_STEP single CSV",
        "reference": "ECG",
        "file": str(csv_path.relative_to(out_dir.parent)),
        "n_subjects": N_BIGIDEAS,
        "n_rows": len(df),
        "row_hz": config.BIGIDEAS_ROW_HZ,
        "sources": config.DATASET_SOURCES["bigideas"],
        "subjects": subjects,
    }


# --------------------------------------------------------------------------
# GalaxyPPG writer (dir tree of per-device signal CSVs + metadata)
# --------------------------------------------------------------------------
def _write_signal_csv(path: Path, fs: float, n: int, columns: dict[str, np.ndarray]) -> None:
    ts = np.arange(n) / fs
    data = {"timestamp": ts}
    data.update(columns)
    pd.DataFrame(data).to_csv(path, index=False)


def gen_galaxyppg(out_dir: Path, rng: np.random.Generator) -> dict:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    subjects = []
    for s in range(N_GALAXY):
        pid = f"P{s + 1:02d}"
        pdir = data_dir / pid
        gt = make_ground_truth(rng, GALAXY_DURATION, GALAXY_ACTIVITIES)
        dur = GALAXY_DURATION

        gw_hr = apply_source_error(rng, gt["hr_truth"], gt["motion"], GALAXY_PROFILES["galaxy_watch"])
        e4_hr = apply_source_error(rng, gt["hr_truth"], gt["motion"], GALAXY_PROFILES["e4"])

        # GalaxyWatch
        gw = pdir / "GalaxyWatch"
        gw.mkdir(parents=True, exist_ok=True)
        n_ppg = int(dur * config.FS["galaxy_ppg"])
        ppg = wf.synth_ppg(gt["hr_truth"], 1.0, config.FS["galaxy_ppg"], rng, gt["motion"], 1.0, motion_gain=0.9)
        _write_signal_csv(gw / "PPG.csv", config.FS["galaxy_ppg"], n_ppg, {"ppg": ppg[:n_ppg]})
        _write_signal_csv(gw / "HR.csv", config.FS["galaxy_hr"], dur, {"hr": gw_hr})
        n_acc = int(dur * config.FS["galaxy_acc"])
        acc = wf.synth_acc(gt["motion"], 1.0, config.FS["galaxy_acc"], rng)
        _write_signal_csv(gw / "ACC.csv", config.FS["galaxy_acc"], n_acc, {"x": acc[:n_acc, 0], "y": acc[:n_acc, 1], "z": acc[:n_acc, 2]})

        # E4
        e4 = pdir / "E4"
        e4.mkdir(parents=True, exist_ok=True)
        n_bvp = int(dur * config.FS["e4_bvp"])
        bvp = wf.synth_ppg(gt["hr_truth"], 1.0, config.FS["e4_bvp"], rng, gt["motion"], 1.0, motion_gain=1.2)
        _write_signal_csv(e4 / "BVP.csv", config.FS["e4_bvp"], n_bvp, {"bvp": bvp[:n_bvp]})
        _write_signal_csv(e4 / "HR.csv", config.FS["e4_hr"], dur, {"hr": e4_hr})
        n_acc_e4 = int(dur * config.FS["e4_acc"])
        acc_e4 = wf.synth_acc(gt["motion"], 1.0, config.FS["e4_acc"], rng)
        _write_signal_csv(e4 / "ACC.csv", config.FS["e4_acc"], n_acc_e4, {"x": acc_e4[:n_acc_e4, 0], "y": acc_e4[:n_acc_e4, 1], "z": acc_e4[:n_acc_e4, 2]})
        n_temp = int(dur * config.FS["e4_temp"])
        _write_signal_csv(e4 / "TEMP.csv", config.FS["e4_temp"], n_temp, {"temp": 33.0 + rng.normal(0, 0.2, n_temp)})
        n_eda = int(dur * config.FS["e4_eda"])
        _write_signal_csv(e4 / "EDA.csv", config.FS["e4_eda"], n_eda, {"eda": np.clip(0.5 + _resample_1hz(gt["motion"], config.FS["e4_eda"], n_eda) * 0.3 + rng.normal(0, 0.05, n_eda), 0, None)})

        # PolarH10 (reference)
        polar = pdir / "PolarH10"
        polar.mkdir(parents=True, exist_ok=True)
        n_ecg = int(dur * config.FS["polar_ecg"])
        ecg = wf.synth_ecg(gt["hr_truth"], 1.0, config.FS["polar_ecg"], rng)
        _write_signal_csv(polar / "ECG.csv", config.FS["polar_ecg"], n_ecg, {"ecg": ecg[:n_ecg]})
        _write_signal_csv(polar / "HR.csv", config.FS["polar_hr"], dur, {"hr": np.round(gt["reference"], 1)})
        n_acc_p = int(dur * config.FS["polar_acc"])
        acc_p = wf.synth_acc(gt["motion"], 1.0, config.FS["polar_acc"], rng)
        _write_signal_csv(polar / "ACC.csv", config.FS["polar_acc"], n_acc_p, {"x": acc_p[:n_acc_p, 0], "y": acc_p[:n_acc_p, 1], "z": acc_p[:n_acc_p, 2]})

        meta_rows.append({"ParticipantID": pid, "DurationSec": dur, "AgeGroup": str(rng.choice(["20s", "30s", "40s"])), "Sex": str(rng.choice(["M", "F"]))})
        subjects.append({"id": pid, "duration_sec": dur})

    pd.DataFrame(meta_rows).to_csv(out_dir / "metadata.csv", index=False)
    return {
        "format": "GalaxyPPG dir tree (data/Pxx/{E4,GalaxyWatch,PolarH10})",
        "reference": "PolarH10",
        "metadata": "metadata.csv",
        "n_subjects": N_GALAXY,
        "sources": config.DATASET_SOURCES["galaxyppg"],
        "subjects": subjects,
    }


# --------------------------------------------------------------------------
# PPG-DaLiA writer (per-subject pickle)
# --------------------------------------------------------------------------
def _dalia_labels(reference_1hz: np.ndarray) -> np.ndarray:
    """Reference HR per 8 s window / 2 s shift (DaLiA's native cadence)."""
    win = int(config.WINDOW_SEC)
    shift = int(config.SHIFT_SEC)
    labels = []
    i = 0
    while i + win <= len(reference_1hz):
        labels.append(float(np.nanmean(reference_1hz[i : i + win])))
        i += shift
    return np.array(labels)


def gen_ppg_dalia(out_dir: Path, rng: np.random.Generator) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = []
    for s in range(N_DALIA):
        sid = f"S{s + 1}"
        sdir = out_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)
        gt = make_ground_truth(rng, DALIA_DURATION, DALIA_ACTIVITIES)
        dur = DALIA_DURATION

        # chest ECG (reference) 700 Hz + chest ACC
        n_ecg = int(dur * config.FS["dalia_ecg"])
        ecg = wf.synth_ecg(gt["hr_truth"], 1.0, config.FS["dalia_ecg"], rng).reshape(-1, 1)[:n_ecg]
        n_cacc = int(dur * config.FS["dalia_ecg"])  # chest ACC at 700 Hz in real DaLiA
        cacc = wf.synth_acc(gt["motion"], 1.0, config.FS["dalia_ecg"], rng)[:n_cacc]

        # wrist BVP 64 Hz + wrist ACC 32 Hz + EDA/TEMP 4 Hz
        n_bvp = int(dur * config.FS["dalia_bvp"])
        bvp = wf.synth_ppg(gt["hr_truth"], 1.0, config.FS["dalia_bvp"], rng, gt["motion"], 1.0, motion_gain=1.1).reshape(-1, 1)[:n_bvp]
        n_wacc = int(dur * config.FS["dalia_acc"])
        wacc = wf.synth_acc(gt["motion"], 1.0, config.FS["dalia_acc"], rng)[:n_wacc]
        n_eda = int(dur * config.FS["dalia_eda"])
        eda = np.clip(0.5 + _resample_1hz(gt["motion"], config.FS["dalia_eda"], n_eda) * 0.3 + rng.normal(0, 0.05, n_eda), 0, None).reshape(-1, 1)
        n_temp = int(dur * config.FS["dalia_temp"])
        temp = (33.0 + rng.normal(0, 0.2, n_temp)).reshape(-1, 1)

        labels = _dalia_labels(gt["reference"])
        # activity codes per label window (majority activity in the window)
        act_codes = []
        i = 0
        win = int(config.WINDOW_SEC)
        shift = int(config.SHIFT_SEC)
        while i + win <= dur:
            seg = gt["activity"][i : i + win]
            vals, counts = np.unique(seg, return_counts=True)
            name = str(vals[int(np.argmax(counts))])
            act_codes.append(DALIA_ACT_CODES.get(name, 0))
            i += shift
        activity = np.array(act_codes)

        data = {
            "subject": sid,
            "signal": {
                "chest": {"ECG": ecg.astype(np.float32), "ACC": cacc.astype(np.float32)},
                "wrist": {
                    "BVP": bvp.astype(np.float32),
                    "ACC": wacc.astype(np.float32),
                    "EDA": eda.astype(np.float32),
                    "TEMP": temp.astype(np.float32),
                },
            },
            "label": labels.astype(np.float32),
            "activity": activity.reshape(-1, 1),
            "rpeaks": np.array([], dtype=np.int64),
            "questionnaire": {"AGE": int(rng.integers(20, 50)), "GENDER": str(rng.choice([" m", " f"]))},
        }
        with open(sdir / f"{sid}.pkl", "wb") as f:
            pickle.dump(data, f, protocol=4)
        subjects.append({"id": sid, "duration_sec": dur, "n_labels": int(len(labels))})

    return {
        "format": "PPG-DaLiA per-subject pickle (Sn/Sn.pkl)",
        "reference": "chest ECG",
        "n_subjects": N_DALIA,
        "sources": config.DATASET_SOURCES["ppg_dalia"],
        "subjects": subjects,
    }


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def generate_all(root: Path | None = None, seed: int | None = None) -> dict:
    """Generate all three synthetic datasets + a manifest. Returns the manifest."""
    root = Path(root) if root is not None else config.HR_RAW_DIR
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED if seed is None else seed)

    info = {}
    info["bigideas"] = gen_bigideas(root / "bigideas", rng)
    info["galaxyppg"] = gen_galaxyppg(root / "galaxyppg", rng)
    info["ppg_dalia"] = gen_ppg_dalia(root / "ppg_dalia", rng)

    manifest = {
        "data_type": "synthetic",
        "description": "Synthetic HR source-selection datasets in real on-disk formats (BigIdeasLab_STEP, GalaxyPPG, PPG-DaLiA).",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.SEED if seed is None else seed,
        "window_sec": config.WINDOW_SEC,
        "shift_sec": config.SHIFT_SEC,
        "canonical_sources": config.CANONICAL_SOURCES,
        "datasets": info,
        "failure_modes": [
            "device bias (constant per-device offset)",
            "motion-correlated error (worse during Activity)",
            "reporting lag",
            "dropouts (NaN segments)",
            "skin-tone optical degradation (BigIdeas)",
            "raw PPG motion artifacts corrupting FFT HR estimates",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
