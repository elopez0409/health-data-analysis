"""Synthesize physiologically-plausible raw waveforms.

These produce *actual* time-domain signals (not just HR numbers) so the
raw-signal datasets (GalaxyPPG, PPG-DaLiA) exercise the signal-quality feature
extractor and the deep track exactly as real data would:

- ``synth_ecg``  : PQRST template placed at the instantaneous HR (reference).
- ``synth_ppg``  : systolic+dicrotic pulse at instantaneous HR, corrupted by
                   motion artifacts whose strength tracks the ACC motion level.
- ``synth_acc``  : quasi-periodic 3-axis accelerometer motion driven by the
                   per-second motion level.
"""

from __future__ import annotations

import numpy as np


def _instantaneous_hr(hr_series: np.ndarray, hr_fs: float, n: int, fs: float) -> np.ndarray:
    """Linearly interpolate an HR series (bpm) onto ``n`` samples at ``fs``."""
    hr_series = np.asarray(hr_series, dtype=float)
    hr_series = np.where(np.isnan(hr_series), np.nanmean(hr_series), hr_series)
    src_t = np.arange(hr_series.shape[0]) / hr_fs
    dst_t = np.arange(n) / fs
    if src_t.shape[0] == 1:
        return np.full(n, hr_series[0])
    return np.interp(dst_t, src_t, hr_series)


def _cardiac_phase(hr_inst_bpm: np.ndarray, fs: float) -> np.ndarray:
    """Accumulate cardiac phase (fraction in [0,1)) from instantaneous HR."""
    freq = hr_inst_bpm / 60.0  # Hz
    dphi = freq / fs
    phase = np.cumsum(dphi)
    return np.mod(phase, 1.0)


# McSharry-style PQRST feature params: (amplitude, angular position theta, width)
_ECG_FEATURES = [
    (0.12, 2 * np.pi * 0.20, 0.25),  # P
    (-0.18, 2 * np.pi * 0.38, 0.10),  # Q
    (1.00, 2 * np.pi * 0.42, 0.08),  # R
    (-0.30, 2 * np.pi * 0.46, 0.10),  # S
    (0.30, 2 * np.pi * 0.70, 0.40),  # T
]


def synth_ecg(
    hr_series: np.ndarray,
    hr_fs: float,
    fs: float,
    rng: np.random.Generator,
    noise: float = 0.02,
) -> np.ndarray:
    """Synthesize an ECG waveform (reference signal)."""
    n = int(round(hr_series.shape[0] / hr_fs * fs))
    hr_inst = _instantaneous_hr(hr_series, hr_fs, n, fs)
    frac = _cardiac_phase(hr_inst, fs)
    theta = 2 * np.pi * frac
    sig = np.zeros(n)
    for amp, th, width in _ECG_FEATURES:
        dtheta = np.mod(theta - th + np.pi, 2 * np.pi) - np.pi
        sig += amp * np.exp(-(dtheta**2) / (2 * width**2))
    # baseline wander + measurement noise
    t = np.arange(n) / fs
    sig += 0.05 * np.sin(2 * np.pi * 0.25 * t)
    sig += rng.normal(0, noise, n)
    return sig.astype(np.float32)


def _ppg_pulse(theta: np.ndarray) -> np.ndarray:
    """Systolic + dicrotic PPG pulse as a function of cardiac angle."""
    systolic = np.exp(-((theta - 2 * np.pi * 0.25) ** 2) / (2 * 0.55**2))
    dicrotic = 0.45 * np.exp(-((theta - 2 * np.pi * 0.55) ** 2) / (2 * 0.8**2))
    return systolic + dicrotic


def synth_ppg(
    hr_series: np.ndarray,
    hr_fs: float,
    fs: float,
    rng: np.random.Generator,
    motion_level: np.ndarray,
    motion_fs: float,
    motion_gain: float = 1.0,
    noise: float = 0.03,
) -> np.ndarray:
    """Synthesize a PPG/BVP waveform corrupted by motion artifacts.

    The cardiac component sits at the true instantaneous HR; motion artifacts
    are quasi-periodic components near the locomotion frequency whose amplitude
    scales with ``motion_level`` (so heavy motion biases FFT-based HR estimates).
    """
    n = int(round(hr_series.shape[0] / hr_fs * fs))
    hr_inst = _instantaneous_hr(hr_series, hr_fs, n, fs)
    frac = _cardiac_phase(hr_inst, fs)
    theta = 2 * np.pi * frac
    sig = _ppg_pulse(theta)

    motion = _instantaneous_hr(motion_level, motion_fs, n, fs)  # reuse interpolator
    t = np.arange(n) / fs
    # Locomotion artifact: frequency wanders around ~2 Hz (walking/running).
    walk_f = 1.8 + 0.6 * (motion / (np.max(motion) + 1e-9))
    walk_phase = 2 * np.pi * np.cumsum(walk_f) / fs
    artifact = motion_gain * motion * (
        np.sin(walk_phase) + 0.4 * np.sin(2 * walk_phase)
    )
    sig = sig + artifact
    # perfusion/baseline wander + sensor noise
    sig += 0.1 * np.sin(2 * np.pi * 0.2 * t)
    sig += rng.normal(0, noise * (1 + motion), n)
    return sig.astype(np.float32)


def synth_acc(
    motion_level: np.ndarray,
    motion_fs: float,
    fs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize a 3-axis accelerometer signal (g units) from motion level."""
    n = int(round(motion_level.shape[0] / motion_fs * fs))
    motion = _instantaneous_hr(motion_level, motion_fs, n, fs)
    t = np.arange(n) / fs
    walk_f = 1.8 + 0.6 * (motion / (np.max(motion) + 1e-9))
    walk_phase = 2 * np.pi * np.cumsum(walk_f) / fs
    acc = np.zeros((n, 3), dtype=np.float32)
    # gravity baseline on z + motion-modulated oscillation per axis
    base = np.array([0.0, 0.0, 1.0])
    for ax in range(3):
        phase_off = ax * 2 * np.pi / 3
        acc[:, ax] = (
            base[ax]
            + motion * 0.6 * np.sin(walk_phase + phase_off)
            + rng.normal(0, 0.02 + 0.05 * motion, n)
        )
    return acc
