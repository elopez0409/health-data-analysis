"""Signal-quality features from raw PPG/BVP + ACC windows.

For raw-signal datasets these features tell the model how trustworthy each
source is in a given window (independent of the device's reported HR):

- ``q_hr_est``    : FFT peak HR estimate in the cardiac band (bpm)
- ``q_snr``       : spectral SNR = peak power / total in-band power
- ``q_entropy``   : spectral entropy of the PPG (lower = cleaner periodicity)
- ``q_acc_mag``   : RMS accelerometer magnitude (motion energy)
- ``q_perfusion`` : AC/DC amplitude ratio (perfusion index proxy)
- ``q_template``  : autocorrelation periodicity score at the dominant HR

Only numpy is required (no scipy dependency) so the package stays importable
with just the ``ml`` group installed.
"""

from __future__ import annotations

import numpy as np

from hr_selection import config

QUALITY_KEYS = ["q_hr_est", "q_snr", "q_entropy", "q_acc_mag", "q_perfusion", "q_template"]


def _bandpass_detrend(x: np.ndarray) -> np.ndarray:
    """Remove DC + slow drift by subtracting a moving average."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.mean(x)
    return x


def ppg_quality(ppg: np.ndarray, fs: float) -> dict[str, float]:
    """Compute spectral / periodicity quality features from a PPG window."""
    out = {k: float("nan") for k in QUALITY_KEYS if k != "q_acc_mag"}
    ppg = np.asarray(ppg, dtype=float)
    n = ppg.shape[0]
    if n < int(fs * 2):  # need >= ~2 s
        return out

    x = _bandpass_detrend(ppg)
    # Perfusion index proxy on the raw (pre-detrend) signal.
    dc = np.mean(np.abs(ppg)) + 1e-9
    ac = np.percentile(ppg, 95) - np.percentile(ppg, 5)
    out["q_perfusion"] = float(ac / dc)

    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    lo, hi = config.PPG_BAND
    band = (freqs >= lo) & (freqs <= hi)
    if not np.any(band) or spec[band].sum() <= 0:
        return out

    band_spec = spec[band]
    band_freqs = freqs[band]
    peak_idx = int(np.argmax(band_spec))
    peak_freq = band_freqs[peak_idx]
    out["q_hr_est"] = float(peak_freq * 60.0)

    total = band_spec.sum() + 1e-12
    out["q_snr"] = float(band_spec[peak_idx] / total)

    p = band_spec / total
    p = p[p > 0]
    out["q_entropy"] = float(-np.sum(p * np.log(p)) / np.log(len(band_spec) + 1e-9))

    # Autocorrelation periodicity score at the dominant cardiac lag.
    lag = int(round(fs / max(peak_freq, 1e-6)))
    if 0 < lag < n:
        x0 = x - np.mean(x)
        denom = np.sum(x0 * x0) + 1e-12
        out["q_template"] = float(np.sum(x0[:-lag] * x0[lag:]) / denom)
    return out


def acc_motion(acc: np.ndarray, fs: float) -> float:
    """RMS accelerometer magnitude after removing the per-axis mean (gravity)."""
    acc = np.asarray(acc, dtype=float)
    if acc.size == 0:
        return float("nan")
    if acc.ndim == 1:
        acc = acc[:, None]
    centered = acc - np.mean(acc, axis=0, keepdims=True)
    mag = np.sqrt(np.sum(centered**2, axis=1))
    return float(np.sqrt(np.mean(mag**2)))


def window_quality(ppg: np.ndarray, ppg_fs: float, acc: np.ndarray, acc_fs: float) -> dict[str, float]:
    """All quality features for one source's raw window."""
    feats = ppg_quality(ppg, ppg_fs)
    feats["q_acc_mag"] = acc_motion(acc, acc_fs)
    return {k: feats.get(k, float("nan")) for k in QUALITY_KEYS}
