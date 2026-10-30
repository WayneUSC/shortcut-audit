"""Small, transparent feature extractors for raw sensor windows.

The audits deliberately use simple features (per-channel offset, spread and log band
power). They are not meant to be a strong classifier; they are a probe that tells you
*where* label information lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Band = Tuple[float, float]


@dataclass
class FeatureSet:
    X: np.ndarray
    names: List[str]
    blocks: Dict[str, List[int]] = field(default_factory=dict)

    def block(self, key: str) -> np.ndarray:
        return self.X[:, self.blocks[key]]

    def without(self, keys: Sequence[str]) -> np.ndarray:
        drop = {i for k in keys for i in self.blocks.get(k, [])}
        keep = [i for i in range(self.X.shape[1]) if i not in drop]
        return self.X[:, keep]


def _fmt_hz(f: float) -> str:
    if f >= 1000:
        return f"{f / 1000:g}k"
    if f >= 10:
        return f"{f:.0f}"
    return f"{f:.2g}"


def band_label(band: Band) -> str:
    return f"{_fmt_hz(band[0])}–{_fmt_hz(band[1])} Hz"


def default_bands(fs: float, n_samples: int, max_bands: int = 10) -> List[Band]:
    """Octave bands from the lowest resolvable frequency up to Nyquist.

    The lowest band absorbs everything below the first octave edge so that slow drift
    inside a window still has a home.
    """
    nyq = fs / 2.0
    f_res = fs / n_samples
    lo0 = max(f_res, nyq / 2 ** (max_bands - 1))
    edges = [lo0]
    while edges[-1] * 2 < nyq * 0.999:
        edges.append(edges[-1] * 2)
    edges.append(nyq)
    bands: List[Band] = []
    if lo0 > f_res * 1.5:
        bands.append((f_res, lo0))
    bands.extend(zip(edges[:-1], edges[1:]))
    return [(float(a), float(b)) for a, b in bands]


def band_powers(signals: np.ndarray, fs: float, bands: Sequence[Band]) -> Tuple[np.ndarray, List[Band]]:
    """Log10 band power per (sample, channel, band) from a Hann-windowed periodogram.

    The per-window mean is removed first, so DC offset never leaks into band power.
    Returns an array of shape (N, C, B_kept) and the list of bands that had at least
    one frequency bin.
    """
    x = signals - signals.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    out, kept = [], []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi) if hi < fs / 2 else (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            continue
        out.append(np.log10(spec[..., mask].sum(axis=-1) + 1e-20))
        kept.append((lo, hi))
    if not out:
        return np.zeros(signals.shape[:2] + (0,)), []
    return np.stack(out, axis=-1), kept


def signal_features(
    signals: np.ndarray,
    fs: float,
    channel_names: Optional[Sequence[str]] = None,
    bands: Optional[Sequence[Band]] = None,
) -> FeatureSet:
    """Per-channel offset (mean), spread (std) and log band powers.

    Feature blocks:
      ``offset``     per-channel mean (DC level / baseline)
      ``spread``     per-channel standard deviation
      ``band:<lo-hi>`` per-channel log band power
    """
    n, c, t = signals.shape
    channel_names = list(channel_names or [f"ch{i}" for i in range(c)])
    bands = list(bands) if bands is not None else default_bands(fs, t)
    bp, kept = band_powers(signals, fs, bands)
    cols, names = [], []
    blocks: Dict[str, List[int]] = {}

    def add(block: str, values: np.ndarray, suffix: str) -> None:
        start = len(names)
        for j in range(c):
            cols.append(values[:, j])
            names.append(f"{channel_names[j]}:{suffix}")
        blocks.setdefault(block, []).extend(range(start, start + c))

    add("offset", signals.mean(axis=-1), "offset")
    add("spread", signals.std(axis=-1), "std")
    for b, band in enumerate(kept):
        add(f"band:{band_label(band)}", bp[:, :, b], f"logpow[{band_label(band)}]")
    return FeatureSet(X=np.column_stack(cols), names=names, blocks=blocks)


def baseline_features(
    signals: np.ndarray,
    fs: float,
    baseline_s: float,
    channel_names: Optional[Sequence[str]] = None,
) -> FeatureSet:
    """Mean and std of the first ``baseline_s`` seconds of each window (pre-contact level)."""
    n_base = max(2, int(round(baseline_s * fs)))
    n_base = min(n_base, signals.shape[-1])
    seg = signals[..., :n_base]
    c = signals.shape[1]
    channel_names = list(channel_names or [f"ch{i}" for i in range(c)])
    X = np.concatenate([seg.mean(axis=-1), seg.std(axis=-1)], axis=1)
    names = [f"{ch}:baseline_mean" for ch in channel_names] + [f"{ch}:baseline_std" for ch in channel_names]
    return FeatureSet(X=X, names=names, blocks={"baseline_mean": list(range(c)), "baseline_std": list(range(c, 2 * c))})
