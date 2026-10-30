"""Synthetic shaking-box datasets with *planted* shortcuts.

A robot shakes a closed box; four accelerometers record the response; the task is to
tell what is inside (empty, rice, coins, water). Each scenario plants one classic
failure so you can see what the audit reports when you know the ground truth:

``clean``
    Counterbalanced design, randomized schedule, calibrated sensors. Should pass.
``blocked_drift``
    Two classes recorded on each of two days, classes in runs, sensor offset drifts
    from day to day and over time. The baseline predicts the label.
``instance_memorization``
    Each physical box has a strong, unique resonance; the content signal is weak.
    Random splits look great because the model recognizes boxes, not contents.
``window_leakage``
    Long trials are cut into overlapping windows and the windows are split at random,
    so near-copies of test windows sit in the training set.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .dataset import AuditDataset

CLASSES = ["empty", "rice", "coins", "water"]
CLASS_FREQS = [35.0, 70.0, 140.0, 280.0]  # one octave apart

SCENARIOS: Dict[str, Dict] = {
    "clean": {
        "description": "Counterbalanced, randomized, calibrated. Expected: no findings.",
        "options": {"nested": ["instance"], "baseline_s": 0.2},
    },
    "blocked_drift": {
        "description": "Classes recorded in runs on different days while the sensor offset drifts.",
        "options": {"nested": ["instance"], "baseline_s": 0.2},
    },
    "instance_memorization": {
        "description": "Each box has a unique resonance; the content signal is weak.",
        "options": {"nested": ["instance"], "baseline_s": 0.2, "cross": ("session", "instance")},
    },
    "window_leakage": {
        "description": "Overlapping windows from the same trial on both sides of a random split.",
        "options": {"nested": ["instance"]},
    },
}


def _narrowband(rng: np.random.Generator, n_ch: int, n: int, fs: float, fc: float, bw: float) -> np.ndarray:
    w = rng.normal(size=(n_ch, n))
    spec = np.fft.rfft(w, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    x = np.fft.irfft(spec * np.exp(-0.5 * ((f - fc) / bw) ** 2), n=n, axis=-1)
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)


def make_dataset(
    scenario: str = "clean",
    seed: int = 0,
    n_sessions: int = 4,
    n_instances: int = 3,
    trials_per_cell: int = 3,
    fs: float = 1000.0,
    n_channels: int = 4,
) -> AuditDataset:
    """Build a synthetic dataset for ``scenario`` (see module docstring)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    n_classes = len(CLASSES)
    rest_s = 0.2
    shake_hz = 5.0
    trial_s = 1.0
    windows_per_trial, window_s, hop_s = 1, 1.0, 1.0
    a_cls, a_inst, a_trial = 0.45, 0.25, 0.10
    noise = 0.6
    session_sd, trend_per_trial = 0.002, 0.0
    if scenario == "blocked_drift":
        session_sd, trend_per_trial = 0.06, 0.0008
        a_cls = 0.15
    elif scenario == "instance_memorization":
        a_cls, a_inst = 0.10, 1.2
    elif scenario == "window_leakage":
        rest_s = 0.0
        trial_s, window_s, hop_s, windows_per_trial = 3.0, 1.0, 0.5, 5
        a_cls, a_trial = 0.35, 1.0

    chan_gain = rng.uniform(0.6, 1.4, size=n_channels)
    base_offset = rng.normal(0, 0.5, size=n_channels)
    # Instances: per-box resonance and channel coupling.
    inst_freq = {}
    inst_gain = {}
    for c in range(n_classes):
        for i in range(n_instances):
            key = (c, i)
            if scenario == "instance_memorization":
                inst_freq[key] = float(np.exp(rng.uniform(np.log(25), np.log(400))))
                inst_gain[key] = np.exp(rng.normal(0, 0.7, size=n_channels))
            else:
                inst_freq[key] = CLASS_FREQS[c] * float(np.exp(rng.normal(0, 0.06)))
                inst_gain[key] = np.exp(rng.normal(0, 0.1, size=n_channels))

    # Recording schedule.
    cells = []
    for s in range(n_sessions):
        if scenario == "blocked_drift":
            # Day A: empty + rice, day B: coins + water (two sessions per day).
            session_classes = [0, 1] if s < n_sessions // 2 else [2, 3]
            reps = trials_per_cell * 2
        else:
            session_classes = list(range(n_classes))
            reps = trials_per_cell
        block = [(s, c, i) for c in session_classes for i in range(n_instances) for _ in range(reps)]
        if scenario == "blocked_drift":
            block.sort(key=lambda t: t[1])  # classes in runs
        else:
            block = [block[j] for j in rng.permutation(len(block))]
        cells.extend(block)
    session_offsets = rng.normal(0, session_sd, size=(n_sessions, n_channels))

    n_trial = int(round(trial_s * fs))
    n_win = int(round(window_s * fs))
    hop = int(round(hop_s * fs))
    t = np.arange(n_trial) / fs
    shaking = (t >= rest_s).astype(float)
    env = np.abs(np.sin(2 * np.pi * shake_hz * (t - rest_s))) * shaking
    shake = 0.8 * np.sin(2 * np.pi * shake_hz * (t - rest_s)) * shaking

    signals, y, session, instance, trial, order = [], [], [], [], [], []
    for k, (s, c, i) in enumerate(cells):
        trial_freq = float(np.exp(rng.uniform(np.log(25), np.log(400))))
        trial_gain = np.exp(rng.normal(0, 0.5, size=n_channels))
        x = (
            base_offset[:, None]
            + session_offsets[s][:, None]
            + trend_per_trial * k
            + chan_gain[:, None] * shake[None, :]
            + a_cls * chan_gain[:, None] * _narrowband(rng, n_channels, n_trial, fs, CLASS_FREQS[c], 6.0) * env
            + a_inst * inst_gain[(c, i)][:, None] * _narrowband(rng, n_channels, n_trial, fs, inst_freq[(c, i)], 5.0) * env
            + a_trial * trial_gain[:, None] * _narrowband(rng, n_channels, n_trial, fs, trial_freq, 5.0) * env
            + noise * rng.normal(size=(n_channels, n_trial))
        )
        for w in range(windows_per_trial):
            seg = x[:, w * hop: w * hop + n_win]
            signals.append(seg)
            y.append(CLASSES[c])
            session.append(f"S{s + 1}")
            instance.append(f"{CLASSES[c]}#{i + 1}")
            trial.append(f"T{k:04d}")
            order.append(k)
    signals = np.stack(signals)
    meta = {
        "session": np.array(session),
        "instance": np.array(instance),
        "trial": np.array(trial),
        "order": np.array(order),
    }
    split = None
    if scenario == "window_leakage":
        split = np.where(rng.random(len(y)) < 0.8, "train", "test")
    return AuditDataset(
        y=np.array(y),
        signals=signals,
        fs=fs,
        channel_names=[f"acc{j}" for j in range(n_channels)],
        meta=meta,
        groups=["session", "instance"],
        order="order",
        unit="trial",
        split=split,
        name=f"synthetic:{scenario}",
    )


def scenario_options(scenario: str) -> Dict:
    """Recommended :func:`shortcut_audit.audit` keyword arguments for a scenario."""
    return dict(SCENARIOS[scenario]["options"])


def make_benchmark(scenario: str, seed: int = 0) -> Tuple[AuditDataset, Dict]:
    return make_dataset(scenario, seed=seed), scenario_options(scenario)
