"""The dataset container that every audit reads from, plus loaders for common formats."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class DatasetError(ValueError):
    """Raised when the input cannot be audited (missing columns, wrong shapes, ...)."""


@dataclass
class AuditDataset:
    """Samples to audit.

    A *sample* is whatever your model classifies: a sensor window, a trial, an episode.

    Parameters
    ----------
    y:
        Label per sample, shape ``(N,)``. Any hashable values.
    features:
        Optional precomputed features, shape ``(N, F)``.
    signals:
        Optional raw time series, shape ``(N, C, T)`` (channels x time). Requires ``fs``.
    fs:
        Sampling rate of ``signals`` in Hz.
    meta:
        Per-sample metadata columns, each shape ``(N,)``: session, object instance,
        subject, recording order, duration, operator, ...
    groups:
        Names of ``meta`` columns that are *grouping factors* you want to generalize
        across (session, instance, subject, day, ...).
    order:
        Name of the ``meta`` column holding recording order or a timestamp.
    unit:
        Name of the ``meta`` column identifying the recording a sample was cut from
        (trial / episode id). Samples sharing a unit are not independent.
    nuisance:
        Extra ``meta`` columns that should carry no label information (gain setting,
        temperature, file size, ...). Each one is probed on its own.
    split:
        Optional ``(N,)`` array of split names (``"train"``/``"test"``/...) for the
        leakage audit.
    """

    y: np.ndarray
    features: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    signals: Optional[np.ndarray] = None
    fs: Optional[float] = None
    channel_names: Optional[List[str]] = None
    meta: Dict[str, np.ndarray] = field(default_factory=dict)
    groups: List[str] = field(default_factory=list)
    order: Optional[str] = None
    unit: Optional[str] = None
    nuisance: List[str] = field(default_factory=list)
    split: Optional[np.ndarray] = None
    name: str = "dataset"
    # Extra nuisance feature blocks computed by adapters (e.g. LeRobot initial state).
    nuisance_blocks: Dict[str, np.ndarray] = field(default_factory=dict)
    # Human-readable explanation per nuisance probe, shown when it fires.
    nuisance_advice: Dict[str, str] = field(default_factory=dict)
    # Optional per-sample sequences for the open-loop replay probe (LeRobot actions).
    sequences: Optional[List[np.ndarray]] = None
    # Optional per-sample resting level of each sensor channel, shape (N, C), for the
    # drift audit when there are no fixed-length signal windows (e.g. LeRobot episodes).
    levels: Optional[np.ndarray] = None
    level_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y)
        if self.y.ndim != 1:
            raise DatasetError(f"y must be 1-D, got shape {self.y.shape}")
        n = len(self.y)
        if n < 4:
            raise DatasetError("need at least 4 samples")
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=float)
            if self.features.ndim != 2 or self.features.shape[0] != n:
                raise DatasetError(f"features must be (N, F) with N={n}, got {self.features.shape}")
            if self.feature_names is None:
                self.feature_names = [f"f{i}" for i in range(self.features.shape[1])]
        if self.signals is not None:
            self.signals = np.asarray(self.signals, dtype=float)
            if self.signals.ndim == 2:
                self.signals = self.signals[:, None, :]
            if self.signals.ndim != 3 or self.signals.shape[0] != n:
                raise DatasetError(f"signals must be (N, C, T) with N={n}, got {self.signals.shape}")
            if self.fs is None or self.fs <= 0:
                raise DatasetError("signals given but fs (sampling rate, Hz) is missing")
            if self.channel_names is None:
                self.channel_names = [f"ch{i}" for i in range(self.signals.shape[1])]
        meta = {}
        for key, col in self.meta.items():
            col = np.asarray(col)
            if col.shape != (n,):
                raise DatasetError(f"meta column {key!r} must have shape ({n},), got {col.shape}")
            meta[str(key)] = col
        self.meta = meta
        for key in list(self.groups) + list(self.nuisance) + [self.order, self.unit]:
            if key is not None and key not in self.meta:
                raise DatasetError(f"column {key!r} not found in meta (have: {sorted(self.meta)})")
        if self.split is not None:
            self.split = np.asarray(self.split).astype(str)
            if self.split.shape != (n,):
                raise DatasetError("split must have one entry per sample")
        for key, block in list(self.nuisance_blocks.items()):
            block = np.asarray(block, dtype=float)
            if block.ndim == 1:
                block = block[:, None]
            if block.shape[0] != n:
                raise DatasetError(f"nuisance block {key!r} has {block.shape[0]} rows, expected {n}")
            self.nuisance_blocks[key] = block
        if len(np.unique(self.y)) < 2 and self.sequences is None and self.levels is None:
            raise DatasetError("need at least two classes")
        if self.sequences is not None and len(self.sequences) != n:
            raise DatasetError("sequences must have one entry per sample")
        if self.levels is not None:
            self.levels = np.asarray(self.levels, dtype=float)
            if self.levels.ndim == 1:
                self.levels = self.levels[:, None]
            if self.levels.shape[0] != n:
                raise DatasetError("levels must have one row per sample")
            if self.level_names is None:
                self.level_names = [f"level{i}" for i in range(self.levels.shape[1])]

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.y)

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y)

    @property
    def y_codes(self) -> np.ndarray:
        return np.unique(self.y, return_inverse=True)[1]

    @property
    def has_signals(self) -> bool:
        return self.signals is not None

    def summary(self) -> Dict[str, object]:
        classes, counts = np.unique(self.y, return_counts=True)
        out: Dict[str, object] = {
            "name": self.name,
            "n_samples": int(len(self.y)),
            "n_classes": int(len(classes)),
            "class_counts": {str(c): int(k) for c, k in zip(classes, counts)},
            "groups": {g: int(len(np.unique(self.meta[g]))) for g in self.groups},
            "order": self.order,
            "unit": self.unit,
            "nuisance": list(self.nuisance) + sorted(self.nuisance_blocks),
            "has_split": self.split is not None,
        }
        if self.unit:
            out["n_units"] = int(len(np.unique(self.meta[self.unit])))
        if self.features is not None:
            out["n_features"] = int(self.features.shape[1])
        if self.signals is not None:
            out["signals"] = {
                "channels": int(self.signals.shape[1]),
                "samples_per_window": int(self.signals.shape[2]),
                "fs_hz": float(self.fs),
                "window_s": float(self.signals.shape[2] / self.fs),
            }
        return out


# ---------------------------------------------------------------------- loaders


def _split_list(value: Optional[Sequence[str] | str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def from_dataframe(
    df: pd.DataFrame,
    label: str,
    groups: Optional[Sequence[str] | str] = None,
    order: Optional[str] = None,
    unit: Optional[str] = None,
    nuisance: Optional[Sequence[str] | str] = None,
    split: Optional[str] = None,
    features: Optional[Sequence[str] | str] = None,
    name: str = "dataset",
) -> AuditDataset:
    """Build an :class:`AuditDataset` from a table (one row per sample).

    Every numeric column that is not the label, a grouping/meta column or the split
    column is treated as a feature unless ``features`` lists them explicitly.
    """
    groups_l = _split_list(groups)
    nuis_l = _split_list(nuisance)
    if label not in df.columns:
        raise DatasetError(f"label column {label!r} not found (columns: {list(df.columns)[:20]}...)")
    meta_cols = [c for c in groups_l + nuis_l + [order, unit] if c]
    for c in meta_cols + ([split] if split else []):
        if c not in df.columns:
            raise DatasetError(f"column {c!r} not found")
    if features is not None:
        feat_cols = _split_list(features)
    else:
        exclude = set(meta_cols) | {label} | ({split} if split else set())
        feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].to_numpy(dtype=float) if feat_cols else None
    if X is not None and np.isnan(X).any():
        raise DatasetError("feature columns contain NaN; impute or drop them before auditing")
    meta = {c: df[c].to_numpy() for c in dict.fromkeys(meta_cols)}
    return AuditDataset(
        y=df[label].to_numpy(),
        features=X,
        feature_names=feat_cols or None,
        meta=meta,
        groups=groups_l,
        order=order,
        unit=unit,
        nuisance=nuis_l,
        split=df[split].to_numpy() if split else None,
        name=name,
    )


def from_npz(
    path: str,
    label: str = "y",
    groups: Optional[Sequence[str] | str] = None,
    order: Optional[str] = None,
    unit: Optional[str] = None,
    nuisance: Optional[Sequence[str] | str] = None,
    split: Optional[str] = None,
    fs: Optional[float] = None,
    name: Optional[str] = None,
) -> AuditDataset:
    """Load an ``.npz`` with ``signals`` (N, C, T) and/or ``features`` (N, F), labels, and
    any number of 1-D metadata arrays of length N."""
    data = np.load(path, allow_pickle=False)
    if label not in data:
        raise DatasetError(f"{path}: no array named {label!r} (have: {sorted(data.files)})")
    y = data[label]
    n = len(y)
    signals = data["signals"] if "signals" in data else (data["X"] if "X" in data and data["X"].ndim == 3 else None)
    features = data["features"] if "features" in data else (data["X"] if "X" in data and data["X"].ndim == 2 else None)
    if signals is None and features is None:
        raise DatasetError(f"{path}: need an array named 'signals' (N,C,T) or 'features' (N,F)")
    if fs is None and "fs" in data:
        fs = float(np.asarray(data["fs"]).ravel()[0])
    reserved = {label, "signals", "features", "X", "fs", "feature_names", "channel_names"}
    meta = {k: data[k] for k in data.files if k not in reserved and data[k].shape == (n,)}
    feature_names = [str(s) for s in data["feature_names"]] if "feature_names" in data else None
    channel_names = [str(s) for s in data["channel_names"]] if "channel_names" in data else None
    split_arr = meta.pop(split) if split else None
    return AuditDataset(
        y=y,
        features=features,
        feature_names=feature_names,
        signals=signals,
        fs=fs,
        channel_names=channel_names,
        meta=meta,
        groups=_split_list(groups),
        order=order,
        unit=unit,
        nuisance=_split_list(nuisance),
        split=split_arr,
        name=name or os.path.splitext(os.path.basename(path))[0],
    )


def load(path: str, label: str, **kwargs) -> AuditDataset:
    """Load ``.npz``, ``.csv``, ``.tsv`` or ``.parquet`` by extension."""
    ext = os.path.splitext(path)[1].lower()
    name = kwargs.pop("name", None) or os.path.splitext(os.path.basename(path))[0]
    if ext == ".npz":
        kwargs.pop("features", None)
        return from_npz(path, label=label, name=name, **kwargs)
    kwargs.pop("fs", None)
    if ext in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise DatasetError(f"unsupported file type {ext!r}; use .npz, .csv, .tsv or .parquet")
    return from_dataframe(df, label=label, name=name, **kwargs)
