"""Adapter for LeRobot datasets (v2.1 and v3.0 layouts), local or on the Hugging Face Hub.

Episode-level questions this adapter sets up:

* **Can the task be guessed before the robot moves?** The first frame of
  ``observation.state`` is probed against the task label. If the starting pose gives the
  task away, a policy can read the task from proprioception and ignore the instruction.
* **Do non-visual sensors start from a label-correlated level?** For every other numeric
  observation (tactile, force/torque, effort, current, ...) the mean of the first frames
  is probed, and its drift over the recording campaign is measured.
* **Can a blind replay explain the actions?** Time-normalized mean trajectories of the
  other demos of the same task are replayed against each held-out episode.
* **Were episodes recorded in task blocks, or duplicated?**

Only parquet files are read; videos are never downloaded or decoded.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dataset import AuditDataset, DatasetError

NON_NUMERIC = {"video", "image", "string", "str"}
BOOKKEEPING = {"timestamp", "frame_index", "episode_index", "index", "task_index", "next.done", "next.reward", "next.success"}


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("reading LeRobot datasets needs pyarrow: pip install 'shortcut-audit[lerobot]'") from exc


def resolve_root(path_or_repo: str, revision: Optional[str] = None, cache_dir: Optional[str] = None) -> str:
    """Return a local dataset root. ``hf://user/name`` or ``user/name`` (when no such
    local folder exists) is fetched from the Hub (meta/ and data/ only, no videos)."""
    if path_or_repo.startswith("hf://"):
        repo = path_or_repo[len("hf://"):]
    elif os.path.isdir(path_or_repo):
        return path_or_repo
    elif path_or_repo.count("/") == 1 and not os.path.exists(path_or_repo):
        repo = path_or_repo
    else:
        raise DatasetError(f"{path_or_repo!r} is not a directory")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("downloading from the Hub needs huggingface_hub: pip install 'shortcut-audit[lerobot]'") from exc
    return snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["meta/*", "meta/**/*", "data/*", "data/**/*"],
    )


def _read_info(root: str) -> Dict:
    path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(path):
        raise DatasetError(f"{root}: meta/info.json not found — is this a LeRobot dataset?")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_tasks(root: str) -> Dict[int, str]:
    jsonl = os.path.join(root, "meta", "tasks.jsonl")
    pq = os.path.join(root, "meta", "tasks.parquet")
    tasks: Dict[int, str] = {}
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
        if "task" in df.columns:
            tasks = {int(i): str(t) for i, t in zip(df["task_index"], df["task"])}
        else:  # v3.0 stores the task text as the index
            tasks = {int(i): str(t) for t, i in zip(df.index, df["task_index"])}
    elif os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    tasks[int(rec["task_index"])] = str(rec["task"])
    return tasks


def _numeric_keys(info: Dict, columns: Sequence[str]) -> List[str]:
    keys = []
    for key, spec in info.get("features", {}).items():
        if key in BOOKKEEPING or key not in columns:
            continue
        if str(spec.get("dtype", "")).lower() in NON_NUMERIC:
            continue
        keys.append(key)
    return keys


def _column_to_array(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack([np.asarray(v, dtype=float).ravel() for v in series.to_numpy()])
    return series.to_numpy(dtype=float)[:, None]


def load_lerobot(
    path_or_repo: str,
    target: str = "task",
    state_key: str = "observation.state",
    action_key: str = "action",
    sensor_keys: Optional[Sequence[str]] = None,
    baseline_frames: int = 5,
    episode_meta: Optional[str] = None,
    groups: Optional[Sequence[str]] = None,
    nuisance: Optional[Sequence[str]] = None,
    revision: Optional[str] = None,
) -> AuditDataset:
    """Load a LeRobot dataset as an episode-level :class:`AuditDataset`.

    Parameters
    ----------
    target:
        ``"task"`` (task text, default), ``"task_index"``, or any per-frame column or
        ``episode_meta`` column to use as the label.
    sensor_keys:
        Numeric observation keys to treat as sensors (baseline probes + drift). Default:
        every numeric feature except state, action and bookkeeping columns.
    baseline_frames:
        Number of initial frames averaged for the resting level of each sensor.
    episode_meta:
        Optional CSV with an ``episode_index`` column plus per-episode metadata
        (session, operator, day, ...). Use with ``groups`` / ``nuisance``.
    """
    _require_pyarrow()
    root = resolve_root(path_or_repo, revision=revision)
    info = _read_info(root)
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise DatasetError(f"{root}: no parquet files under data/")
    frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "episode_index" not in frames.columns:
        raise DatasetError("data parquet has no episode_index column")
    sort_cols = ["episode_index"] + (["frame_index"] if "frame_index" in frames.columns else [])
    frames = frames.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    numeric = _numeric_keys(info, frames.columns)
    if sensor_keys is None:
        sensor_keys = [k for k in numeric if k not in (state_key, action_key)]
    tasks = _read_tasks(root)

    epi = frames["episode_index"].to_numpy()
    episodes, starts = np.unique(epi, return_index=True)
    ends = np.r_[starts[1:], len(frames)]

    meta_df = None
    if episode_meta:
        meta_df = pd.read_csv(episode_meta)
        if "episode_index" not in meta_df.columns:
            raise DatasetError(f"{episode_meta}: needs an episode_index column")
        meta_df = meta_df.set_index("episode_index")

    def per_episode_label() -> np.ndarray:
        if target in ("task", "task_index"):
            if "task_index" not in frames.columns:
                raise DatasetError("no task_index column; pass target=<column>")
            ti = frames["task_index"].to_numpy()
            idx = np.array([np.bincount(ti[s:e].astype(int)).argmax() for s, e in zip(starts, ends)])
            if target == "task" and tasks:
                return np.array([tasks.get(int(i), f"task {i}") for i in idx])
            return idx
        if meta_df is not None and target in meta_df.columns:
            return meta_df.loc[episodes, target].to_numpy()
        if target in frames.columns:
            col = frames[target].to_numpy()
            return np.array([pd.Series(col[s:e]).mode().iloc[0] for s, e in zip(starts, ends)])
        raise DatasetError(f"target {target!r} not found in frames or episode_meta")

    y = per_episode_label()
    lengths = ends - starts
    meta: Dict[str, np.ndarray] = {"episode_index": episodes.astype(int), "length": lengths.astype(int)}
    if meta_df is not None:
        missing = set(episodes) - set(meta_df.index)
        if missing:
            raise DatasetError(f"episode_meta lacks {len(missing)} episodes, e.g. {sorted(missing)[:5]}")
        for col in meta_df.columns:
            meta[col] = meta_df.loc[episodes, col].to_numpy()

    nuisance_blocks: Dict[str, np.ndarray] = {}
    advice: Dict[str, str] = {}
    level_cols, level_names = [], []
    if state_key in frames.columns:
        state = _column_to_array(frames[state_key])
        nuisance_blocks["initial state"] = state[starts]
        advice["initial state"] = (
            "The task can be predicted from the robot's starting configuration, before it moves. A policy can "
            "infer the task from proprioception and ignore the language instruction. Unless start poses differ "
            "by design, randomize them across tasks."
        )
    for key in sensor_keys:
        if key not in frames.columns:
            raise DatasetError(f"sensor key {key!r} not in data columns")
        arr = _column_to_array(frames[key])
        base = np.stack([arr[s:min(e, s + baseline_frames)].mean(axis=0) for s, e in zip(starts, ends)])
        name = f"baseline: {key}"
        nuisance_blocks[name] = base
        advice[name] = (
            f"The resting level of '{key}' (first {baseline_frames} frames) predicts the task. Sensor offset "
            f"or drift differs between tasks, so a policy can read the task from the offset. Re-zero the sensor "
            f"per episode and interleave tasks during recording."
        )
        level_cols.append(base)
        level_names.extend(f"{key}[{j}]" for j in range(base.shape[1]))

    sequences = None
    if action_key in frames.columns:
        act = _column_to_array(frames[action_key])
        sequences = [act[s:e] for s, e in zip(starts, ends)]

    split = None
    splits = info.get("splits") or {}
    if len(splits) > 1:
        split = np.full(len(episodes), "unassigned", dtype=object)
        for name, rng in splits.items():
            try:
                a, b = (int(v) for v in str(rng).split(":"))
            except ValueError:
                continue
            split[(episodes >= a) & (episodes < b)] = name
        split = split.astype(str)

    levels = np.hstack(level_cols) if level_cols else None
    return AuditDataset(
        y=np.asarray(y),
        meta=meta,
        groups=list(groups or []),
        order="episode_index",
        nuisance=list(nuisance or []),
        split=split,
        name=os.path.basename(os.path.normpath(path_or_repo)) if os.path.isdir(path_or_repo) else path_or_repo,
        nuisance_blocks=nuisance_blocks,
        nuisance_advice=advice,
        sequences=sequences,
        levels=levels,
        level_names=level_names or None,
    )


def write_toy_lerobot(root: str, n_episodes_per_task: int = 12, seed: int = 0, shortcut: bool = True,
                      fps: int = 30) -> str:
    """Write a tiny LeRobot v3-style dataset (two tasks, state, action, a tactile array).

    With ``shortcut=True`` the operator starts each task from a different pose, records
    tasks in blocks, the tactile sensor drifts during the session, and demos are near
    replays — every LeRobot-specific audit should fire. Used by tests and the docs.
    """
    _require_pyarrow()
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    os.makedirs(os.path.join(root, "data", "chunk-000"), exist_ok=True)
    tasks = ["pick up the sponge", "press the button"]
    schedule = [t for t in range(len(tasks)) for _ in range(n_episodes_per_task)]
    if not shortcut:
        schedule = list(rng.permutation(schedule))
    rows = []
    gidx = 0
    for ep, task in enumerate(schedule):
        length = int(rng.integers(40, 60))
        phase = np.linspace(0, 1, length)
        home = np.array([0.0, -0.5, 0.8, 0.0, 0.3, 0.0])
        start = home + (np.array([0.25, 0.1, -0.1, 0.0, 0.0, 0.0]) * (1 if task else -1) if shortcut else 0) \
            + rng.normal(0, 0.01 if shortcut else 0.15, 6)
        goal = home + np.array([0.4, 0.3, -0.4, 0.2, -0.2, 0.5]) * (1 if task == 0 else -1)
        noise_sd = 0.005 if shortcut else 0.15
        traj = start + (goal - start) * (0.5 - 0.5 * np.cos(np.pi * phase))[:, None] + rng.normal(0, noise_sd, (length, 6))
        drift = 0.04 * ep if shortcut else 0.0
        contact = (phase > 0.5).astype(float)
        tactile = drift + contact[:, None] * rng.uniform(0.5, 1.0, 16) + rng.normal(0, 0.02, (length, 16))
        for f in range(length):
            rows.append({
                "observation.state": traj[f].astype(np.float32),
                "observation.tactile": tactile[f].astype(np.float32),
                "action": (traj[min(f + 1, length - 1)]).astype(np.float32),
                "timestamp": np.float32(f / fps),
                "frame_index": f,
                "episode_index": ep,
                "index": gidx,
                "task_index": task,
            })
            gidx += 1
    pd.DataFrame(rows).to_parquet(os.path.join(root, "data", "chunk-000", "file-000.parquet"))
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(
        os.path.join(root, "meta", "tasks.parquet"))
    info = {
        "codebase_version": "v3.0",
        "robot_type": "toy_arm",
        "fps": fps,
        "total_episodes": len(schedule),
        "total_frames": gidx,
        "total_tasks": len(tasks),
        "splits": {"train": f"0:{len(schedule)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6], "names": [f"j{i}" for i in range(6)]},
            "observation.tactile": {"dtype": "float32", "shape": [16], "names": None},
            "action": {"dtype": "float32", "shape": [6], "names": [f"j{i}" for i in range(6)]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(os.path.join(root, "meta", "info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    return root
