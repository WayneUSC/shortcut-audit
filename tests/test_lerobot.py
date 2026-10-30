import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

import shortcut_audit as sa  # noqa: E402
from shortcut_audit.cli import main  # noqa: E402
from shortcut_audit.lerobot import load_lerobot, write_toy_lerobot  # noqa: E402


def verdicts(report):
    return {(f.audit, f.check): f.verdict.value for f in report.findings}


def test_toy_with_shortcuts(tmp_path):
    root = write_toy_lerobot(str(tmp_path / "toy"), shortcut=True)
    ds = load_lerobot(root)
    assert set(ds.classes) == {"pick up the sponge", "press the button"}
    assert "initial state" in ds.nuisance_blocks
    assert "baseline: observation.tactile" in ds.nuisance_blocks
    v = verdicts(sa.audit(ds, skip=["spectral", "protocol"]))
    assert v[("nuisance", "initial state")] == "fail"
    assert v[("nuisance", "baseline: observation.tactile")] == "fail"
    assert v[("confound", "recording order")] == "warn"
    assert v[("replay", "open-loop replay R²")] == "warn"
    assert v[("drift", "trend over episode_index")] == "warn"


def test_toy_without_shortcuts(tmp_path):
    root = write_toy_lerobot(str(tmp_path / "toy"), shortcut=False, seed=3)
    rep = sa.audit(load_lerobot(root), skip=["spectral", "protocol"])
    assert rep.verdict == sa.Verdict.PASS, rep.to_text()


def test_v21_layout_and_episode_meta(tmp_path):
    """v2.1: one parquet per episode, tasks.jsonl; plus an operator CSV."""
    root = tmp_path / "v21"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    rng = np.random.default_rng(0)
    tasks = ["fold", "stack"]
    for ep in range(16):
        task = ep % 2
        n = 20
        df = pd.DataFrame({
            "observation.state": list(rng.normal(size=(n, 3)).astype(np.float32)),
            "action": list(rng.normal(size=(n, 3)).astype(np.float32)),
            "observation.effort": list((rng.normal(size=(n, 2)) + (2.0 if ep < 8 else 0.0)).astype(np.float32)),
            "timestamp": np.arange(n) / 30.0,
            "frame_index": np.arange(n),
            "episode_index": ep,
            "index": np.arange(n) + ep * n,
            "task_index": task,
        })
        df.to_parquet(root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    with open(root / "meta" / "tasks.jsonl", "w") as fh:
        for i, t in enumerate(tasks):
            fh.write(json.dumps({"task_index": i, "task": t}) + "\n")
    info = {"codebase_version": "v2.1", "fps": 30, "splits": {"train": "0:12", "test": "12:16"},
            "features": {"observation.state": {"dtype": "float32", "shape": [3]},
                         "action": {"dtype": "float32", "shape": [3]},
                         "observation.effort": {"dtype": "float32", "shape": [2]},
                         "observation.images.top": {"dtype": "video", "shape": [3, 64, 64]}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pd.DataFrame({"episode_index": range(16), "operator": ["ana"] * 8 + ["bo"] * 8}).to_csv(tmp_path / "ops.csv", index=False)

    ds = load_lerobot(str(root), episode_meta=str(tmp_path / "ops.csv"), groups=["operator"])
    assert list(ds.classes) == ["fold", "stack"]
    assert ds.split is not None and set(ds.split) == {"train", "test"}
    assert ds.levels.shape == (16, 2)
    rep = sa.audit(ds, skip=["spectral", "protocol"])
    v = verdicts(rep)
    assert v[("confound", "operator")] == "pass"  # tasks alternate, operators balanced
    assert ("drift", "across operator") in v  # effort offset differs by operator
    assert v[("drift", "across operator")] in ("warn", "fail")


def test_cli_lerobot(tmp_path):
    root = write_toy_lerobot(str(tmp_path / "toy"), shortcut=True)
    assert main(["lerobot", root, "--quiet"]) == 1
    assert main(["lerobot", str(tmp_path / "missing_dir")]) == 2
