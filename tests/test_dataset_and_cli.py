import json

import numpy as np
import pandas as pd
import pytest

import shortcut_audit as sa
from shortcut_audit.cli import main
from shortcut_audit.synthetic import make_dataset


def test_dataset_validation():
    with pytest.raises(sa.DatasetError):
        sa.AuditDataset(y=[0, 0, 0, 0, 0])  # one class
    with pytest.raises(sa.DatasetError):
        sa.AuditDataset(y=[0, 1, 0, 1], features=np.zeros((3, 2)))
    with pytest.raises(sa.DatasetError):
        sa.AuditDataset(y=[0, 1, 0, 1], signals=np.zeros((4, 1, 10)))  # no fs
    with pytest.raises(sa.DatasetError):
        sa.AuditDataset(y=[0, 1, 0, 1], meta={"s": [1, 2, 3, 4]}, groups=["session"])


def test_from_dataframe_picks_numeric_features():
    df = pd.DataFrame({
        "label": ["a", "b"] * 10,
        "session": ["s1"] * 10 + ["s2"] * 10,
        "f1": np.arange(20.0),
        "f2": np.ones(20),
        "note": ["x"] * 20,
    })
    ds = sa.from_dataframe(df, label="label", groups="session")
    assert ds.feature_names == ["f1", "f2"]
    assert ds.groups == ["session"]


def test_unknown_option_is_rejected():
    ds = make_dataset("clean")
    with pytest.raises(TypeError):
        sa.audit(ds, not_an_option=1)
    with pytest.raises(ValueError):
        sa.audit(ds, only=["nope"])


@pytest.fixture()
def npz_path(tmp_path):
    ds = make_dataset("blocked_drift")
    path = tmp_path / "boxes.npz"
    np.savez(path, signals=ds.signals, y=ds.y, fs=ds.fs, **ds.meta)
    return path


def test_cli_run_npz(npz_path, tmp_path, capsys):
    html, js = tmp_path / "r.html", tmp_path / "r.json"
    code = main(["run", str(npz_path), "--label", "y", "--groups", "session,instance", "--nested", "instance",
                 "--order", "order", "--unit", "trial", "--baseline-s", "0.2", "--html", str(html),
                 "--json", str(js)])
    assert code == 1  # FAIL verdict -> exit 1
    data = json.loads(js.read_text())
    assert data["verdict"] == "fail"
    assert html.read_text().startswith("<!doctype html>")
    assert "VERDICT" in capsys.readouterr().out


def test_cli_fail_on_never_and_errors(npz_path):
    assert main(["run", str(npz_path), "--label", "y", "--order", "order", "--fail-on", "never", "--quiet",
                 "--only", "confound"]) == 0
    assert main(["run", str(npz_path), "--label", "missing"]) == 2


def test_cli_run_csv(tmp_path):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"y": rng.permutation(np.repeat([0, 1], 30)), "f": rng.normal(size=60), "order": np.arange(60)})
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    assert main(["run", str(path), "--label", "y", "--order", "order", "--quiet"]) == 0


def test_cli_demo(capsys):
    assert main(["demo", "--scenario", "instance_memorization", "--quiet"]) == 0
    assert "FAIL" in capsys.readouterr().out
