"""End-to-end: every planted shortcut is caught, and the clean dataset passes."""

import pytest

import shortcut_audit as sa
from shortcut_audit.synthetic import make_benchmark


def verdicts(report):
    return {(f.audit, f.check): f.verdict.value for f in report.findings}


@pytest.fixture(scope="module")
def reports():
    out = {}
    for name in ["clean", "blocked_drift", "instance_memorization", "window_leakage"]:
        ds, opts = make_benchmark(name, seed=0)
        out[name] = sa.audit(ds, **opts)
    return out


def test_clean_passes(reports):
    r = reports["clean"]
    assert r.verdict == sa.Verdict.PASS, r.to_text()


def test_blocked_drift_is_caught(reports):
    v = verdicts(reports["blocked_drift"])
    assert v[("confound", "session")] == "fail"
    assert v[("confound", "recording order")] == "warn"
    assert v[("nuisance", "order")] == "fail"
    assert v[("nuisance", "baseline (first 0.2 s)")] == "fail"
    assert v[("drift", "across session")] == "fail"
    assert v[("spectral", "offset only")] == "warn"
    assert reports["blocked_drift"].verdict == sa.Verdict.FAIL


def test_instance_memorization_is_caught(reports):
    r = reports["instance_memorization"]
    v = verdicts(r)
    assert v[("protocol", "leave-instance-out")] == "fail"
    assert v[("protocol", "leave-session-and-instance-out")] == "fail"
    # The design itself is fine: nothing else should fire.
    assert all(f.verdict.value in ("pass", "skip") for f in r.findings if f.audit != "protocol")


def test_window_leakage_is_caught(reports):
    v = verdicts(reports["window_leakage"])
    assert v[("leakage", "trial overlap")] == "fail"
    assert v[("protocol", "leave-trial-out")] in ("warn", "fail")
    # Windows of one trial are adjacent in time; blockiness must be computed per trial.
    assert v[("confound", "recording order")] == "pass"


def test_reports_render(reports, tmp_path):
    for name, r in reports.items():
        html = r.to_html(str(tmp_path / f"{name}.html"))
        assert "<svg" in html and r.verdict.value.upper() in html
        js = r.to_json()
        assert '"tool": "shortcut-audit"' in js
        assert r.to_text()
