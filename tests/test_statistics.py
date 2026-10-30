import numpy as np
import pytest

from shortcut_audit.audits import blockiness, cramers_v, kruskal_epsilon2
from shortcut_audit.probes import (
    balanced_accuracy,
    cross_folds,
    cv_score,
    group_folds,
    holm,
    lift,
    make_model,
    permutation_probe,
    random_folds,
)


def test_holm_matches_reference():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert holm([]) == []


def test_balanced_accuracy_and_lift():
    y = np.array([0, 0, 0, 0, 1, 1])
    assert balanced_accuracy(y, np.zeros(6, int)) == pytest.approx(0.5)
    assert balanced_accuracy(y, y) == 1.0
    assert lift(0.5, 0.5) == 0.0 and lift(1.0, 0.25) == 1.0


def test_cramers_v_extremes():
    a = np.repeat(["s1", "s2", "s3", "s4"], 25)
    v, *_ = cramers_v(a, a)
    assert v == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(0)
    v0, *_ = cramers_v(rng.integers(0, 4, 2000), rng.integers(0, 4, 2000))
    assert v0 < 0.05


def test_blockiness_extremes():
    y = np.repeat([0, 1, 2, 3], 20)
    b, same, expected, p = blockiness(np.arange(80), y, n_perm=500)
    assert b > 0.9 and p < 0.01
    rng = np.random.default_rng(1)
    b2, *_ , p2 = blockiness(np.arange(80), rng.permutation(y), n_perm=500)
    assert abs(b2) < 0.3


def test_kruskal_epsilon2():
    g = np.repeat(["a", "b"], 50)
    rng = np.random.default_rng(0)
    e_big, p_big = kruskal_epsilon2(np.r_[rng.normal(0, 1, 50), rng.normal(5, 1, 50)], g)
    e_none, _ = kruskal_epsilon2(rng.normal(0, 1, 100), g)
    assert e_big > 0.6 and p_big < 1e-6 and e_none < 0.05


def test_group_folds_are_label_independent_and_disjoint():
    groups = np.repeat(np.arange(30), 3)
    y1 = np.tile([0, 1, 2], 30)
    y2 = np.random.default_rng(0).permutation(y1)
    f1, f2 = group_folds(y1, groups, 5, 0), group_folds(y2, groups, 5, 0)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(f1, f2))
    for train, test in f1:
        assert not set(groups[train]) & set(groups[test])


def test_cross_folds_hold_out_both_factors():
    a = np.repeat(["s1", "s2", "s3"], 12)
    b = np.tile(np.repeat(["o1", "o2", "o3", "o4"], 3), 3)
    for train, test in cross_folds(a, b):
        assert not set(a[train]) & set(a[test])
        assert not set(b[train]) & set(b[test])


def test_permutation_probe_null_and_signal():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 40)
    noise = rng.normal(size=(120, 2))
    signal = np.c_[y + rng.normal(0, 0.3, 120), rng.normal(size=120)]

    def scorer(X):
        return lambda yy: cv_score(lambda: make_model("knn"), X, yy, random_folds(yy, 5, 0))

    null = permutation_probe(scorer(noise), y, None, 99, 0)
    hit = permutation_probe(scorer(signal), y, None, 99, 0)
    assert null.p_value > 0.05
    assert hit.p_value <= 0.01 and hit.score > 0.8


def test_unit_level_permutation_keeps_units_intact():
    y = np.repeat([0, 1], 50)
    units = np.repeat(np.arange(20), 5)
    X = np.random.default_rng(0).normal(size=(100, 1))
    res = permutation_probe(lambda yy: cv_score(lambda: make_model("knn"), X, yy, group_folds(yy, units, 5, 0)),
                            y, units, 19, 0)
    assert res.permutation_level == "unit"


def test_null_false_positive_rate_is_controlled():
    """No signal anywhere: nuisance probes should rarely fire."""
    import shortcut_audit as sa

    fired = 0
    runs = 12
    for seed in range(runs):
        rng = np.random.default_rng(100 + seed)
        n = 80
        ds = sa.AuditDataset(
            y=rng.integers(0, 3, n),
            features=rng.normal(size=(n, 5)),
            meta={"order": rng.permutation(n), "gain": rng.normal(size=n)},
            order="order",
            nuisance=["gain"],
        )
        rep = sa.audit(ds, only=["nuisance"], seed=seed)
        fired += any(f.verdict.value in ("warn", "fail") for f in rep.findings)
    assert fired <= 2
