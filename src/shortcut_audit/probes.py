"""Cross-validated probes, label-permutation tests and multiple-testing correction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Folds = List[Tuple[np.ndarray, np.ndarray]]

MAX_LEAVE_ONE_OUT_GROUPS = 12


# ------------------------------------------------------------------ models


class _AdaptiveKNN(KNeighborsClassifier):
    """k-NN whose k shrinks for tiny training sets (k = min(7, n_train - 1))."""

    def __init__(self, n_neighbors: int = 7):
        super().__init__(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.n_neighbors = int(max(1, min(7, len(y) - 1)))
        return super().fit(X, y)


def make_model(kind: str, seed: int = 0, n_trees: int = 64):
    """Probe classifiers.

    * ``knn`` (default for nuisance probes): k-nearest neighbours on standardized
      features. Local and non-parametric, so it catches blocks in recording order,
      thresholds on an offset or clusters of a session — and it is fast enough to run
      a full permutation test.
    * ``forest``: extremely randomized trees; slower, more robust to many irrelevant
      dimensions.
    * ``linear``: the classic linear probe (standardized logistic regression).
    """
    if kind == "knn":
        return make_pipeline(StandardScaler(), _AdaptiveKNN())
    if kind == "forest":
        return ExtraTreesClassifier(
            n_estimators=n_trees, min_samples_leaf=1, max_features="sqrt", n_jobs=1, random_state=seed
        )
    if kind == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    raise ValueError(f"unknown probe model {kind!r} (use 'knn', 'forest' or 'linear')")


# ------------------------------------------------------------------ folds


def _n_splits_for(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(max(2, min(requested, counts.min())))


def random_folds(y: np.ndarray, n_splits: int, seed: int) -> Folds:
    k = _n_splits_for(y, n_splits)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def group_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> Folds:
    """Grouped folds that do not depend on the labels.

    Label-independent folds keep the permutation test exact: permuting labels never
    changes which samples are held out together. Up to 12 groups -> leave-one-group-out;
    more -> groups shuffled (seeded) and dealt into ``n_splits`` folds.
    """
    groups = np.asarray(groups).astype(str)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("need at least two groups for grouped cross-validation")
    if len(uniq) <= MAX_LEAVE_ONE_OUT_GROUPS:
        return list(LeaveOneGroupOut().split(np.zeros(len(groups)), None, groups))
    k = min(n_splits, len(uniq))
    rng = np.random.default_rng(seed)
    fold_of_group = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % k))
    fold = np.array([fold_of_group[g] for g in groups])
    idx = np.arange(len(groups))
    return [(idx[fold != f], idx[fold == f]) for f in range(k)]


def cross_folds(a: np.ndarray, b: np.ndarray) -> Folds:
    """Joint leave-out: for every (a_i, b_j) cell, train on samples with a != a_i AND
    b != b_j, test on the cell. Neither the session nor the instance of a test sample
    was seen during training."""
    folds: Folds = []
    idx = np.arange(len(a))
    for va in np.unique(a):
        for vb in np.unique(b):
            test = idx[(a == va) & (b == vb)]
            if len(test) == 0:
                continue
            train = idx[(a != va) & (b != vb)]
            if len(train) == 0:
                continue
            folds.append((train, test))
    return folds


# ------------------------------------------------------------------ scoring


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in classes]
    return float(np.mean(recalls))


def chance_level(y_tested: np.ndarray) -> float:
    return 1.0 / len(np.unique(y_tested))


def lift(score: float, chance: float) -> float:
    """Rescale balanced accuracy so that chance = 0 and perfect = 1."""
    if chance >= 1:
        return 0.0
    return float((score - chance) / (1.0 - chance))


def cv_predict(model_factory: Callable[[], object], X: np.ndarray, y: np.ndarray, folds: Folds) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions. Returns (predictions, tested_mask)."""
    pred = np.full(len(y), -1, dtype=int)
    tested = np.zeros(len(y), dtype=bool)
    for train, test in folds:
        ytr = y[train]
        if len(np.unique(ytr)) < 2:
            pred[test] = ytr[0]
        else:
            model = model_factory()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X[train], ytr)
                pred[test] = model.predict(X[test])
        tested[test] = True
    return pred, tested


def cv_score(model_factory: Callable[[], object], X: np.ndarray, y: np.ndarray, folds: Folds) -> Tuple[float, float, int]:
    """Balanced accuracy over all out-of-fold predictions, chance level, n tested."""
    pred, tested = cv_predict(model_factory, X, y, folds)
    yt, pt = y[tested], pred[tested]
    if len(yt) == 0:
        return float("nan"), float("nan"), 0
    return balanced_accuracy(yt, pt), chance_level(yt), int(tested.sum())


# ------------------------------------------------------------------ permutation test


def _unit_permuter(y: np.ndarray, units: Optional[np.ndarray]):
    """Return a function rng -> permuted labels.

    If samples come in units (several windows cut from one trial) and the label is
    constant within each unit, labels are permuted *between units*. Permuting single
    windows would treat dependent windows as independent and make p-values too small.
    """
    if units is not None:
        uniq, inv = np.unique(units, return_inverse=True)
        unit_label = np.full(len(uniq), -1)
        consistent = True
        for u in range(len(uniq)):
            vals = np.unique(y[inv == u])
            if len(vals) != 1:
                consistent = False
                break
            unit_label[u] = vals[0]
        if consistent:
            def permute(rng: np.random.Generator) -> np.ndarray:
                return rng.permutation(unit_label)[inv]

            return permute, "unit"

    def permute_samples(rng: np.random.Generator) -> np.ndarray:
        return rng.permutation(y)

    return permute_samples, "sample"


@dataclass
class ProbeResult:
    score: float
    chance: float
    lift: float
    p_value: float
    n_permutations: int
    n_tested: int
    permutation_level: str
    null_mean: float
    null_q95: float

    def as_metrics(self) -> dict:
        return {
            "balanced_accuracy": round(self.score, 4),
            "chance": round(self.chance, 4),
            "lift": round(self.lift, 4),
            "p_value": self.p_value,
            "permutations": self.n_permutations,
            "permutation_level": self.permutation_level,
            "null_mean": round(self.null_mean, 4),
            "null_q95": round(self.null_q95, 4),
            "n_tested": self.n_tested,
        }


def permutation_probe(
    score_fn: Callable[[np.ndarray], Tuple[float, float, int]],
    y: np.ndarray,
    units: Optional[np.ndarray],
    n_permutations: int,
    seed: int,
    stop_after_exceedances: int = 10,
) -> ProbeResult:
    """Score the probe on true labels, then on permuted labels.

    Uses the Besag & Clifford (1991) sequential Monte Carlo rule: once
    ``stop_after_exceedances`` permuted scores reach the observed score the probe is
    clearly null and permutation stops early (p = h / l). Otherwise
    p = (exceedances + 1) / (n_permutations + 1).
    """
    observed, chance, n_tested = score_fn(y)
    permute, level = _unit_permuter(y, units)
    rng = np.random.default_rng(seed)
    null: List[float] = []
    exceed = 0
    p_value = None
    for i in range(n_permutations):
        s, _, _ = score_fn(permute(rng))
        null.append(s)
        if s >= observed - 1e-12:
            exceed += 1
            if exceed >= stop_after_exceedances:
                p_value = exceed / (i + 1)
                break
    if p_value is None:
        p_value = (exceed + 1) / (len(null) + 1)
    null_arr = np.asarray(null) if null else np.asarray([chance])
    return ProbeResult(
        score=observed,
        chance=chance,
        lift=lift(observed, chance),
        p_value=float(p_value),
        n_permutations=len(null),
        n_tested=n_tested,
        permutation_level=level,
        null_mean=float(null_arr.mean()),
        null_q95=float(np.quantile(null_arr, 0.95)),
    )


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ------------------------------------------------------------------ encoding


def encode_column(col: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Numeric columns pass through; anything else is one-hot encoded."""
    col = np.asarray(col)
    if np.issubdtype(col.dtype, np.number) or np.issubdtype(col.dtype, np.bool_):
        return col.astype(float)[:, None], ["value"]
    if np.issubdtype(col.dtype, np.datetime64):
        return col.astype("datetime64[ns]").astype("int64").astype(float)[:, None], ["time"]
    uniq, inv = np.unique(col.astype(str), return_inverse=True)
    out = np.zeros((len(col), len(uniq)))
    out[np.arange(len(col)), inv] = 1.0
    return out, [str(u) for u in uniq]
