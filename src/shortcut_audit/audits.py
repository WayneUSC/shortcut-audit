"""The audits. Each one answers a single question and returns a :class:`Section`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .config import AuditConfig
from .dataset import AuditDataset
from .features import FeatureSet, baseline_features, signal_features
from .probes import (
    Folds,
    cross_folds,
    cv_score,
    encode_column,
    group_folds,
    holm,
    lift,
    make_model,
    permutation_probe,
    random_folds,
)
from .result import Chart, Finding, Section, Verdict

# ---------------------------------------------------------------------- context


@dataclass
class Context:
    ds: AuditDataset
    cfg: AuditConfig
    y: np.ndarray
    eval_cv: str
    main_X: Optional[np.ndarray]
    main_names: Optional[List[str]]
    sig: Optional[FeatureSet]
    nested: Dict[str, bool] = field(default_factory=dict)  # group -> treated as nested in label
    cramers: Dict[str, float] = field(default_factory=dict)
    _fold_cache: Dict[str, Folds] = field(default_factory=dict)

    @property
    def units(self) -> Optional[np.ndarray]:
        return self.ds.meta[self.ds.unit] if self.ds.unit else None

    def folds(self, strategy: str, y: np.ndarray) -> Folds:
        """Folds for a strategy. Grouped folds ignore labels and are cached; stratified
        random folds are rebuilt for every (permuted) label vector."""
        ds, cfg = self.ds, self.cfg
        if strategy == "random":
            return random_folds(y, cfg.n_splits, cfg.seed)
        if strategy in self._fold_cache:
            return self._fold_cache[strategy]
        if strategy == "unit":
            folds = group_folds(y, ds.meta[ds.unit], cfg.n_splits, cfg.seed)
        elif strategy.startswith("group:"):
            folds = group_folds(y, ds.meta[strategy.split(":", 1)[1]], cfg.n_splits, cfg.seed)
        elif strategy.startswith("cross:"):
            a, b = strategy.split(":", 1)[1].split(",")
            folds = cross_folds(ds.meta[a].astype(str), ds.meta[b].astype(str))
        else:
            raise ValueError(f"unknown cv strategy {strategy!r}")
        self._fold_cache[strategy] = folds
        return folds

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))

    def crossed_groups(self) -> List[str]:
        return [g for g in self.ds.groups if not self.nested.get(g, False)]


def _pure_values(groups: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Group values that contain a single label, and the per-sample 'is pure' mask."""
    uniq, inv = np.unique(groups.astype(str), return_inverse=True)
    pure = np.zeros(len(uniq), dtype=bool)
    for i in range(len(uniq)):
        pure[i] = len(np.unique(y[inv == i])) == 1
    return uniq[pure], pure[inv]


def build_context(ds: AuditDataset, cfg: AuditConfig) -> Context:
    y = ds.y_codes
    if cfg.cv == "auto":
        eval_cv = "unit" if ds.unit else "random"
    else:
        eval_cv = cfg.cv
        if eval_cv == "unit" and not ds.unit:
            raise ValueError("cv='unit' requires the dataset to declare a unit column")
        if eval_cv.startswith("group:") and eval_cv.split(":", 1)[1] not in ds.meta:
            raise ValueError(f"cv={cfg.cv!r}: column not found")
    sig = signal_features(ds.signals, ds.fs, ds.channel_names, cfg.bands) if ds.has_signals else None
    if ds.features is not None:
        main_X, main_names = ds.features, list(ds.feature_names or [])
    elif sig is not None:
        main_X, main_names = sig.X, sig.names
    else:
        main_X, main_names = None, None
    ctx = Context(ds=ds, cfg=cfg, y=y, eval_cv=eval_cv, main_X=main_X, main_names=main_names, sig=sig)
    n_classes = len(np.unique(y))
    for g in ds.groups:
        col = ds.meta[g].astype(str)
        _, pure_mask = _pure_values(col, y)
        n_values = len(np.unique(col))
        if g in cfg.nested:
            ctx.nested[g] = True
        else:
            # Fully nested with several values per class looks like an identity factor
            # (object instance, subject). It is still reported, see audit_confound.
            ctx.nested[g] = bool(pure_mask.all() and n_values > n_classes)
        ctx.cramers[g] = cramers_v(col, ds.y)[0]
    return ctx


# ---------------------------------------------------------------------- statistics


def cramers_v(a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray, float, List[str], List[str]]:
    """Bias-corrected Cramér's V (Bergsma 2013) with its contingency table and chi² p-value."""
    ua, ia = np.unique(np.asarray(a).astype(str), return_inverse=True)
    ub, ib = np.unique(np.asarray(b).astype(str), return_inverse=True)
    table = np.zeros((len(ua), len(ub)))
    np.add.at(table, (ia, ib), 1)
    n = table.sum()
    r, k = table.shape
    if min(r, k) < 2 or n < 2:
        return 0.0, table, 1.0, list(ua), list(ub)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    v = float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0
    return min(v, 1.0), table, float(p), list(ua), list(ub)


def blockiness(order: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 0) -> Tuple[float, float, float, float]:
    """How much more often consecutive recordings share a label than chance.

    Returns (blockiness, observed same-label rate, expected rate, p-value). Blockiness is
    0 for a random schedule and 1 when every class was recorded in one uninterrupted run.
    """
    idx = np.argsort(order, kind="stable")
    ys = y[idx]
    same = float(np.mean(ys[1:] == ys[:-1]))
    _, counts = np.unique(y, return_counts=True)
    n = len(y)
    expected = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(ys) for _ in range(n_perm)])
    null = np.mean(perms[:, 1:] == perms[:, :-1], axis=1)
    p = float((np.sum(null >= same - 1e-12) + 1) / (n_perm + 1))
    b = (same - expected) / (1 - expected) if expected < 1 else 0.0
    return float(b), same, expected, p


def kruskal_epsilon2(values: np.ndarray, groups: np.ndarray) -> Tuple[float, float]:
    """Kruskal–Wallis H test across groups with epsilon² effect size (0.26 ≈ large)."""
    uniq = np.unique(groups.astype(str))
    samples = [values[groups.astype(str) == g] for g in uniq]
    samples = [s for s in samples if len(s) > 0]
    n = len(values)
    k = len(samples)
    if k < 2 or n <= k:
        return 0.0, 1.0
    if np.allclose(values, values[0]):
        return 0.0, 1.0
    h, p = stats.kruskal(*samples)
    eps2 = max(0.0, (h - k + 1) / (n - k))
    return float(min(eps2, 1.0)), float(p)


def _fmt_p(p: float) -> str:
    return f"p={p:.2g}" if p >= 0.001 else "p<0.001"


def _single_class(ctx: "Context", sec: Section) -> bool:
    if ctx.n_classes >= 2:
        return False
    sec.findings.append(Finding(
        sec.name, "labels", Verdict.SKIP,
        "Only one class: label-based probes need at least two. Choose another target to audit.",
    ))
    return True


# ---------------------------------------------------------------------- 1. confound


def audit_confound(ctx: Context) -> Section:
    ds, cfg, y = ctx.ds, ctx.cfg, ctx.y
    sec = Section(
        name="confound",
        title="Confounded design",
        question="Is the label entangled with how or when the data was recorded?",
    )
    if _single_class(ctx, sec):
        return sec
    labels, values = [], []
    n_classes = len(np.unique(y))
    for g in ds.groups:
        col = ds.meta[g]
        v, table, p, rows, cols = cramers_v(col, ds.y)
        pure_vals, pure_mask = _pure_values(col, y)
        n_values = len(rows)
        metrics = {
            "cramers_v": round(v, 4),
            "chi2_p": p,
            "n_values": n_values,
            "values_with_single_label": int(len(pure_vals)),
            "samples_in_single_label_values": round(float(pure_mask.mean()), 4),
        }
        if n_values <= 30 and n_classes <= 30:
            metrics["table"] = {r: {c: int(table[i, j]) for j, c in enumerate(cols)} for i, r in enumerate(rows)}
        labels.append(f"{g} (nested)" if ctx.nested.get(g) else g)
        values.append(v)
        if g in cfg.nested:
            sec.findings.append(Finding(
                "confound", g, Verdict.PASS,
                f"'{g}' is nested in the label by design (declared). Generalization across it is tested in the protocol audit.",
                metrics,
            ))
            continue
        if pure_mask.all() and n_values <= n_classes:
            sec.findings.append(Finding(
                "confound", g, Verdict.FAIL,
                f"Label and '{g}' are aliased one-to-one: every '{g}' contains exactly one class. "
                f"Nothing in the data can separate a class effect from a '{g}' effect.",
                metrics,
                advice=f"Record every class in every '{g}' (counterbalance), or drop claims that the model "
                f"recognizes the class rather than the '{g}'.",
            ))
            continue
        if pure_mask.all():
            sec.findings.append(Finding(
                "confound", g, Verdict.WARN,
                f"Every '{g}' value holds a single class. That is expected if '{g}' is an identity "
                f"(object instance, subject) — declare it with nested=['{g}'] / --nested {g}. "
                f"If '{g}' is an acquisition factor (session, day, operator), the design is fully blocked.",
                metrics,
                advice=f"If '{g}' is a session-like factor, record every class in every '{g}'.",
            ))
            continue
        if v >= cfg.confound_fail:
            verdict = Verdict.FAIL
        elif v >= cfg.confound_warn:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.PASS
        summary = f"Cramér's V between label and '{g}' = {v:.2f}"
        if verdict == Verdict.PASS:
            summary += " — classes are spread evenly enough across values."
        else:
            summary += f"; {int(len(pure_vals))}/{n_values} values contain a single class."
        sec.findings.append(Finding(
            "confound", g, verdict, summary, metrics,
            advice=None if verdict == Verdict.PASS else
            f"Class proportions differ across '{g}'. Any '{g}'-specific artifact (calibration, temperature, "
            f"wear, lighting) becomes a label proxy. Counterbalance, or evaluate with leave-{g}-out.",
        ))
    if ds.order:
        order_col, y_ord = ds.meta[ds.order], y
        if ds.unit:
            # One entry per recording: windows of the same trial are trivially adjacent.
            _, first = np.unique(ds.meta[ds.unit].astype(str), return_index=True)
            order_col, y_ord = order_col[first], y[first]
        b, same, expected, p = blockiness(order_col, y_ord, seed=cfg.seed)
        verdict = Verdict.WARN if (b >= cfg.blockiness_warn and p < cfg.alpha) else Verdict.PASS
        labels.append(f"{ds.order} (blockiness)")
        values.append(max(0.0, b))
        sec.findings.append(Finding(
            "confound", "recording order", verdict,
            f"Consecutive recordings share a label {same:.0%} of the time (random schedule: {expected:.0%}); "
            f"blockiness {b:.2f}, {_fmt_p(p)}.",
            {"blockiness": round(b, 4), "same_label_rate": round(same, 4), "expected_rate": round(expected, 4), "p_value": p},
            advice=None if verdict == Verdict.PASS else
            "Classes were recorded in runs. Any slow change (sensor drift, gel wear, temperature, battery, "
            "operator fatigue) is now correlated with the label. Interleave classes in a randomized schedule.",
        ))
    if not sec.findings:
        sec.findings.append(Finding(
            "confound", "design", Verdict.SKIP,
            "No grouping factors or recording order given. Pass groups=[...] / --groups and order= / --order.",
        ))
    if labels:
        sec.charts.append(Chart(
            title="Label association per factor", labels=labels, values=values,
            reference=cfg.confound_warn, reference_label="warn", value_label="Cramér's V / blockiness",
        ))
    return sec


# ---------------------------------------------------------------------- 2. nuisance


@dataclass
class _Probe:
    name: str
    X: np.ndarray
    kind: str  # order | group | column | baseline | block | combined
    advice: str


def _nuisance_probes(ctx: Context) -> List[_Probe]:
    ds, cfg = ctx.ds, ctx.cfg
    probes: List[_Probe] = []
    if ds.order:
        X, _ = encode_column(ds.meta[ds.order])
        probes.append(_Probe(
            f"{ds.order}", X, "order",
            "Recording order predicts the label: classes were recorded in runs and something drifted. "
            "Randomize the schedule, re-baseline sensors per trial, and report leave-session-out results.",
        ))
    for g in ctx.crossed_groups():
        X, _ = encode_column(ds.meta[g].astype(str))
        probes.append(_Probe(
            f"{g} id", X, "group",
            f"Knowing only the '{g}' predicts the label because class proportions differ across '{g}'. "
            f"Counterbalance the design.",
        ))
    for col in ds.nuisance:
        X, _ = encode_column(ds.meta[col])
        probes.append(_Probe(
            col, X, "column",
            ds.nuisance_advice.get(col, f"'{col}' should carry no label information but does. Find out why "
            f"(it may be recorded differently per class) and remove or balance it."),
        ))
    if ds.has_signals and cfg.baseline_s:
        fb = baseline_features(ds.signals, ds.fs, cfg.baseline_s, ds.channel_names)
        probes.append(_Probe(
            f"baseline (first {cfg.baseline_s:g} s)", fb.X, "baseline",
            "The sensor state before interaction predicts the label: baseline drift, wear or temperature "
            "differs by class. Subtract a per-trial baseline and re-zero sensors between classes.",
        ))
    for key, block in ds.nuisance_blocks.items():
        probes.append(_Probe(
            key, block, "block",
            ds.nuisance_advice.get(key, f"'{key}' should carry no label information but does."),
        ))
    if len(probes) >= 2:
        probes.append(_Probe(
            "all recording metadata", np.hstack([p.X for p in probes]), "combined",
            "Together, the recording metadata predicts the label. See the individual probes above.",
        ))
    return probes


def audit_nuisance(ctx: Context) -> Section:
    ds, cfg = ctx.ds, ctx.cfg
    sec = Section(
        name="nuisance",
        title="Nuisance-only probes",
        question="Can a model that sees only things that should be irrelevant beat chance?",
    )
    if _single_class(ctx, sec):
        return sec
    probes = _nuisance_probes(ctx)
    if not probes:
        sec.findings.append(Finding(
            "nuisance", "probes", Verdict.SKIP,
            "Nothing to probe. Provide recording order (--order), nuisance columns (--nuisance), "
            "crossed grouping factors (--groups) or a pre-contact window (--baseline-s).",
        ))
        return sec
    n_perm = cfg.permutations or int(max(99, np.ceil(2 * len(probes) / cfg.alpha) - 1))
    results = []
    for i, probe in enumerate(probes):
        X = probe.X

        def score_fn(yy: np.ndarray, X=X) -> Tuple[float, float, int]:
            return cv_score(lambda: make_model(cfg.probe, cfg.seed), X, yy, ctx.folds(ctx.eval_cv, yy))

        results.append(permutation_probe(score_fn, ctx.y, ctx.units, n_perm, cfg.seed + i))
    p_adj = holm([r.p_value for r in results])
    for probe, res, pa in zip(probes, results, p_adj):
        significant = pa < cfg.alpha
        if significant and res.lift >= cfg.fail_lift:
            verdict = Verdict.FAIL
        elif significant and res.lift >= cfg.warn_lift:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.PASS
        metrics = res.as_metrics()
        metrics["p_holm"] = pa
        metrics["n_features"] = int(probe.X.shape[1])
        summary = (
            f"Seeing only '{probe.name}': balanced accuracy {res.score:.2f} vs chance {res.chance:.2f} "
            f"(Holm {_fmt_p(pa)})."
        )
        sec.findings.append(Finding("nuisance", probe.name, verdict, summary, metrics,
                                    advice=probe.advice if verdict != Verdict.PASS else None))
    sec.charts.append(Chart(
        title=f"Balanced accuracy from nuisance alone (cv: {ctx.eval_cv})",
        labels=[p.name for p in probes],
        values=[r.score for r in results],
        reference=float(np.mean([r.chance for r in results])),
        highlight=[f.verdict.value if f.verdict in (Verdict.WARN, Verdict.FAIL) else "" for f in sec.findings],
    ))
    sec.notes.append(
        f"Probe model: {cfg.probe}; cross-validation: {ctx.eval_cv}; up to {n_perm} label "
        f"permutations per probe ({results[0].permutation_level}-level); Holm correction across "
        f"{len(probes)} probes."
    )
    return sec


# ---------------------------------------------------------------------- 3. protocol


def audit_protocol(ctx: Context) -> Section:
    ds, cfg = ctx.ds, ctx.cfg
    sec = Section(
        name="protocol",
        title="Evaluation protocol gap",
        question="How much does accuracy depend on having seen the same session / object / recording?",
    )
    if _single_class(ctx, sec):
        return sec
    if ctx.main_X is None:
        sec.findings.append(Finding("protocol", "reference probe", Verdict.SKIP,
                                    "No features or signals to train a reference probe on."))
        return sec
    X = ctx.main_X
    protocols: List[Tuple[str, str]] = [("random split", "random")]
    if ds.unit:
        protocols.append((f"leave-{ds.unit}-out", "unit"))
    for g in ds.groups:
        if len(np.unique(ds.meta[g])) >= 2:
            protocols.append((f"leave-{g}-out", f"group:{g}"))
    if cfg.cross:
        a, b = cfg.cross
        protocols.append((f"leave-{a}-and-{b}-out", f"cross:{a},{b}"))
    if len(protocols) == 1:
        sec.findings.append(Finding(
            "protocol", "grouped splits", Verdict.SKIP,
            "Only a random split is possible. Declare grouping factors (--groups) or a unit (--unit) to compare protocols.",
        ))
    probe_names = list(cfg.reference_probes)
    # scores[label] = {probe: (balanced accuracy, chance, n_tested)}
    scores: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for label, strategy in protocols:
        try:
            folds = ctx.folds(strategy, ctx.y)
        except ValueError as exc:
            sec.notes.append(f"{label}: skipped ({exc})")
            continue
        if not folds:
            sec.notes.append(f"{label}: skipped (no usable folds)")
            continue
        scores[label] = {
            p: cv_score(lambda p=p: make_model(p, cfg.seed), X, ctx.y, folds) for p in probe_names
        }
    ref = scores["random split"]
    flagged: Dict[str, str] = {}
    for label, strategy in protocols:
        if label not in scores:
            continue
        per = scores[label]
        metrics = {p: {"balanced_accuracy": round(s, 4), "chance": round(c, 4), "n_tested": n}
                   for p, (s, c, n) in per.items()}
        chance = next(iter(per.values()))[1]
        if strategy == "random":
            txt = ", ".join(f"{p} {s:.2f}" for p, (s, _, _) in per.items())
            sec.findings.append(Finding(
                "protocol", label, Verdict.PASS,
                f"Reference — random split: {txt} (chance {chance:.2f}).", metrics,
            ))
            continue
        gaps = {p: ref[p][0] - per[p][0] for p in per}
        for p in per:
            metrics[p]["gap_vs_random"] = round(gaps[p], 4)
        worst_probe = max(gaps, key=gaps.get)
        gap = gaps[worst_probe]
        if gap >= cfg.gap_fail:
            verdict = Verdict.FAIL
        elif gap >= cfg.gap_warn:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.PASS
        metrics["max_gap"] = round(gap, 4)
        target = label.replace("leave-", "").replace("-out", "").replace("-and-", " and ")
        parts = [f"{p} {per[p][0]:.2f} ({'−' if gaps[p] >= 0 else '+'}{abs(gaps[p]):.2f})" for p in per]
        summary = f"{label}: " + ", ".join(parts) + " vs random split."
        advice = None
        if verdict != Verdict.PASS:
            advice = (
                f"Random-split numbers overstate generalization to unseen {target} by up to {gap:.2f} "
                f"({worst_probe} probe). Report {label} as the headline number, or collect more {target} diversity."
            )
        sec.findings.append(Finding("protocol", label, verdict, summary, metrics, advice=advice))
        flagged[label] = verdict.value if verdict in (Verdict.WARN, Verdict.FAIL) else ""
    labels = list(scores)
    chances = [next(iter(scores[l].values()))[1] for l in labels]
    sec.charts.append(Chart(
        title="Reference probes under different splits",
        labels=labels,
        values=[scores[l][probe_names[0]][0] for l in labels],
        series_name=probe_names[0],
        extra={p: [scores[l][p][0] for l in labels] for p in probe_names[1:]},
        reference=float(np.nanmean(chances)),
        highlight=[flagged.get(l, "") for l in labels],
    ))
    sec.notes.append(
        "Reference probes are deliberately simple: a linear probe (generalizable structure) and k-NN "
        "(a memorizer, the most sensitive to near-copies). Absolute numbers will be below a tuned model; "
        "the gap between protocols is what matters."
    )
    return sec


# ---------------------------------------------------------------------- 4. leakage


def _row_hashes(X: np.ndarray) -> np.ndarray:
    Xr = np.round(np.asarray(X, dtype=float), 9)
    Xr = np.ascontiguousarray(Xr.reshape(len(Xr), -1))
    return np.array([hash(row.tobytes()) for row in Xr])


def audit_leakage(ctx: Context) -> Section:
    ds, cfg = ctx.ds, ctx.cfg
    sec = Section(
        name="leakage",
        title="Split leakage",
        question="Does the test set contain recordings, groups or near-copies the model saw in training?",
    )
    # Exact duplicates (no split needed)
    raw = ds.signals if ds.has_signals else ctx.main_X
    hashes = None
    if raw is not None:
        hashes = _row_hashes(raw)
    elif ds.sequences is not None:
        hashes = np.array([hash(np.round(np.asarray(q, dtype=float), 9).tobytes()) for q in ds.sequences])
    if hashes is not None:
        _, inv, counts = np.unique(hashes, return_inverse=True, return_counts=True)
        dup_mask = counts[inv] > 1
        n_dup = int(dup_mask.sum())
        conflicting = 0
        for h in np.unique(inv[dup_mask]):
            if len(np.unique(ctx.y[inv == h])) > 1:
                conflicting += 1
        if n_dup == 0:
            sec.findings.append(Finding("leakage", "exact duplicates", Verdict.PASS,
                                        "No two samples are identical.", {"duplicate_samples": 0}))
        else:
            verdict = Verdict.FAIL if conflicting else Verdict.WARN
            sec.findings.append(Finding(
                "leakage", "exact duplicates", verdict,
                f"{n_dup} samples ({n_dup / len(ctx.y):.1%}) are exact copies of another sample"
                + (f"; {conflicting} duplicate sets carry conflicting labels." if conflicting else "."),
                {"duplicate_samples": n_dup, "conflicting_sets": conflicting},
                advice="Duplicates usually come from copied files or re-exported episodes. "
                "Deduplicate before splitting so a copy cannot sit on both sides.",
            ))
    if ds.split is None:
        sec.findings.append(Finding(
            "leakage", "train/test split", Verdict.SKIP,
            "No split given. Pass split= / --split to check a concrete split; the protocol audit already "
            "compares split strategies.",
        ))
        return sec
    train = ds.split == "train"
    if not train.any() or train.all():
        sec.findings.append(Finding("leakage", "train/test split", Verdict.SKIP,
                                    "Split column needs a 'train' value and at least one other value."))
        return sec
    test = ~train
    for col, is_unit in ([(ds.unit, True)] if ds.unit else []) + [(g, False) for g in ds.groups]:
        vals = ds.meta[col].astype(str)
        shared = np.intersect1d(np.unique(vals[train]), np.unique(vals[test]))
        frac = float(np.isin(vals[test], shared).mean())
        metrics = {"shared_values": int(len(shared)), "test_samples_with_seen_value": round(frac, 4)}
        if len(shared) == 0:
            sec.findings.append(Finding("leakage", f"{col} overlap", Verdict.PASS,
                                        f"No '{col}' appears in both train and test.", metrics))
        elif is_unit:
            sec.findings.append(Finding(
                "leakage", f"{col} overlap", Verdict.FAIL,
                f"{len(shared)} '{col}' values are split across train and test ({frac:.0%} of test samples). "
                f"Windows cut from the same recording are near-copies.",
                metrics, advice=f"Split by '{col}' (group split), never by window.",
            ))
        else:
            sec.findings.append(Finding(
                "leakage", f"{col} overlap", Verdict.WARN,
                f"{len(shared)} '{col}' values appear in both train and test ({frac:.0%} of test samples).",
                metrics,
                advice=f"Fine for an in-distribution claim; if you claim generalization to new '{col}', "
                f"hold whole '{col}' values out.",
            ))
    if ctx.main_X is not None:
        sec.findings.append(_near_duplicates(ctx, train, test))
    return sec


def _near_duplicates(ctx: Context, train: np.ndarray, test: np.ndarray) -> Finding:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    cfg = ctx.cfg
    X = ctx.main_X
    rng = np.random.default_rng(cfg.seed)
    tr_idx, te_idx = np.flatnonzero(train), np.flatnonzero(test)
    cap = cfg.max_neighbors_samples
    if len(tr_idx) > cap:
        tr_idx = np.sort(rng.choice(tr_idx, cap, replace=False))
    if len(te_idx) > cap:
        te_idx = np.sort(rng.choice(te_idx, cap, replace=False))
    scaler = StandardScaler().fit(X[tr_idx])
    Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
    k = min(len(tr_idx), 25)
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    d_cross = nn.kneighbors(Xte, n_neighbors=1)[0][:, 0]
    # Reference: nearest *independent* training neighbour (different unit if known).
    dist, ind = nn.kneighbors(Xtr, n_neighbors=k)
    units = ctx.units[tr_idx] if ctx.units is not None else None
    d_ref = np.empty(len(tr_idx))
    for i in range(len(tr_idx)):
        ok = ind[i] != i
        if units is not None:
            ok &= units[ind[i]] != units[i]
        d_ref[i] = dist[i][ok][0] if ok.any() else dist[i][-1]
    thr = float(np.quantile(d_ref, cfg.neardup_quantile))
    rate = float(np.mean(d_cross <= thr))
    ratio = rate / cfg.neardup_quantile
    metrics = {
        "near_duplicate_rate": round(rate, 4),
        "expected_rate": cfg.neardup_quantile,
        "ratio": round(ratio, 2),
        "threshold_distance": round(thr, 6),
        "median_test_to_train_distance": round(float(np.median(d_cross)), 4),
        "median_independent_distance": round(float(np.median(d_ref)), 4),
    }
    if ratio >= cfg.neardup_fail:
        verdict = Verdict.FAIL
    elif ratio >= cfg.neardup_warn:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.PASS
    summary = (
        f"{rate:.1%} of test samples sit closer to a training sample than {cfg.neardup_quantile:.0%} of "
        f"independent training pairs do (expected ≈{cfg.neardup_quantile:.0%}; {ratio:.1f}×)."
    )
    return Finding(
        "leakage", "near-duplicates across split", verdict, summary, metrics,
        advice=None if verdict == Verdict.PASS else
        "Test samples have near-copies in training (overlapping windows, repeated takes, augmented copies). "
        "Split by recording or deduplicate with a distance threshold.",
    )


# ---------------------------------------------------------------------- 5. drift


def audit_drift(ctx: Context) -> Section:
    ds, cfg = ctx.ds, ctx.cfg
    sec = Section(
        name="drift",
        title="Sensor baseline drift",
        question="Does the sensor's resting level move between sessions or over time?",
    )
    if ds.levels is not None:
        level = ds.levels
        how = "resting level"
        channels = list(ds.level_names or [])
    elif ds.has_signals:
        if cfg.baseline_s:
            n_base = min(ds.signals.shape[-1], max(2, int(round(cfg.baseline_s * ds.fs))))
            level = ds.signals[..., :n_base].mean(axis=-1)
            how = f"mean of the first {cfg.baseline_s:g} s"
        else:
            level = np.median(ds.signals, axis=-1)
            how = "median of each window (pass --baseline-s for a pre-contact window)"
        channels = list(ds.channel_names or [])
    else:
        sec.findings.append(Finding("drift", "baseline", Verdict.SKIP,
                                    "Needs raw signals (N, C, T) or per-sample sensor levels."))
        return sec
    labels, values = [], []
    for g in ctx.crossed_groups():
        eps = [kruskal_epsilon2(level[:, c], ds.meta[g]) for c in range(level.shape[1])]
        worst = int(np.argmax([e[0] for e in eps]))
        e2, p = eps[worst]
        p = min(1.0, p * level.shape[1])  # Bonferroni over channels
        conf = ctx.cramers.get(g, 0.0)
        if e2 >= cfg.drift_warn and p < cfg.alpha:
            verdict = Verdict.FAIL if conf >= cfg.confound_warn else Verdict.WARN
        else:
            verdict = Verdict.PASS
        labels.append(f"across {g}")
        values.append(e2)
        msg = f"Baseline ({how}) differs across '{g}': ε²={e2:.2f} on channel '{channels[worst]}' ({_fmt_p(p)}, Bonferroni over {level.shape[1]} channels)."
        if verdict == Verdict.FAIL:
            msg += f" Class proportions also differ across '{g}' (V={conf:.2f}), so the drift is label-correlated."
        sec.findings.append(Finding(
            "drift", f"across {g}", verdict, msg,
            {"epsilon2": round(e2, 4), "p_value": p, "worst_channel": channels[worst],
             "epsilon2_per_channel": {channels[c]: round(eps[c][0], 4) for c in range(len(eps))}},
            advice=None if verdict == Verdict.PASS else
            f"Normalize per '{g}' (subtract the {g} baseline) or re-zero sensors before each trial.",
        ))
    if ds.order:
        order = ds.meta[ds.order]
        if np.issubdtype(order.dtype, np.number) or np.issubdtype(order.dtype, np.datetime64):
            order_num = order.astype("datetime64[ns]").astype("int64") if np.issubdtype(order.dtype, np.datetime64) else order.astype(float)
            rhos = []
            for c in range(level.shape[1]):
                if np.allclose(level[:, c], level[0, c]):
                    rhos.append((0.0, 1.0))
                    continue
                r = stats.spearmanr(order_num, level[:, c])
                rhos.append((float(r.statistic if hasattr(r, "statistic") else r[0]), float(r.pvalue if hasattr(r, "pvalue") else r[1])))
            worst = int(np.argmax([abs(r[0]) for r in rhos]))
            rho, p = rhos[worst]
            p = min(1.0, p * level.shape[1])  # Bonferroni over channels
            verdict = Verdict.WARN if (abs(rho) >= cfg.trend_warn and p < cfg.alpha) else Verdict.PASS
            labels.append(f"trend over {ds.order}")
            values.append(abs(rho))
            sec.findings.append(Finding(
                "drift", f"trend over {ds.order}", verdict,
                f"Baseline trend over '{ds.order}': Spearman ρ={rho:+.2f} on channel '{channels[worst]}' "
                f"({_fmt_p(p)}, Bonferroni over {level.shape[1]} channels).",
                {"spearman_rho": round(rho, 4), "p_value": p, "worst_channel": channels[worst]},
                advice=None if verdict == Verdict.PASS else
                "The resting level creeps over the recording campaign (wear, temperature, adhesive relaxation). "
                "Combined with a blocked schedule this becomes a label proxy.",
            ))
    if not sec.findings:
        sec.findings.append(Finding("drift", "baseline", Verdict.SKIP,
                                    "Needs a crossed grouping factor (--groups) or recording order (--order)."))
    if labels:
        sec.charts.append(Chart(title="Baseline drift (ε² across groups, |ρ| over time)", labels=labels,
                                values=values, reference=cfg.drift_warn, reference_label="large", value_label="effect size"))
    return sec


# ---------------------------------------------------------------------- 6. spectral


def audit_spectral(ctx: Context) -> Section:
    ds, cfg = ctx.ds, ctx.cfg
    sec = Section(
        name="spectral",
        title="Where the label lives (band sweep)",
        question="Which frequency bands carry label information — and is it only the DC offset?",
    )
    if _single_class(ctx, sec):
        return sec
    if ctx.sig is None:
        sec.findings.append(Finding("spectral", "band sweep", Verdict.SKIP, "Needs raw signals (N, C, T) and fs."))
        return sec
    sig = ctx.sig
    folds = ctx.folds(ctx.eval_cv, ctx.y)

    def score(X: np.ndarray) -> Tuple[float, float]:
        s, chance, _ = cv_score(lambda: make_model(cfg.reference_probe, cfg.seed), X, ctx.y, folds)
        return s, chance

    block_names = list(sig.blocks)
    block_scores = {b: score(sig.block(b)) for b in block_names}
    full, chance = score(sig.X)
    no_offset, _ = score(sig.without(["offset"]))
    off_s, _ = block_scores["offset"]
    off_lift = lift(off_s, chance)
    if off_lift >= cfg.fail_lift:
        v_off = Verdict.WARN
        msg = (f"The per-channel DC offset alone reaches balanced accuracy {off_s:.2f} (chance {chance:.2f}). "
               f"Unless static load is the intended signal, this is usually baseline drift or calibration.")
    else:
        v_off = Verdict.PASS
        msg = f"DC offset alone: {off_s:.2f} (chance {chance:.2f})."
    sec.findings.append(Finding("spectral", "offset only", v_off, msg,
                                {"balanced_accuracy": round(off_s, 4), "chance": round(chance, 4), "lift": round(off_lift, 4)},
                                advice=None if v_off == Verdict.PASS else
                                "High-pass or per-trial re-zero the signals and check that accuracy survives (see ablation)."))
    drop = full - no_offset
    v_abl = Verdict.WARN if drop >= cfg.gap_warn else Verdict.PASS
    sec.findings.append(Finding(
        "spectral", "offset ablation", v_abl,
        f"All features: {full:.2f}; without offset: {no_offset:.2f} (drop {drop:+.2f}).",
        {"all_features": round(full, 4), "without_offset": round(no_offset, 4), "drop": round(drop, 4)},
        advice=None if v_abl == Verdict.PASS else
        "The probe leans on the DC offset. Results may not survive re-calibration or a new session.",
    ))
    band_items = [(b, block_scores[b][0]) for b in block_names if b.startswith("band:")]
    if band_items:
        best = sorted(band_items, key=lambda t: -t[1])[:3]
        sec.findings.append(Finding(
            "spectral", "informative bands", Verdict.PASS,
            "Most informative bands: " + ", ".join(f"{b[5:]} ({s:.2f})" for b, s in best) + ".",
            {"per_block": {b: round(s, 4) for b, (s, _) in block_scores.items()}},
        ))
    sec.charts.append(Chart(
        title=f"Balanced accuracy per feature block (cv: {ctx.eval_cv})",
        labels=[b.replace("band:", "") for b in block_names],
        values=[block_scores[b][0] for b in block_names],
        reference=chance,
        highlight=["warn" if (b == "offset" and v_off != Verdict.PASS) else "" for b in block_names],
    ))
    return sec


# ---------------------------------------------------------------------- 7. replay


def _phase_resample(seq: np.ndarray, bins: int) -> np.ndarray:
    t = np.linspace(0, 1, len(seq))
    tt = np.linspace(0, 1, bins)
    return np.stack([np.interp(tt, t, seq[:, d]) for d in range(seq.shape[1])], axis=1)


def audit_replay(ctx: Context) -> Section:
    ds, cfg = ctx.ds, ctx.cfg
    sec = Section(
        name="replay",
        title="Open-loop replay baseline",
        question="Can a blind replay of the average demonstration explain the recorded actions?",
    )
    if ds.sequences is None:
        sec.findings.append(Finding("replay", "replay", Verdict.SKIP, "Needs per-episode action sequences (LeRobot adapter)."))
        return sec
    seqs = [np.asarray(s, dtype=float).reshape(len(s), -1) for s in ds.sequences]
    seqs = [s for s in seqs]
    allf = np.concatenate(seqs, axis=0)
    mu, sd = allf.mean(axis=0), allf.std(axis=0)
    sd[sd < 1e-9] = 1.0
    z = [(s - mu) / sd for s in seqs]
    phased = [_phase_resample(s, cfg.replay_bins) if len(s) >= 2 else None for s in z]
    ss_res = ss_tot = 0.0
    per_class = {}
    for c in np.unique(ctx.y):
        idx = [i for i in np.flatnonzero(ctx.y == c) if phased[i] is not None]
        if len(idx) < 2:
            continue
        stack = np.stack([phased[i] for i in idx])
        total = stack.sum(axis=0)
        r_c = t_c = 0.0
        for j, i in enumerate(idx):
            template = (total - stack[j]) / (len(idx) - 1)
            t_orig = np.linspace(0, 1, len(z[i]))
            pred = np.stack([np.interp(t_orig, np.linspace(0, 1, cfg.replay_bins), template[:, d])
                             for d in range(template.shape[1])], axis=1)
            r_c += float(np.sum((z[i] - pred) ** 2))
            t_c += float(np.sum(z[i] ** 2))
        per_class[str(ds.classes[c])] = round(1 - r_c / t_c, 4) if t_c > 0 else None
        ss_res += r_c
        ss_tot += t_c
    if ss_tot == 0:
        sec.findings.append(Finding("replay", "replay", Verdict.SKIP, "Need at least two episodes per task."))
        return sec
    r2 = 1 - ss_res / ss_tot
    verdict = Verdict.WARN if r2 >= cfg.replay_warn else Verdict.PASS
    sec.findings.append(Finding(
        "replay", "open-loop replay R²", verdict,
        f"Replaying the mean trajectory of the other demos of the same task (time-normalized, no observations) "
        f"explains {r2:.0%} of action variance in held-out episodes.",
        {"r2": round(r2, 4), "per_task_r2": per_class, "phase_bins": cfg.replay_bins},
        advice=None if verdict == Verdict.PASS else
        "Demonstrations are nearly identical. A policy can score well in-distribution by memorizing a "
        "trajectory and ignoring observations. Evaluate under perturbed initial conditions and add variation "
        "to the demos.",
    ))
    sec.charts.append(Chart(title="Replay R² per task", labels=list(per_class), values=[v or 0.0 for v in per_class.values()],
                            reference=cfg.replay_warn, reference_label="warn", value_label="R²"))
    return sec


AUDITS: Dict[str, Callable[[Context], Section]] = {
    "confound": audit_confound,
    "nuisance": audit_nuisance,
    "protocol": audit_protocol,
    "leakage": audit_leakage,
    "drift": audit_drift,
    "spectral": audit_spectral,
    "replay": audit_replay,
}
