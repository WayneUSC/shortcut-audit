"""Audit configuration and decision thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class AuditConfig:
    """Settings shared by all audits.

    Thresholds are expressed on interpretable scales:

    * **lift** rescales balanced accuracy so chance = 0 and perfect = 1.
    * **gap** is the drop in balanced accuracy when moving from a random split to a
      grouped split.
    * **Cramér's V** (bias-corrected) measures label/group association, 0..1.
    * **ε²** is the Kruskal–Wallis effect size (0.26 ≈ large).
    """

    # How the probes are cross-validated. "auto" = group by `unit` if the dataset has
    # one (windows of the same recording never straddle folds), else stratified k-fold.
    # Also accepts "random", "unit" or "group:<column>".
    cv: str = "auto"
    n_splits: int = 5
    # Label permutations per nuisance probe. None = enough for Holm correction to be
    # able to reach `alpha`: max(99, 2 * n_probes / alpha - 1).
    permutations: Optional[int] = None
    alpha: float = 0.05
    seed: int = 0

    # Nuisance probes
    probe: str = "knn"
    warn_lift: float = 0.10
    fail_lift: float = 0.30

    # Protocol gap (random split vs grouped split). Two reference probes: a linear probe
    # (generalizable structure) and k-NN (a memorizer, most sensitive to near-copies).
    reference_probes: Tuple[str, ...] = ("linear", "knn")
    # Probe used for the spectral band sweep.
    reference_probe: str = "linear"
    gap_warn: float = 0.10
    gap_fail: float = 0.20
    cross: Optional[Tuple[str, str]] = None

    # Confounding
    nested: List[str] = field(default_factory=list)
    confound_warn: float = 0.20
    confound_fail: float = 0.50
    blockiness_warn: float = 0.30

    # Drift
    baseline_s: Optional[float] = None
    drift_warn: float = 0.26
    trend_warn: float = 0.50

    # Spectral
    bands: Optional[Sequence[Tuple[float, float]]] = None

    # Leakage
    neardup_quantile: float = 0.01
    neardup_warn: float = 3.0
    neardup_fail: float = 10.0
    max_neighbors_samples: int = 20000

    # Open-loop replay (sequence datasets, e.g. LeRobot actions)
    replay_bins: int = 50
    replay_warn: float = 0.90

    def to_dict(self) -> dict:
        return asdict(self)
