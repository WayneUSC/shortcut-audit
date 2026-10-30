"""Run a set of audits on a dataset and collect a :class:`Report`."""

from __future__ import annotations

import datetime as _dt
from typing import Iterable, Optional

from .audits import AUDITS, build_context
from .config import AuditConfig
from .dataset import AuditDataset
from .result import Report

DEFAULT_ORDER = ["confound", "nuisance", "protocol", "leakage", "drift", "spectral", "replay"]


def audit(
    dataset: AuditDataset,
    config: Optional[AuditConfig] = None,
    only: Optional[Iterable[str]] = None,
    skip: Optional[Iterable[str]] = None,
    **overrides,
) -> Report:
    """Audit ``dataset`` and return a :class:`Report`.

    ``config`` fields can also be passed directly as keyword arguments, e.g.
    ``audit(ds, baseline_s=0.2, nested=["instance"])``.
    """
    from . import __version__

    cfg = config or AuditConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown option {key!r}")
        setattr(cfg, key, value)
    if cfg.cross is not None:
        cfg.cross = tuple(cfg.cross)
        for col in cfg.cross:
            if col not in dataset.meta:
                raise ValueError(f"cross={cfg.cross}: column {col!r} not in dataset meta")
    for col in cfg.nested:
        if col not in dataset.groups:
            raise ValueError(f"nested column {col!r} is not one of the dataset groups {dataset.groups}")

    names = list(only) if only else list(DEFAULT_ORDER)
    unknown = [n for n in names if n not in AUDITS]
    if unknown:
        raise ValueError(f"unknown audits {unknown}; choose from {list(AUDITS)}")
    skip_set = set(skip or [])
    ctx = build_context(dataset, cfg)
    sections = [AUDITS[n](ctx) for n in names if n not in skip_set]
    if dataset.sequences is None:
        sections = [s for s in sections if s.name != "replay"]
    info = dataset.summary()
    info["evaluation_cv"] = ctx.eval_cv
    info["nested_groups"] = [g for g, nested in ctx.nested.items() if nested]
    return Report(
        dataset=info,
        sections=sections,
        config=cfg.to_dict(),
        version=__version__,
        created=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
