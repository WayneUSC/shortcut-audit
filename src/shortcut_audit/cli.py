"""Command-line interface: ``shortcut-audit run | lerobot | demo``."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .config import AuditConfig
from .dataset import DatasetError, load
from .result import Report, Verdict
from .runner import audit

FAIL_ON = {"fail": Verdict.FAIL, "warn": Verdict.WARN, "never": None}


def _csv(value: Optional[str]) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _add_common(p: argparse.ArgumentParser) -> None:
    out = p.add_argument_group("output")
    out.add_argument("--html", metavar="PATH", help="write a self-contained HTML report")
    out.add_argument("--json", metavar="PATH", help="write the report as JSON")
    out.add_argument("--fail-on", choices=list(FAIL_ON), default="fail",
                     help="exit with code 1 at this severity or worse (default: fail)")
    out.add_argument("--quiet", action="store_true", help="only print the verdict line")
    out.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    st = p.add_argument_group("statistics")
    st.add_argument("--permutations", type=int, default=None,
                    help="label permutations per nuisance probe (default: enough for Holm at --alpha)")
    st.add_argument("--alpha", type=float, default=0.05)
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--probe", choices=["knn", "forest", "linear"], default="knn",
                    help="model for nuisance probes (default: knn)")
    st.add_argument("--only", help="comma-separated audits to run")
    st.add_argument("--skip", help="comma-separated audits to skip")


def _config(args: argparse.Namespace) -> AuditConfig:
    cfg = AuditConfig(permutations=args.permutations, alpha=args.alpha, seed=args.seed, probe=args.probe)
    for name in ("cv", "baseline_s"):
        if getattr(args, name, None) is not None:
            setattr(cfg, name, getattr(args, name))
    if getattr(args, "nested", None):
        cfg.nested = _csv(args.nested)
    if getattr(args, "cross", None):
        pair = _csv(args.cross)
        if len(pair) != 2:
            raise DatasetError("--cross takes exactly two columns, e.g. --cross session,instance")
        cfg.cross = (pair[0], pair[1])
    return cfg


def _emit(report: Report, args: argparse.Namespace) -> int:
    color = not args.no_color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    if args.quiet:
        c = report.counts()
        print(f"{report.dataset.get('name')}: {report.verdict.value.upper()} "
              f"({c['fail']} fail, {c['warn']} warn, {c['pass']} pass)")
    else:
        print(report.to_text(color=color))
    if args.html:
        report.to_html(args.html)
        print(f"\nHTML report: {args.html}")
    if args.json:
        report.to_json(args.json)
        print(f"JSON report: {args.json}")
    threshold = FAIL_ON[args.fail_on]
    if threshold is not None and report.verdict.rank >= threshold.rank:
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ds = load(
        args.data,
        label=args.label,
        groups=args.groups,
        order=args.order,
        unit=args.unit,
        nuisance=args.nuisance,
        split=args.split,
        **({"features": args.features} if args.features else {}),
        **({"fs": args.fs} if args.fs else {}),
    )
    rep = audit(ds, _config(args), only=_csv(args.only) or None, skip=_csv(args.skip))
    return _emit(rep, args)


def cmd_lerobot(args: argparse.Namespace) -> int:
    from .lerobot import load_lerobot

    ds = load_lerobot(
        args.dataset,
        target=args.target,
        state_key=args.state_key,
        action_key=args.action_key,
        sensor_keys=_csv(args.sensor_keys) or None,
        baseline_frames=args.baseline_frames,
        episode_meta=args.episode_meta,
        groups=_csv(args.groups),
        nuisance=_csv(args.nuisance),
        revision=args.revision,
    )
    skip = set(_csv(args.skip)) | {"spectral", "protocol"}
    rep = audit(ds, _config(args), only=_csv(args.only) or None, skip=skip)
    return _emit(rep, args)


def cmd_demo(args: argparse.Namespace) -> int:
    from .synthetic import SCENARIOS, make_benchmark

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    worst = 0
    for name in names:
        ds, opts = make_benchmark(name, seed=args.seed)
        cfg = _config(args)
        for k, v in opts.items():
            setattr(cfg, k, v)
        rep = audit(ds, cfg, only=_csv(args.only) or None, skip=_csv(args.skip))
        if len(names) > 1:
            print(f"\n{'=' * 78}\n{name}: {SCENARIOS[name]['description']}\n{'=' * 78}")
        html = args.html
        if html and len(names) > 1:
            root, ext = os.path.splitext(html)
            html = f"{root}-{name}{ext or '.html'}"
        sub = argparse.Namespace(**{**vars(args), "html": html,
                                    "json": None if len(names) > 1 else args.json})
        worst = max(worst, _emit(rep, sub))
    return worst


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shortcut-audit",
        description="Can a model cheat on your dataset? Shortcut and leakage audits for sensor and robot datasets.",
    )
    p.add_argument("--version", action="version", version=f"shortcut-audit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="audit a .npz / .csv / .tsv / .parquet dataset",
                       description="Audit a table (one row per sample) or an .npz with signals (N,C,T) / features (N,F).")
    r.add_argument("data", help="path to .npz, .csv, .tsv or .parquet")
    r.add_argument("--label", required=True, help="label column / array name")
    r.add_argument("--groups", help="grouping factors to generalize across, e.g. session,object,subject")
    r.add_argument("--nested", help="groups nested in the label by design (object instance, subject)")
    r.add_argument("--order", help="recording order or timestamp column")
    r.add_argument("--unit", help="recording/trial id when samples are windows cut from longer recordings")
    r.add_argument("--nuisance", help="columns that should carry no label information")
    r.add_argument("--split", help="train/test split column for the leakage audit")
    r.add_argument("--features", help="feature columns (default: all other numeric columns)")
    r.add_argument("--fs", type=float, help="sampling rate in Hz for signals (if not stored in the .npz)")
    r.add_argument("--baseline-s", type=float, dest="baseline_s",
                   help="seconds at the start of each window that precede contact (pre-contact baseline)")
    r.add_argument("--cross", help="two groups for joint leave-out, e.g. session,instance")
    r.add_argument("--cv", help="cross-validation for probes: auto | random | unit | group:<column>")
    _add_common(r)
    r.set_defaults(func=cmd_run)

    lr = sub.add_parser("lerobot", help="audit a LeRobot dataset (local folder or Hub repo id)",
                        description="Episode-level audit of a LeRobot v2.1/v3.0 dataset. Reads parquet only, never videos.")
    lr.add_argument("dataset", help="local dataset root, or Hub repo id (user/name or hf://user/name)")
    lr.add_argument("--target", default="task", help="label: task (default), task_index or a column name")
    lr.add_argument("--state-key", default="observation.state")
    lr.add_argument("--action-key", default="action")
    lr.add_argument("--sensor-keys", help="numeric observation keys to treat as sensors (default: all others)")
    lr.add_argument("--baseline-frames", type=int, default=5)
    lr.add_argument("--episode-meta", help="CSV with episode_index + per-episode metadata (session, operator, ...)")
    lr.add_argument("--groups", help="episode-meta columns that are grouping factors")
    lr.add_argument("--nuisance", help="episode-meta columns that should carry no label information")
    lr.add_argument("--revision", help="Hub revision (branch, tag or commit)")
    _add_common(lr)
    lr.set_defaults(func=cmd_lerobot)

    d = sub.add_parser("demo", help="audit synthetic datasets with planted shortcuts",
                       description="Run the audit on synthetic shaking-box datasets where the shortcut is known.")
    d.add_argument("--scenario", default="blocked_drift",
                   choices=["all", "clean", "blocked_drift", "instance_memorization", "window_leakage"])
    _add_common(d)
    d.set_defaults(func=cmd_demo, fail_on="never")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DatasetError, ValueError, ImportError, FileNotFoundError) as exc:
        print(f"shortcut-audit: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
