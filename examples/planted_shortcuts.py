"""Run every planted-shortcut scenario and print a Markdown summary table.

    python examples/planted_shortcuts.py            # table only
    python examples/planted_shortcuts.py --html out # also write out/<scenario>.html
"""

import argparse
import os

import shortcut_audit as sa
from shortcut_audit.synthetic import SCENARIOS, make_benchmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="directory for HTML reports")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.html:
        os.makedirs(args.html, exist_ok=True)

    print("| scenario | verdict | flagged checks |")
    print("|---|---|---|")
    for name in SCENARIOS:
        ds, opts = make_benchmark(name, seed=args.seed)
        report = sa.audit(ds, **opts)
        flagged = [f"{f.verdict.value.upper()} {f.audit}/{f.check}" for f in report.findings
                   if f.verdict in (sa.Verdict.WARN, sa.Verdict.FAIL)]
        print(f"| `{name}` | **{report.verdict.value.upper()}** | {'<br>'.join(flagged) or '—'} |")
        if args.html:
            report.to_html(os.path.join(args.html, f"{name}.html"))


if __name__ == "__main__":
    main()
