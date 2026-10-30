"""Render a :class:`Report` as plain text or as a single self-contained HTML page."""

from __future__ import annotations

import html
from typing import List

from .result import Chart, Report, Verdict

# ---------------------------------------------------------------------- text

_ANSI = {"pass": "\033[32m", "warn": "\033[33m", "fail": "\033[31m", "skip": "\033[90m"}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _dataset_line(report: Report) -> str:
    d = report.dataset
    parts = [f"{d.get('n_samples')} samples", f"{d.get('n_classes')} classes"]
    if d.get("groups"):
        parts.append("groups: " + ", ".join(f"{g}({n})" for g, n in d["groups"].items()))
    if d.get("unit"):
        parts.append(f"unit: {d['unit']} ({d.get('n_units', '?')})")
    if d.get("order"):
        parts.append(f"order: {d['order']}")
    if d.get("signals"):
        s = d["signals"]
        parts.append(f"signals: {s['channels']}ch × {s['window_s']:g}s @ {s['fs_hz']:g} Hz")
    parts.append(f"cv: {d.get('evaluation_cv')}")
    return " · ".join(parts)


def render_text(report: Report, color: bool = False) -> str:
    def c(v: Verdict, text: str) -> str:
        return f"{_ANSI[v.value]}{text}{_RESET}" if color else text

    def b(text: str) -> str:
        return f"{_BOLD}{text}{_RESET}" if color else text

    counts = report.counts()
    lines: List[str] = []
    lines.append(b(f"shortcut-audit {report.version} — {report.dataset.get('name')}"))
    lines.append(_dataset_line(report))
    tally = " · ".join(f"{counts[k]} {k}" for k in ("fail", "warn", "pass", "skip") if counts[k])
    lines.append(b("VERDICT: ") + c(report.verdict, report.verdict.value.upper()) + f"   ({tally})")
    for sec in report.sections:
        lines.append("")
        lines.append(b(f"{sec.title}") + f" — {sec.question}")
        for f in sec.findings:
            lines.append(f"  {c(f.verdict, f.verdict.symbol)} {f.check:<30} {f.summary}")
            if f.advice and f.verdict in (Verdict.WARN, Verdict.FAIL):
                lines.append(f"      → {f.advice}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- html

_CSS = """
:root{--bg:#f7f7f5;--card:#ffffff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--accent:#3b5bdb;--accent2:#9fb0f0;
--pass:#2b8a3e;--warn:#c77700;--fail:#d6336c;--skip:#9a9a94;--passbg:#ebf7ee;--warnbg:#fff4e0;--failbg:#fdecf2;--skipbg:#f1f1ef}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--muted:#a3a29b;
--line:#33332f;--accent:#7b93f0;--accent2:#3f4f8f;--pass:#58c27a;--warn:#f0a73c;--fail:#f06595;--skip:#77776f;
--passbg:#1d2e22;--warnbg:#33281a;--failbg:#351c27;--skipbg:#262624}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
header{margin-bottom:24px}.tool{font-size:13px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}
h1{font-size:28px;margin:4px 0 8px;font-weight:650;word-break:break-word}.facts{color:var(--muted);font-size:14px}
.verdict{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0 0}
.badge{font-weight:700;font-size:14px;padding:6px 14px;border-radius:999px;letter-spacing:.05em}
.badge.pass{background:var(--passbg);color:var(--pass)}.badge.warn{background:var(--warnbg);color:var(--warn)}
.badge.fail{background:var(--failbg);color:var(--fail)}.badge.skip{background:var(--skipbg);color:var(--skip)}
.chip{font-size:13px;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:16px 0}
.card h2{font-size:18px;margin:0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.q{color:var(--muted);margin:4px 0 12px;font-size:14px}
.top li{margin:6px 0}.top .adv{color:var(--muted);font-size:14px}
table{width:100%;border-collapse:collapse;font-size:14px}td{padding:8px 6px;border-top:1px solid var(--line);vertical-align:top}
td.v{width:26px;text-align:center;font-weight:700}td.k{width:200px;font-weight:600;word-break:break-word}
.v.pass{color:var(--pass)}.v.warn{color:var(--warn)}.v.fail{color:var(--fail)}.v.skip{color:var(--skip)}
.adv{margin-top:4px;color:var(--muted)}.adv:before{content:"→ "}
.pill{font-size:12px;padding:2px 10px;border-radius:999px;font-weight:700;letter-spacing:.04em}
.pill.pass{background:var(--passbg);color:var(--pass)}.pill.warn{background:var(--warnbg);color:var(--warn)}
.pill.fail{background:var(--failbg);color:var(--fail)}.pill.skip{background:var(--skipbg);color:var(--skip)}
.chart{margin-top:14px;overflow-x:auto}.chart svg{max-width:100%;height:auto}.ctitle{font-size:13px;color:var(--muted);margin-bottom:4px}
.notes{font-size:13px;color:var(--muted);margin-top:10px}
details{margin-top:6px}summary{cursor:pointer;color:var(--muted);font-size:13px}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:var(--bg);padding:10px;border-radius:8px}
footer{color:var(--muted);font-size:13px;margin-top:32px}
a{color:var(--accent)}
@media (max-width:640px){td.k{width:110px}h1{font-size:22px}.card{padding:16px}}
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _svg_chart(chart: Chart) -> str:
    series = [(chart.series_name, chart.values)] + list(chart.extra.items())
    n_series = len(series)
    label_w, right_pad, width = 200, 56, 680
    bar_h = 14 if n_series == 1 else 10
    gap = 3
    row_h = n_series * bar_h + (n_series - 1) * gap + 12
    top = 22 if n_series > 1 else 8
    height = top + row_h * len(chart.labels) + 26
    x0, x1 = label_w, width - right_pad
    span = (chart.vmax - chart.vmin) or 1.0

    def X(v: float) -> float:
        v = min(max(v, chart.vmin), chart.vmax)
        return x0 + (v - chart.vmin) / span * (x1 - x0)

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(chart.title)}" xmlns="http://www.w3.org/2000/svg" font-size="12">']
    if n_series > 1:
        lx = x0
        for si, (name, _) in enumerate(series):
            color = "var(--accent)" if si == 0 else "var(--accent2)"
            out.append(f'<rect x="{lx}" y="4" width="10" height="10" rx="2" fill="{color}"/>')
            out.append(f'<text x="{lx + 14}" y="13" fill="var(--muted)">{_e(name)}</text>')
            lx += 24 + 7 * len(name)
    for t in range(5):
        v = chart.vmin + span * t / 4
        x = X(v)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - 22}" stroke="var(--line)"/>')
        out.append(f'<text x="{x:.1f}" y="{height - 8}" fill="var(--muted)" text-anchor="middle">{v:g}</text>')
    for i, label in enumerate(chart.labels):
        y = top + i * row_h + 6
        lab = label if len(label) <= 30 else label[:29] + "…"
        mid = y + (n_series * bar_h + (n_series - 1) * gap) / 2 + 4
        out.append(f'<text x="{x0 - 8}" y="{mid:.1f}" fill="var(--ink)" text-anchor="end">{_e(lab)}</text>')
        hl = chart.highlight[i] if i < len(chart.highlight) else ""
        if hl is True:
            hl = "fail"
        hl_color = {"fail": "var(--fail)", "warn": "var(--warn)"}.get(hl or "", "")
        for si, (_, vals) in enumerate(series):
            v = vals[i]
            by = y + si * (bar_h + gap)
            if v is None or v != v:  # NaN
                out.append(f'<text x="{x0 + 4}" y="{by + bar_h - 3}" fill="var(--muted)">n/a</text>')
                continue
            color = (hl_color or "var(--accent)") if si == 0 else ("var(--warn)" if hl_color else "var(--accent2)")
            w = max(1.0, X(v) - x0)
            out.append(f'<rect x="{x0}" y="{by}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
            out.append(f'<text x="{x0 + w + 5:.1f}" y="{by + bar_h - 3}" fill="var(--muted)">{v:.2f}</text>')
    if chart.reference is not None:
        x = X(chart.reference)
        out.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{height - 22}" stroke="var(--ink)" stroke-dasharray="4 3" opacity=".7"/>')
        out.append(f'<text x="{x + 4:.1f}" y="{top + 4 if n_series == 1 else top - 6}" fill="var(--ink)" opacity=".8">{_e(chart.reference_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def render_html(report: Report) -> str:
    counts = report.counts()
    v = report.verdict.value
    title = f"Shortcut audit — {report.dataset.get('name')}"
    parts: List[str] = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><main>",
        "<header>",
        '<div class="tool">shortcut-audit · can a model cheat on your dataset?</div>',
        f"<h1>{_e(report.dataset.get('name'))}</h1>",
        f'<div class="facts">{_e(_dataset_line(report))}</div>',
        f'<div class="verdict"><span class="badge {v}">{v.upper()}</span>',
    ]
    for k in ("fail", "warn", "pass", "skip"):
        if counts[k]:
            parts.append(f'<span class="chip">{counts[k]} {k}</span>')
    parts.append("</div></header>")

    flagged = [f for f in report.findings if f.verdict in (Verdict.FAIL, Verdict.WARN)]
    flagged.sort(key=lambda f: -f.verdict.rank)
    if flagged:
        parts.append('<section class="card top"><h2>What to fix first</h2><ul>')
        for f in flagged[:8]:
            parts.append(
                f'<li><span class="pill {f.verdict.value}">{f.verdict.value.upper()}</span> '
                f"<strong>{_e(f.check)}</strong> — {_e(f.summary)}"
                + (f'<div class="adv">{_e(f.advice)}</div>' if f.advice else "")
                + "</li>"
            )
        parts.append("</ul></section>")
    else:
        parts.append('<section class="card top"><h2>No shortcut found</h2><p class="q">'
                     "None of the audits that could run found a problem. That is evidence, not proof: "
                     "the audits only probe the metadata and features you provided.</p></section>")

    for sec in report.sections:
        sv = sec.verdict.value
        parts.append(f'<section class="card" id="{_e(sec.name)}">')
        parts.append(f'<h2>{_e(sec.title)} <span class="pill {sv}">{sv.upper()}</span></h2>')
        parts.append(f'<div class="q">{_e(sec.question)}</div><table>')
        for f in sec.findings:
            parts.append(
                f'<tr><td class="v {f.verdict.value}">{_e(f.verdict.symbol)}</td>'
                f'<td class="k">{_e(f.check)}</td><td>{_e(f.summary)}'
                + (f'<div class="adv">{_e(f.advice)}</div>' if f.advice else "")
                + "</td></tr>"
            )
        parts.append("</table>")
        for chart in sec.charts:
            parts.append(f'<div class="chart"><div class="ctitle">{_e(chart.title)}</div>{_svg_chart(chart)}</div>')
        for note in sec.notes:
            parts.append(f'<div class="notes">{_e(note)}</div>')
        parts.append("</section>")

    import json

    parts.append(
        "<footer>"
        f"Generated by shortcut-audit {_e(report.version)} on {_e(report.created)}. "
        'Audits flag <em>risks</em> that a model could exploit; a flag is a reason to look closer, '
        "a pass is not a proof of absence. "
        '<a href="https://github.com/WayneUSC/shortcut-audit/blob/main/docs/audits.md">How to read this report</a>.'
        "<details><summary>Configuration</summary><pre>"
        + _e(json.dumps(report.to_dict()["config"], indent=2))
        + "</pre></details></footer></main></body></html>"
    )
    return "".join(parts)
