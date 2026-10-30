"""Result containers: verdicts, findings, sections and the full report."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


class Verdict(str, enum.Enum):
    """Outcome of a single check. Ordered by severity: SKIP < PASS < WARN < FAIL."""

    SKIP = "skip"
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

    @property
    def rank(self) -> int:
        return {"skip": 0, "pass": 1, "warn": 2, "fail": 3}[self.value]

    @property
    def symbol(self) -> str:
        return {"skip": "·", "pass": "✓", "warn": "!", "fail": "✗"}[self.value]

    @staticmethod
    def worst(verdicts: Iterable["Verdict"]) -> "Verdict":
        verdicts = list(verdicts)
        if not verdicts:
            return Verdict.SKIP
        return max(verdicts, key=lambda v: v.rank)


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays (recursively) into plain Python types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, Verdict):
        return value.value
    return value


@dataclass
class Finding:
    """One check inside an audit, e.g. 'can recording order predict the label?'."""

    audit: str
    check: str
    verdict: Verdict
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    advice: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit": self.audit,
            "check": self.check,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "metrics": _jsonable(self.metrics),
            "advice": self.advice,
        }


@dataclass
class Chart:
    """Data for a (grouped) horizontal bar chart rendered in the HTML report.

    ``values`` is the primary series; ``extra`` holds additional named series drawn
    next to it (e.g. a second probe model).
    """

    title: str
    labels: List[str]
    values: List[float]
    reference: Optional[float] = None  # e.g. chance level, drawn as a dashed line
    reference_label: str = "chance"
    value_label: str = "balanced accuracy"
    vmin: float = 0.0
    vmax: float = 1.0
    highlight: List[str] = field(default_factory=list)  # per bar: "fail", "warn" or ""
    series_name: str = ""
    extra: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass
class Section:
    """All findings produced by one audit."""

    name: str
    title: str
    question: str
    findings: List[Finding] = field(default_factory=list)
    charts: List[Chart] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        return Verdict.worst(f.verdict for f in self.findings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "question": self.question,
            "verdict": self.verdict.value,
            "findings": [f.to_dict() for f in self.findings],
            "charts": [c.to_dict() for c in self.charts],
            "notes": list(self.notes),
        }


@dataclass
class Report:
    """The full audit report."""

    dataset: Dict[str, Any]
    sections: List[Section]
    config: Dict[str, Any]
    version: str
    created: str

    @property
    def verdict(self) -> Verdict:
        return Verdict.worst(s.verdict for s in self.sections)

    @property
    def findings(self) -> List[Finding]:
        return [f for s in self.sections for f in s.findings]

    def section(self, name: str) -> Section:
        for s in self.sections:
            if s.name == name:
                return s
        raise KeyError(name)

    def counts(self) -> Dict[str, int]:
        out = {v.value: 0 for v in Verdict}
        for f in self.findings:
            out[f.verdict.value] += 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": "shortcut-audit",
            "version": self.version,
            "created": self.created,
            "verdict": self.verdict.value,
            "counts": self.counts(),
            "dataset": _jsonable(self.dataset),
            "config": _jsonable(self.config),
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_text(self, color: bool = False) -> str:
        from .report import render_text

        return render_text(self, color=color)

    def to_html(self, path: Optional[str] = None) -> str:
        from .report import render_html

        html = render_html(self)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.to_text()
