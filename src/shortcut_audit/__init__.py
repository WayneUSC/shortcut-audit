"""shortcut-audit: can a model cheat on your dataset?

Quick start::

    import shortcut_audit as sa

    ds = sa.AuditDataset(y=labels, signals=windows, fs=1000,
                         meta={"session": session, "object": obj, "trial": trial, "t": t},
                         groups=["session", "object"], unit="trial", order="t")
    report = sa.audit(ds, nested=["object"], baseline_s=0.2)
    print(report.to_text())
    report.to_html("audit.html")
"""

from .config import AuditConfig
from .dataset import AuditDataset, DatasetError, from_dataframe, from_npz, load
from .result import Finding, Report, Section, Verdict
from .runner import audit

__version__ = "0.1.0"

__all__ = [
    "AuditConfig",
    "AuditDataset",
    "DatasetError",
    "Finding",
    "Report",
    "Section",
    "Verdict",
    "audit",
    "from_dataframe",
    "from_npz",
    "load",
    "__version__",
]
