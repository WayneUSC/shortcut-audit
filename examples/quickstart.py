"""Audit your own windows in ~20 lines.

Replace the synthetic arrays with yours:
  signals  (N, C, T) raw sensor windows
  labels   (N,)      what the model should predict
  session  (N,)      when / where each window was recorded
  obj      (N,)      which physical object it came from
  trial    (N,)      which recording it was cut from
  t        (N,)      recording order or timestamp
"""

import shortcut_audit as sa
from shortcut_audit.synthetic import make_dataset

demo = make_dataset("blocked_drift")  # stand-in for your data
signals, labels, fs = demo.signals, demo.y, demo.fs
session, obj, trial, t = (demo.meta[k] for k in ("session", "instance", "trial", "order"))

ds = sa.AuditDataset(
    y=labels,
    signals=signals,
    fs=fs,
    meta={"session": session, "object": obj, "trial": trial, "t": t},
    groups=["session", "object"],  # factors you want to generalize across
    unit="trial",                  # windows from one trial are not independent
    order="t",                     # recording order (or a timestamp)
    name="my-dataset",
)

report = sa.audit(
    ds,
    nested=["object"],       # each object belongs to one class by design
    baseline_s=0.2,          # first 0.2 s of every window precede contact
    cross=("session", "object"),
)

print(report.to_text())
report.to_html("audit.html")
report.to_json("audit.json")
