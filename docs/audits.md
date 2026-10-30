# How the audits work and how to read them

Every audit answers one question and returns findings with a verdict:

- **PASS** — the probe found nothing at the configured threshold.
- **WARN** — a risk worth a closer look; often fine if you know why.
- **FAIL** — information a model could exploit is clearly present, or the design cannot separate the label from
  how the data was recorded.
- **SKIP** — the input needed for this check was not provided.

A flag is a reason to look, not a verdict on your model. A pass is evidence, not proof.

## Vocabulary

| Term | Meaning |
|---|---|
| sample | the thing your model classifies: a window, a trial, an episode |
| unit | the recording a sample was cut from (`--unit`). Samples sharing a unit are not independent |
| group | a factor you want to generalize across: session, day, operator, object, subject (`--groups`) |
| nested group | a group whose values each belong to a single class by design, e.g. object instance (`--nested`) |
| order | recording order or timestamp (`--order`) |
| balanced accuracy | mean per-class recall; chance is 1/K regardless of class imbalance |
| lift | (balanced accuracy − chance) / (1 − chance): 0 at chance, 1 when perfect |

## Cross-validation used by the probes

`--cv auto` (default) groups folds by `unit` when one is declared, so windows from one trial never sit on both sides
of a fold; otherwise it uses stratified 5-fold. Grouped folds are built **without looking at labels**
(leave-one-group-out up to 12 groups, otherwise groups dealt into 5 folds with a fixed seed). That keeps folds
identical across label permutations.

## confound — Is the label entangled with how or when the data was recorded?

For every group: **bias-corrected Cramér's V** (Bergsma 2013) between label and group, with the χ² p-value and the
contingency table in the JSON report.

| Situation | Verdict |
|---|---|
| declared nested (`--nested object`) | PASS (generalization is tested by the protocol audit) |
| every value holds one class and #values ≤ #classes | FAIL: label and group are aliased one-to-one |
| every value holds one class, more values than classes | WARN: looks like an identity factor — declare it nested if so |
| V ≥ 0.50 | FAIL |
| V ≥ 0.20 | WARN |

**Recording-schedule blockiness.** Sort recordings by `order` (one entry per unit), measure how often consecutive
recordings share a label, compare with the rate under a random schedule (Σ nₖ(nₖ−1) / N(N−1)), and rescale so a
random schedule is 0 and one uninterrupted run per class is 1. p-value from 2,000 schedule permutations. WARN when
blockiness ≥ 0.30 and p < α. A blocked schedule is not a shortcut by itself, but it turns every slow change in the
rig (drift, wear, temperature, battery) into a label proxy.

## nuisance — Can a model that sees only irrelevant information beat chance?

Each probe sees one block of information that should carry no label signal:

- recording order (`--order`)
- the id of each crossed (non-nested) group
- each `--nuisance` column
- the pre-contact baseline: per-channel mean and std of the first `--baseline-s` seconds
- adapter-specific blocks (LeRobot: initial robot state, resting level of every non-visual sensor)
- all of the above together

The probe is k-NN (k = 7) on standardized features, scored with cross-validated balanced accuracy. Significance
comes from a **label permutation test**:

- When samples come in units with a constant label, labels are permuted **between units**. Permuting windows of
  the same trial independently treats dependent samples as independent and gives p-values that are far too small.
- The number of permutations defaults to max(99, 2m/α − 1) for m probes so that the Holm-corrected p-value can
  actually fall below α.
- Besag & Clifford (1991) sequential stopping: once 10 permuted scores reach the observed score, the probe is
  clearly null and permutation stops early (p = 10 / l).
- **Holm** correction across all probes of the audit.

| Situation | Verdict |
|---|---|
| Holm p < α and lift ≥ 0.30 | FAIL |
| Holm p < α and lift ≥ 0.10 | WARN |
| otherwise | PASS |

`--probe forest` switches to extremely randomized trees (slower, more robust when a block has many irrelevant
dimensions).

## protocol — How much does accuracy depend on having seen the same session / object / recording?

Two deliberately simple reference probes are trained on your features (or on per-channel offset, std and octave
band power when you pass raw signals):

- **linear**: standardized logistic regression — generalizable structure;
- **k-NN**: a memorizer — the most sensitive to near-copies and to recognizing *which* object or session a sample
  came from.

Each is evaluated under a random stratified split (the reference), leave-unit-out, leave-*group*-out for every
group, and optionally a **joint leave-A-and-B-out** split (`--cross session,object`): for every (a, b) cell, train
on samples whose A ≠ a **and** B ≠ b, test on the cell. Neither the session nor the object of a test sample was
seen in training.

The verdict uses the largest drop versus the random split over the two probes: ≥ 0.20 FAIL, ≥ 0.10 WARN. Absolute
numbers will be lower than a tuned model; the gap is the point. A large gap means random-split numbers overstate
generalization to new sessions/objects — report the grouped number as your headline.

## leakage — Does the test set contain recordings, groups or near-copies seen in training?

- **Exact duplicates** (no split needed): identical signals or feature rows (rounded to 1e-9); FAIL when a
  duplicate set carries conflicting labels.
- **Unit overlap** (`--split` + `--unit`): any recording on both sides of the split is FAIL.
- **Group overlap**: WARN — fine for an in-distribution claim, not for a generalization claim.
- **Near-duplicates**: features standardized on the training set; for each test sample, distance to its nearest
  training sample; reference distribution = distance from each training sample to its nearest training sample from
  a *different unit*. The rate of test samples closer than the 1st percentile of the reference should be ≈ 1%.
  ≥ 3× WARN, ≥ 10× FAIL.

## drift — Does the sensor's resting level move between sessions or over time?

Resting level = mean of the first `--baseline-s` seconds (or the window median, or adapter-provided levels).

- Across each crossed group: Kruskal–Wallis H test with effect size ε² = (H − k + 1)/(n − k) per channel; the
  worst channel is reported with a Bonferroni-corrected p. ε² ≥ 0.26 (large) is WARN, and FAIL when the same group
  is also label-imbalanced (Cramér's V ≥ 0.20) — that is drift a model can use.
- Over `order`: Spearman ρ per channel, Bonferroni-corrected; |ρ| ≥ 0.5 is WARN.

## spectral — Which frequency bands carry the label, and is it only the DC offset?

Per-channel features from a Hann-windowed periodogram after removing the window mean: offset (mean), spread (std),
log power in octave bands up to Nyquist. A linear probe is scored on each block.

- **offset only**: lift ≥ 0.30 from the per-channel mean alone is WARN. It is legitimate when static load is the
  signal (pressure while grasping); otherwise it usually means baseline drift or calibration.
- **offset ablation**: a drop ≥ 0.10 when the offset block is removed is WARN.
- **informative bands**: the top three blocks, for orientation.

## replay — Can a blind open-loop replay explain the recorded actions?

For sequence datasets (LeRobot actions). Actions are standardized per dimension; every episode is time-normalized
to 50 phase bins. For each held-out episode, the template is the mean of the *other* episodes of the same task,
interpolated back to its length. R² is computed over all held-out frames. R² ≥ 0.90 is WARN: demonstrations are so
similar that a policy can succeed in distribution by replaying a trajectory, which says little about closed-loop
competence. Evaluate under perturbed initial conditions.

## Thresholds

All thresholds live in `shortcut_audit.AuditConfig` and can be passed to `audit()` as keyword arguments:

```python
sa.audit(ds, fail_lift=0.4, gap_warn=0.05, confound_fail=0.6, permutations=499, alpha=0.01)
```
