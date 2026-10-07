# skew-detector

**A CLI that catches the silent killer of production ML: data drift** — Project 05 of 12 (ZERO_TO_ANTHROPIC).

Point it at a training CSV and a serving CSV. It diffs the schema, runs two
distribution-drift tests per numeric column (Kolmogorov-Smirnov with Bonferroni
correction, and the Population Stability Index), tests label-prevalence shift
with chi-squared, renders a terminal report, writes HTML/JSON, and exits non-zero
on drift so it can gate CI.

Stack: Python 3.12, pandas, SciPy, NumPy, Click, Rich, pytest, matplotlib. Cost: $0.

---

## Build log — the story, day by day

### Day 0 — setup

```bash
mkdir skew-detector && cd skew-detector
python3 -m venv .venv && source .venv/bin/activate
pip install click rich pandas scipy numpy pytest matplotlib
mkdir -p skew_detect tests examples data docs && touch skew_detect/__init__.py
git init && git add -A && git commit -m 'chore: scaffold skew-detector'
```

**The problem, before any code.** A model is trained on one snapshot of the world
and then served on a world that keeps moving: a marketing campaign brings in
younger users, an upstream job starts reporting income in thousands, a policy change
raises the fraud rate. Uptime, latency and error-rate dashboards stay green, because
the model still answers; it just answers worse. That's **training-serving skew**.
The five kinds:

| kind | what changes | example | catchable from two CSVs? |
|---|---|---|---|
| schema drift | columns added / dropped / retyped | `income` renamed `income_usd` | **yes** |
| distribution drift | a column's values shift | mean age 38 → 45 | **yes** |
| label prevalence | positive-class rate changes | defaults 12% → 22% | **yes** |
| concept drift | the X→y relationship itself | last year's signals stop predicting | no (needs production labels) |
| freshness / lag | data arrives too late | stale feature snapshots | no (infra problem) |

### Day 1 (Monday) — the schema check (`skew_detect/schema.py`)

**Goal:** the cheapest, most important check first. If a column vanished or changed
type, the statistics don't matter yet.

`check_schema(train, serve)` compares the column sets and, for shared columns, the
pandas dtypes. Each finding is a `SchemaIssue(kind, column, detail)` with a severity:
a **dropped** column or a **type change** is `fail` (the model will break or silently
misread the feature); an **added** column is `warn` (harmless to the model, but someone
changed the pipeline).

A quick demo with a renamed and a retyped column:
```text
fail dropped      income     present in train, missing in serve
warn added        income_usd new column in serve, never seen in train
fail type_changed age        int64 -> float64
clean pair -> []
```
A rename shows up as one drop plus one add. That's the common real-world signature.

**Commit:** `feat: schema diff (add/drop/type-change)`

### Days 2–5 (Tuesday–Friday) — the drift checks (`skew_detect/drift.py`)

**Goal:** answer, per column, "do train and serve come from the same distribution?"
with statistics, not eyeballing.

**The framing every test shares.** Null hypothesis H0: *train and serve come from the
same distribution*. Compute a statistic; the p-value is how surprising a value this
extreme would be if H0 were true. Below the threshold α → reject H0 → DRIFT. Checking
only the mean is a beginner trap: two columns can share a mean and have completely
different shapes.

**1. Kolmogorov-Smirnov (`ks_drift`).** The CDF of a column is "what fraction of values
is ≤ x". KS's statistic **D** is the largest vertical gap between the train and serve
CDFs. It reacts to shifts in location, scale *and* shape. `scipy.stats.ks_2samp` returns
D and the p-value.

**2. Bonferroni correction (one line in `ks_drift`).** Run one test per column and
false alarms pile up. The fix is `alpha / number_of_columns`. I simulated it: 2000 runs
of 6 columns with **no drift at all**:
```text
no drift, 6 columns, 2000 runs: P(any false alarm) alpha=0.01 -> 0.053;  Bonferroni 0.01/6 -> 0.0075
```
Without correction, about 1 run in 19 cries wolf. With it, the family-wise false alarm
rate stays under the 1% I asked for. (Theory says 1 − 0.99⁶ ≈ 5.9% uncorrected; KS on
finite samples is slightly conservative, hence 5.3%.)

**3. Population Stability Index (`psi`, `psi_drift`).** The credit-risk industry
standard. Cut the *training* data into 10 quantile bins (so each holds 10% of train),
count where serve falls in those same bins, and sum
`(p_serve − p_train) · ln(p_serve / p_train)`. Rules of thumb: < 0.10 stable, 0.10–0.25
moderate, ≥ 0.25 retrain. Empty bins get ε = 1e-6 so `log(0)` can't blow up. Checking
the Part-A worked example (30% of serve lands in bin 0 instead of 10%):
```text
bin-0 term (0.30-0.10)*ln(3) = 0.22 | analytic PSI = 0.27 | psi() on samples = 0.258
```
The bin-0 term alone is 0.22, total PSI ≈ 0.27, "material shift", and my function on
100k random samples gives 0.258 (the small gap is sampling noise in the quantile edges).
The CLI flags DRIFT at PSI ≥ 0.10.

**4. Chi-squared for label prevalence (`label_drift`).** For the label, the question
isn't "did the shape change" but "did the *rate* change" (fraud 12% → 22%). Build a
2×K contingency table (train/serve × classes) and ask `chi2_contingency` whether rows
and columns are independent. Every cell gets +1 so a class missing on one side
can't produce a zero-expected-count error. It also reports both prevalences.

All three return a `DriftResult(column, statistic, p_value, alpha, drifted, extra)`
with `to_dict()`, so the CLI can render it and the JSON exporter can dump it without
recomputing.

**Commit:** `feat: KS + PSI + label-prevalence drift checks`
