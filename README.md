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
