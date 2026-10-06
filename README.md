# tiny-vec

**Linear algebra & autograd from first principles** — Project 01 of 12 (ZERO_TO_ANTHROPIC).

A `Vector`, a `Matrix` with naive and Strassen matmul, a reverse-mode autograd
engine that trains an MLP on XOR, and SVD-by-power-iteration + PCA — all in pure
Python. NumPy appears **only in the tests**, as the oracle I check my math against.

Stack: Python 3.10+, NumPy (tests only), pytest, hypothesis, GitHub Actions. Cost: $0.

---

## Build log — the story, day by day

Each day below is one commit. I describe what I built, why, the exact commands,
the real output I got, and what surprised me.

### Day 0 — set up the project

**Goal:** an empty, versioned skeleton so every later day is just "add a file, test it, commit".

1. Checked the interpreter: `python --version` → `Python 3.12.3` (anything ≥ 3.10 works).
2. Made a virtualenv and installed the dev tools:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install numpy pytest hypothesis pytest-cov scikit-learn matplotlib
   ```
3. Created the folder layout and the first commit:
   ```bash
   mkdir -p tiny_vec tests examples bench docs .github/workflows
   touch tiny_vec/__init__.py
   git init && git add -A && git commit -m 'chore: scaffold tiny-vec'
   ```

**Why this layout:** `tiny_vec/` is the library, `tests/` proves it, `examples/`
shows it working on real problems, `bench/` measures it, `docs/` holds plots.
Keeping those apart is what makes the repo readable to a reviewer in 30 seconds.

### Day 1 (Monday) — the `Vector` class

**Goal:** own the dot product. Every later operation (matmul, norms, neurons) is built from it.

**What I wrote — `tiny_vec/vector.py`:**
- A `Vector` with `__slots__ = ("data",)` holding a plain `List[float]`. `__slots__`
  stops Python creating a per-instance `__dict__`, which saves memory and makes typos
  like `v.dta = …` an error instead of a silent new attribute.
- Dunder plumbing (`__len__`, `__iter__`, `__getitem__`, `__repr__`, `__eq__`) so a
  Vector behaves like a native sequence.
- `_check()` raises `ValueError("dim mismatch: …")` — fail loudly rather than let `zip`
  silently truncate the longer vector (a classic bug).
- Algebra: `+`, `-`, scalar `*` (and `__rmul__` so `2 * v` works), unary `-`.
- `dot()` = `sum(a*b for a, b in zip(...))`.
- `norm(p)` handles L2 (via `sqrt(v·v)`), L1, and general Lp.
- `normalize()` refuses the zero vector with a `ZeroDivisionError` instead of returning NaNs.

**How I tested it — `tests/test_core.py`:**
- Three example tests that compare against NumPy: `dot` vs `np.dot`, `norm` vs
  `np.linalg.norm` (using `math.isclose`, never `==`, for floats), and add/sub.
- One **property-based** test with `hypothesis`: it generates hundreds of random float
  lists and asserts `a·b == b·a`. Example tests check cases I thought of; property tests
  check cases I didn't.

```text
$ pytest tests/test_core.py -q
....                                                                     [100%]
4 passed in 0.50s
```

**Commit:** `feat: Vector with dot/norm + property tests`
