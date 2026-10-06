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

### Day 2 (Tuesday) — the `Matrix` class + naive matmul

**Goal:** an honest, slow baseline. You cannot claim a speedup tomorrow without a
correct slow version today.

**What I wrote — `tiny_vec/matrix.py`:**
- `Matrix` stores `List[List[float]]` plus `rows`/`cols`, and rejects ragged input
  (`ValueError("ragged matrix…")`) in the constructor so a bad shape never gets far.
- Constructors `identity(n)` and `zeros(r, c)`; plumbing (`__getitem__`, `__eq__`,
  `__repr__`, `to_list`), `transpose()` and the `.T` property, elementwise `+`/`-`.
- **Naive matmul**: the triple loop. One real trick: I transpose `other` once
  (`bt = other.transpose().data`) so the inner loop zips *row with row* instead of
  striding down a column — fewer index lookups, same O(n³). Hooked up to `@` via
  `__matmul__ = matmul`.
- The same file already contains the Strassen code (`strassen`, `_pad`, `_add`,
  `_sub`, `_naive`, `_strassen`); I study and benchmark it on Day 3.

**Benchmark vs NumPy (n = 128):**
```text
# first run, NumPy cold (includes BLAS warm-up):
tiny-vec 0.099s  numpy 0.00043s  -> numpy 231x faster
# after a NumPy warm-up call, three runs:
tiny-vec 0.097s  numpy 0.00014s  -> numpy 707x faster
tiny-vec 0.106s  numpy 0.00014s  -> numpy 751x faster
tiny-vec 0.103s  numpy 0.00014s  -> numpy 746x faster
```
Lesson #1 of benchmarking: the first call is not representative — warm up, then
repeat. **NumPy is ~700x faster** on this machine. Why: it calls an optimized BLAS
written in C/assembly (SIMD, cache blocking, multithreading) over contiguous memory,
while my loop pays Python interpreter overhead on every single multiply-add
(boxing floats, dispatching `*` and `+`, iterating lists of pointers).

**Tests added:** matmul vs `np.array(A) @ np.array(B)` with `np.allclose`, transpose
vs `.T`, identity vs `np.eye`.
```text
$ pytest tests/test_core.py -q
.......                                                                  [100%]
7 passed in 0.45s
```

**Commit:** `feat: Matrix + naive matmul, benchmarked vs NumPy`

### Day 3 (Wednesday) — Strassen + an honest benchmark

**Goal:** test a textbook claim against reality. Strassen does **7** recursive
half-size multiplies instead of 8, so its complexity drops from O(n³) to
O(n^log₂7) ≈ O(n^2.807). The textbook says it should win early. Does it, in Python?

**How Strassen works (re-reading the Day-2 code):**
1. `strassen()` pads both matrices with zeros up to the next power of two (`_pad`),
   because the algorithm splits in halves recursively.
2. `_strassen()` splits A and B into four quadrants each (`a11…a22`, `b11…b22`).
3. It forms the seven products `m1…m7` (each one a recursive call on sums/differences
   of quadrants) and rebuilds the four output quadrants from them:
   `c11 = m1 + m4 − m5 + m7`, `c12 = m3 + m5`, `c21 = m2 + m4`, `c22 = m1 − m2 + m3 + m6`.
4. Below `leaf` size it stops recursing and falls back to `_naive` — recursion on tiny
   blocks costs more in additions and list slicing than it saves in multiplies.
5. Finally it crops the padding off.

**What I wrote — `bench/matmul.py`:** a CLI (`--sizes`, `--algo`, `--leaf`) that times
`A.matmul(B)` vs `A.strassen(B, leaf=…)` on random matrices with `time.perf_counter()`
and prints a table with the winner.

**Real results on this machine** (`python bench/matmul.py --sizes 128,256,512 --leaf 32`):
```text
     n |    naive (s) |   strassen (s) | winner
----------------------------------------------------
   128 |       0.1528 |         0.1071 | strassen
   256 |       0.8114 |         0.8384 | naive
   512 |       9.2600 |         5.5680 | strassen
```
That first table contradicts itself (Strassen wins at 128, loses at 256), so I did
not trust a single run. I repeated n = 128/256 three more times:
```text
   128 |       0.0934 |         0.1099 | naive
   256 |       0.8378 |         0.7689 | strassen
   128 |       0.0972 |         0.1096 | naive
   256 |       0.8244 |         0.7791 | strassen
   128 |       0.1364 |         0.1106 | strassen     <- naive had a noisy spike
   256 |       0.8361 |         0.7702 | strassen
```
and once with `--leaf 64`:
```text
   128 |       0.0998 |         0.1078 | naive
   256 |       0.8960 |         0.8348 | strassen
   512 |       9.3361 |         5.6670 | strassen
```

**Honest conclusion:** at n = 128 the two are within noise and naive usually wins;
**Strassen reliably wins from n ≈ 256**, and at n = 512 it is **~1.65x faster**
(5.6 s vs 9.3 s). The crossover sits higher than the textbook's estimate because in
pure Python every one of Strassen's extra matrix additions and list slices runs
through the interpreter, inflating the constant factor that the asymptotic
argument ignores. And all of this is still ~700x slower than NumPy — algorithmic
cleverness does not beat a C/BLAS implementation of the dumb algorithm.

**Cross-check test:** `test_strassen_matches_naive` multiplies two random 130×130
matrices (deliberately *not* a power of two, to exercise padding) and asserts the
result matches NumPy with `np.allclose(..., atol=1e-6)` — not `==`, because Strassen
reorders floating-point additions and float addition is not associative.
```text
$ pytest -q
........                                                                 [100%]
8 passed
```

**Commit:** `feat: Strassen matmul + honest crossover benchmark`
