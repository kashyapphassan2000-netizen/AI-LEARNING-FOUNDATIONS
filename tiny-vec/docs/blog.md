# I built tiny NumPy and learned why NumPy is fast

This week I wrote a linear-algebra and autograd library in pure Python, with zero
dependencies: a `Vector`, a `Matrix` with two matrix-multiply algorithms, a
reverse-mode autodiff engine, and SVD by power iteration. NumPy was allowed in
exactly one place: the tests, as the oracle my answers had to match.

**The speed gap.** My triple-loop matmul on a 128×128 matrix takes about 0.10 s.
NumPy does it in 0.00014 s — roughly 700 times faster. Same algorithm class, so
where does the gap come from? My inner loop runs in the Python interpreter: every
multiply-add boxes a float, dispatches `*` and `+` dynamically, and chases pointers
through a list of lists. NumPy hands the whole job to BLAS: compiled code that works
on contiguous memory, uses SIMD instructions, blocks for the CPU cache, and runs on
several cores. The lesson isn't "Python is slow"; it's "per-element interpreter
overhead is the cost you must eliminate", which is exactly what vectorization does.

**The Strassen surprise.** Strassen's algorithm replaces 8 recursive multiplies
with 7, dropping complexity to about O(n^2.807). The theory says it wins early. In
my measurements it was within noise of the naive loop at n=128, won reliably from
n≈256, and was 1.65x faster at n=512 (5.6 s vs 9.3 s). The crossover is late because
Strassen pays for its saved multiply with many extra matrix additions and slices,
and in Python each of those is interpreter-bound too. Asymptotics tell you the
shape of the curve; only a benchmark tells you where you are on it. My first run
even contradicted itself, so I repeated it before believing anything.

**Autograd.** The engine is about 100 lines: each operation records its inputs and
a closure that applies the local derivative, and `backward()` topologically sorts
the graph and runs those closures in reverse. A 105-parameter MLP learned XOR (loss
0.897 → 0.000004), and my from-scratch PCA matched scikit-learn on iris to 1e-10.

Code, tests (80% coverage) and CI: https://github.com/kashyapphassan2000-netizen/tiny-vec
