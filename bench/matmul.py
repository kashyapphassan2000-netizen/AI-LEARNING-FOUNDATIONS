"""Benchmark naive vs Strassen matmul. Reports wall-clock; honest about Python."""
import argparse
import random
import time
from tiny_vec import Matrix


def rand_matrix(n):
    return Matrix([[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)])


def timeit(fn, *a, **k):
    t0 = time.perf_counter()
    fn(*a, **k)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,128,256")
    ap.add_argument("--algo", default="naive,strassen")
    ap.add_argument("--leaf", type=int, default=64)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    algos = args.algo.split(",")
    print(f"{'n':>6} | {'naive (s)':>12} | {'strassen (s)':>14} | winner")
    print("-" * 52)
    for n in sizes:
        A, B = rand_matrix(n), rand_matrix(n)
        tn = timeit(A.matmul, B) if "naive" in algos else float("nan")
        ts = timeit(A.strassen, B, leaf=args.leaf) if "strassen" in algos else float("nan")
        winner = "naive" if tn < ts else "strassen"
        print(f"{n:>6} | {tn:>12.4f} | {ts:>14.4f} | {winner}")


if __name__ == "__main__":
    main()
