"""Scalar Value engine vs NumPy Tensor engine on the same computation.

Computes sum(relu(X @ W)) and its gradient for X: (16, 64), W: (64, 16) both ways,
checks the gradients agree, and reports the wall-clock ratio.
"""
import time
import numpy as np
from micrograd_fast import Value, Tensor

N, D, H = 16, 64, 16
rng = np.random.default_rng(0)
Xn = rng.standard_normal((N, D))
Wn = rng.standard_normal((D, H))


def scalar_run():
    X = [[Value(v) for v in row] for row in Xn]
    W = [[Value(v) for v in row] for row in Wn]
    out = Value(0.0)
    for i in range(N):
        for j in range(H):
            acc = Value(0.0)
            for k in range(D):
                acc = acc + X[i][k] * W[k][j]
            out = out + acc.relu()
    out.backward()
    return np.array([[w.grad for w in row] for row in W])


def tensor_run():
    X, W = Tensor(Xn), Tensor(Wn)
    out = (X @ W).relu().sum()
    out.backward()
    return W.grad


def best_of(fn, repeats):
    best, result = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


if __name__ == "__main__":
    ts, gs = best_of(scalar_run, 3)
    tt, gt = best_of(tensor_run, 50)
    assert np.allclose(gs, gt), "engines disagree!"
    print(f"problem: X {N}x{D} @ W {D}x{H}, relu, sum, backward")
    print(f"scalar (Value) : {ts * 1e3:8.2f} ms")
    print(f"tensor (NumPy) : {tt * 1e3:8.3f} ms")
    print(f"speedup        : {ts / tt:8.0f}x   (gradients match: True)")
