import math
import numpy as np
from tiny_vec import Value, svd, power_iteration


def test_engine_matches_numeric_gradient():
    def f(a, b):
        return (a * b + a ** 2) * b.relu()
    a, b = Value(-1.5), Value(2.0)
    out = f(a, b); out.backward()
    h = 1e-6
    ga = (f(Value(-1.5 + h), Value(2.0)).data - f(Value(-1.5 - h), Value(2.0)).data) / (2 * h)
    assert math.isclose(a.grad, ga, abs_tol=1e-4)


def test_power_iteration_finds_dominant_eigenpair():
    M = [[2.0, 0.0], [0.0, 1.0]]
    lam, v = power_iteration(M)
    assert math.isclose(lam, 2.0, abs_tol=1e-6)


def test_singular_values_match_numpy():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((5, 3))
    _, S, _ = svd(A.tolist())
    s_np = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(sorted(S, reverse=True), s_np, atol=1e-4)
