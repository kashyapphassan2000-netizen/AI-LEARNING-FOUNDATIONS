import math
import numpy as np
from hypothesis import given, strategies as st
from tiny_vec import Vector, Matrix


def test_vector_dot_matches_numpy():
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    assert Vector(a).dot(Vector(b)) == np.dot(a, b)


def test_vector_norm_matches_numpy():
    a = [3.0, 4.0]
    assert math.isclose(Vector(a).norm(), np.linalg.norm(a))


def test_vector_add_sub():
    a, b = Vector([1, 2]), Vector([3, 4])
    assert (a + b) == Vector([4, 6])
    assert (b - a) == Vector([2, 2])


nums = st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=20)


@given(nums, nums)
def test_dot_is_commutative(xs, ys):
    n = min(len(xs), len(ys))
    a, b = Vector(xs[:n]), Vector(ys[:n])
    assert math.isclose(a.dot(b), b.dot(a), rel_tol=1e-9)


def test_matmul_matches_numpy():
    A = [[1.0, 2.0], [3.0, 4.0]]; B = [[5.0, 6.0], [7.0, 8.0]]
    got = Matrix(A).matmul(Matrix(B)).to_list()
    assert np.allclose(got, np.array(A) @ np.array(B))


def test_transpose():
    A = [[1, 2, 3], [4, 5, 6]]
    assert Matrix(A).T.to_list() == np.array(A).T.tolist()


def test_identity():
    assert Matrix.identity(3).to_list() == np.eye(3).tolist()


def test_strassen_matches_naive():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((130, 130)); B = rng.standard_normal((130, 130))
    got = Matrix(A.tolist()).strassen(Matrix(B.tolist()), leaf=32).to_list()
    assert np.allclose(got, A @ B, atol=1e-6)
