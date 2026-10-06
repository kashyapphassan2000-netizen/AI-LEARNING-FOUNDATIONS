import math
from tiny_vec import Value


def test_engine_matches_numeric_gradient():
    def f(a, b):
        return (a * b + a ** 2) * b.relu()
    a, b = Value(-1.5), Value(2.0)
    out = f(a, b); out.backward()
    h = 1e-6
    ga = (f(Value(-1.5 + h), Value(2.0)).data - f(Value(-1.5 - h), Value(2.0)).data) / (2 * h)
    assert math.isclose(a.grad, ga, abs_tol=1e-4)
