"""NumPy-backed reverse-mode autograd: a miniature PyTorch.

Same idea as the scalar engine, but `data` is a whole NumPy array, so one node
covers an entire layer's worth of numbers. That is the 100x speedup: we do the
math on arrays in C instead of looping over millions of Python objects.
"""
from __future__ import annotations
import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape` to undo NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    # --- elementwise ops ----------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    __matmul__ = matmul

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    # --- the engine ---------------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __repr__(self): return f"Tensor(shape={self.data.shape}, op={self._op!r})"


def softmax_cross_entropy(logits: "Tensor", y_idx: np.ndarray):
    """Numerically stable softmax + cross-entropy in one fused op.

    Returns (loss_tensor, probs). The gradient of this whole block w.r.t. the
    logits is simply (softmax(logits) - onehot(y)) / N -- one of the cleanest
    results in all of deep learning, which is why we fuse the two steps.
    """
    z = logits.data
    z = z - z.max(axis=1, keepdims=True)        # stability shift
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = z.shape[0]
    loss_val = -np.log(probs[np.arange(n), y_idx] + 1e-12).mean()
    out = Tensor(loss_val, (logits,), "softmax_xent")

    def _backward():
        d = probs.copy()
        d[np.arange(n), y_idx] -= 1.0
        d /= n
        logits.grad += d * out.grad
    out._backward = _backward
    return out, probs
