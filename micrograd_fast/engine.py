"""Scalar reverse-mode autograd -- the conceptual foundation.

This is the simplest possible autograd: every number is a Value that remembers
how it was made. It is correct but slow, because a real network would create
millions of these objects. Day 3 replaces it with a NumPy-backed Tensor that
does the same math on whole arrays at once.
"""
from __future__ import annotations
import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __radd__(self, o): return self + o
    def __sub__(self, o): return self + (-o)
    def __rmul__(self, o): return self * o
    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
