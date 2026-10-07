"""A from-scratch Vector type.

No NumPy inside the implementation. NumPy is only ever used in tests as the
"oracle" we check ourselves against. The whole point of Week 1 is that the math
lives in your hands, not behind a C extension.
"""
from __future__ import annotations
import math
from typing import Iterable, List, Union

Number = Union[int, float]


class Vector:
    __slots__ = ("data",)

    def __init__(self, data: Iterable[Number]):
        self.data: List[float] = [float(x) for x in data]

    # --- dunder plumbing so a Vector feels native ---------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __repr__(self) -> str:
        return f"Vector({self.data})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Vector) and self.data == other.data

    def _check(self, other: "Vector") -> None:
        if len(self) != len(other):
            raise ValueError(f"dim mismatch: {len(self)} vs {len(other)}")

    # --- core algebra -------------------------------------------------------
    def __add__(self, other: "Vector") -> "Vector":
        self._check(other)
        return Vector(a + b for a, b in zip(self.data, other.data))

    def __sub__(self, other: "Vector") -> "Vector":
        self._check(other)
        return Vector(a - b for a, b in zip(self.data, other.data))

    def __mul__(self, scalar: Number) -> "Vector":
        # element-wise scalar multiply (Hadamard-with-scalar)
        return Vector(a * scalar for a in self.data)

    __rmul__ = __mul__

    def __neg__(self) -> "Vector":
        return Vector(-a for a in self.data)

    def dot(self, other: "Vector") -> float:
        self._check(other)
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm(self, p: int = 2) -> float:
        if p == 2:
            return math.sqrt(self.dot(self))
        if p == 1:
            return sum(abs(a) for a in self.data)
        return sum(abs(a) ** p for a in self.data) ** (1.0 / p)

    def normalize(self) -> "Vector":
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("cannot normalize the zero vector")
        return Vector(a / n for a in self.data)

    def to_list(self) -> List[float]:
        return list(self.data)
