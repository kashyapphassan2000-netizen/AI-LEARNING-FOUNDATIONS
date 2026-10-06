"""A from-scratch Matrix type with two matmul algorithms.

`matmul` is the obvious triple loop (O(n^3)). `strassen` is the divide-and-
conquer algorithm that does 7 recursive multiplies instead of 8, giving
~O(n^2.807). We keep both so we can *measure* the crossover instead of
asserting it.
"""
from __future__ import annotations
from typing import List, Sequence

Row = Sequence[float]


class Matrix:
    __slots__ = ("data", "rows", "cols")

    def __init__(self, data: Sequence[Row]):
        self.data: List[List[float]] = [[float(x) for x in row] for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.rows else 0
        for r in self.data:
            if len(r) != self.cols:
                raise ValueError("ragged matrix: all rows must share a length")

    # --- constructors -------------------------------------------------------
    @classmethod
    def identity(cls, n: int) -> "Matrix":
        return cls([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        return cls([[0.0] * cols for _ in range(rows)])

    # --- plumbing -----------------------------------------------------------
    def __getitem__(self, idx):
        return self.data[idx]

    def __eq__(self, other) -> bool:
        return isinstance(other, Matrix) and self.data == other.data

    def __repr__(self) -> str:
        return f"Matrix({self.rows}x{self.cols})"

    def to_list(self) -> List[List[float]]:
        return [row[:] for row in self.data]

    def transpose(self) -> "Matrix":
        return Matrix([[self.data[r][c] for r in range(self.rows)]
                       for c in range(self.cols)])

    @property
    def T(self) -> "Matrix":
        return self.transpose()

    def __add__(self, other: "Matrix") -> "Matrix":
        return Matrix([[a + b for a, b in zip(ra, rb)]
                       for ra, rb in zip(self.data, other.data)])

    def __sub__(self, other: "Matrix") -> "Matrix":
        return Matrix([[a - b for a, b in zip(ra, rb)]
                       for ra, rb in zip(self.data, other.data)])

    # --- naive matmul -------------------------------------------------------
    def matmul(self, other: "Matrix") -> "Matrix":
        if self.cols != other.rows:
            raise ValueError(f"shape mismatch: {self.cols} vs {other.rows}")
        bt = other.transpose().data  # cache columns of `other` as rows
        out = [[sum(a * b for a, b in zip(row, col)) for col in bt]
               for row in self.data]
        return Matrix(out)

    __matmul__ = matmul

    # --- Strassen -----------------------------------------------------------
    def strassen(self, other: "Matrix", leaf: int = 64) -> "Matrix":
        if self.cols != other.rows:
            raise ValueError("shape mismatch for strassen")
        n = max(self.rows, self.cols, other.rows, other.cols)
        m = 1
        while m < n:
            m *= 2  # pad up to next power of two
        A = _pad(self.data, m)
        B = _pad(other.data, m)
        C = _strassen(A, B, leaf)
        return Matrix([row[: other.cols] for row in C[: self.rows]])


def _pad(mat: List[List[float]], m: int) -> List[List[float]]:
    out = [[0.0] * m for _ in range(m)]
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            out[i][j] = v
    return out


def _add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(A))]


def _sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(A))]


def _naive(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]


def _strassen(A, B, leaf):
    n = len(A)
    if n <= leaf:
        return _naive(A, B)
    h = n // 2
    a11 = [r[:h] for r in A[:h]]
    a12 = [r[h:] for r in A[:h]]
    a21 = [r[:h] for r in A[h:]]
    a22 = [r[h:] for r in A[h:]]
    b11 = [r[:h] for r in B[:h]]
    b12 = [r[h:] for r in B[:h]]
    b21 = [r[:h] for r in B[h:]]
    b22 = [r[h:] for r in B[h:]]

    m1 = _strassen(_add(a11, a22), _add(b11, b22), leaf)
    m2 = _strassen(_add(a21, a22), b11, leaf)
    m3 = _strassen(a11, _sub(b12, b22), leaf)
    m4 = _strassen(a22, _sub(b21, b11), leaf)
    m5 = _strassen(_add(a11, a12), b22, leaf)
    m6 = _strassen(_sub(a21, a11), _add(b11, b12), leaf)
    m7 = _strassen(_sub(a12, a22), _add(b21, b22), leaf)

    c11 = _add(_sub(_add(m1, m4), m5), m7)
    c12 = _add(m3, m5)
    c21 = _add(m2, m4)
    c22 = _add(_sub(_add(m1, m3), m2), m6)

    C = [[0.0] * n for _ in range(n)]
    for i in range(h):
        for j in range(h):
            C[i][j] = c11[i][j]
            C[i][j + h] = c12[i][j]
            C[i + h][j] = c21[i][j]
            C[i + h][j + h] = c22[i][j]
    return C
