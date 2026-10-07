"""SVD by power iteration, and PCA built on top of it.

We never call a library SVD. We find the top singular triplets of A by power-
iterating on A^T A (whose eigenvectors are the right singular vectors V, and
whose eigenvalues are the squared singular values). Deflation peels off one
component at a time. This is slow but transparent -- exactly what Week 1 wants.
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple

Mat = List[List[float]]
Vec = List[float]


def _matvec(A: Mat, x: Vec) -> Vec:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _transpose(A: Mat) -> Mat:
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _gram(A: Mat) -> Mat:
    """A^T A."""
    At = _transpose(A)
    n = len(At)
    return [[sum(At[i][k] * At[j][k] for k in range(len(At[0])))
             for j in range(n)] for i in range(n)]


def _norm(x: Vec) -> float:
    return math.sqrt(sum(v * v for v in x))


def _normalize(x: Vec) -> Vec:
    n = _norm(x)
    return [v / n for v in x] if n else x


def power_iteration(M: Mat, iters: int = 1000, tol: float = 1e-12,
                    seed: int = 0) -> Tuple[float, Vec]:
    """Dominant eigenvalue/eigenvector of a symmetric matrix M."""
    rng = random.Random(seed)
    n = len(M)
    b = _normalize([rng.gauss(0, 1) for _ in range(n)])
    lam = 0.0
    for _ in range(iters):
        nb = _matvec(M, b)
        nb = _normalize(nb)
        new_lam = sum(nb[i] * sum(M[i][j] * nb[j] for j in range(n))
                      for i in range(n))
        if abs(new_lam - lam) < tol:
            b, lam = nb, new_lam
            break
        b, lam = nb, new_lam
    return lam, b


def svd(A: Mat, k: int = None, seed: int = 0):
    """Return U (m x k), S (length k), Vt (k x n) with A ~= U diag(S) Vt."""
    m, n = len(A), len(A[0])
    k = min(m, n) if k is None else k
    G = _gram(A)  # n x n, symmetric PSD
    Vt: List[Vec] = []
    S: List[float] = []
    for comp in range(k):
        lam, v = power_iteration(G, seed=seed + comp)
        sigma = math.sqrt(max(lam, 0.0))
        Vt.append(v)
        S.append(sigma)
        # deflate: G <- G - lam * v v^T
        G = [[G[i][j] - lam * v[i] * v[j] for j in range(n)] for i in range(n)]
    # U columns: u = A v / sigma
    U = [[0.0] * k for _ in range(m)]
    for c in range(k):
        Av = _matvec(A, Vt[c])
        s = S[c] if S[c] > 1e-12 else 1.0
        for i in range(m):
            U[i][c] = Av[i] / s
    return U, S, Vt


def pca(X: Mat, n_components: int = 2, seed: int = 0):
    """Center X, take top components via SVD. Returns (scores, components, mean)."""
    m, n = len(X), len(X[0])
    mean = [sum(X[i][j] for i in range(m)) / m for j in range(n)]
    Xc = [[X[i][j] - mean[j] for j in range(n)] for i in range(m)]
    U, S, Vt = svd(Xc, k=n_components, seed=seed)
    # scores = Xc @ V = U * S
    scores = [[U[i][c] * S[c] for c in range(n_components)] for i in range(m)]
    return scores, Vt, mean
