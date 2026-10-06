"""Linear layers, an MLP, and the Adam optimizer -- all on the Tensor engine."""
from __future__ import annotations
import numpy as np
from .tensor import Tensor


class Linear:
    """A fully-connected layer: y = x @ W + b."""
    def __init__(self, nin, nout, rng):
        # He initialization keeps activation variance stable through relu.
        scale = np.sqrt(2.0 / nin)
        self.W = Tensor(rng.standard_normal((nin, nout)) * scale)
        self.b = Tensor(np.zeros(nout))

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [Linear(sizes[i], sizes[i + 1], rng)
                       for i in range(len(sizes) - 1)]

    def __call__(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # relu on all but the last layer
                x = x.relu()
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Adam:
    """Adam: per-parameter adaptive learning rates with momentum."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)  # bias correction
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
