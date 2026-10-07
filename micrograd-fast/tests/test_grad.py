import numpy as np
from micrograd_fast import Tensor, softmax_cross_entropy, MLP


def numeric_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + h; fp = f()
        x[i] = old - h; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
        it.iternext()
    return g


def test_matmul_add_relu_gradients():
    rng = np.random.default_rng(0)
    A = Tensor(rng.standard_normal((4, 3)))
    B = Tensor(rng.standard_normal((3, 5)))
    b = Tensor(rng.standard_normal((5,)))

    def forward():
        return ((A @ B + b).relu().sum()).data

    out = (A @ B + b).relu().sum()
    out.backward()

    gA = numeric_grad(forward, A.data)
    assert np.allclose(A.grad, gA, atol=1e-5)


def test_softmax_cross_entropy_gradient():
    rng = np.random.default_rng(1)
    logits = Tensor(rng.standard_normal((6, 4)))
    y = rng.integers(0, 4, size=6)

    loss, _ = softmax_cross_entropy(logits, y)
    loss.backward()

    def forward():
        l, _ = softmax_cross_entropy(Tensor(logits.data), y)
        return l.data

    g = numeric_grad(forward, logits.data)
    assert np.allclose(logits.grad, g, atol=1e-5)


def test_mlp_forward_shape():
    net = MLP([8, 16, 3])
    x = Tensor(np.random.default_rng(0).standard_normal((10, 8)))
    out = net(x)
    assert out.shape == (10, 3)
