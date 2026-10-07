"""Offline check: the same engine on scikit-learn's bundled 8x8 digits (no download).

Identical training loop to train_mnist.py; only the input size (64 vs 784) and
the data source differ. Use this when MNIST can't be downloaded.
"""
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from micrograd_fast import Tensor, MLP, Adam, softmax_cross_entropy


def main():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # pixel values are 0..16
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    net = MLP([64, 128, 10], seed=0)
    opt = Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for epoch in range(30):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            idx = order[s:s + 64]
            net.zero_grad()
            loss, _ = softmax_cross_entropy(net(Tensor(Xtr[idx])), ytr[idx])
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - t0
    acc = (net(Tensor(Xte)).data.argmax(axis=1) == yte).mean()
    print(f"digits 8x8: {len(Xtr)} train / {len(Xte)} test, "
          f"test accuracy {acc:.4f} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
