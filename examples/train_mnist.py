"""Train on real MNIST to >=95% in under 30s on a laptop CPU.

This is the roadmap's actual target. It is the SAME engine as train_digits.py;
only the input dimension (784) and the data loader differ. Run it on a machine
with internet so the dataset can download once (it is then cached).
"""
import os
import time
import numpy as np
from micrograd_fast import Tensor, MLP, Adam, softmax_cross_entropy


def load_mnist():
    """Try a few common sources; whichever is installed wins."""
    try:
        from tensorflow.keras.datasets import mnist
        (Xtr, ytr), (Xte, yte) = mnist.load_data()
        Xtr = Xtr.reshape(-1, 784) / 255.0
        Xte = Xte.reshape(-1, 784) / 255.0
        return Xtr, ytr.astype(int), Xte, yte.astype(int)
    except Exception:
        pass
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype(int)
    return X[:60000], y[:60000], X[60000:], y[60000:]


def accuracy(net, X, y, batch=1000):
    correct = 0
    for s in range(0, len(X), batch):
        logits = net(Tensor(X[s:s + batch])).data
        correct += (logits.argmax(axis=1) == y[s:s + batch]).sum()
    return correct / len(X)


def save_loss_curve(losses, steps_per_epoch, path="docs/loss_curve.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = 20  # moving average so the trend is readable
    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.25, label="per-batch loss")
    ax.plot(np.arange(w - 1, len(losses)), smooth, label=f"{w}-batch moving average")
    ax.axhline(np.log(10), ls="--", c="gray", lw=1, label="ln(10) = random guessing")
    for e in range(1, len(losses) // steps_per_epoch + 1):
        ax.axvline(e * steps_per_epoch, c="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("training step (batch of 128)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("micrograd-fast on MNIST: MLP 784-128-10, Adam lr=1e-3")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    t_load = time.perf_counter()
    Xtr, ytr, Xte, yte = load_mnist()
    print(f"loaded MNIST: train {Xtr.shape}, test {Xte.shape} "
          f"in {time.perf_counter() - t_load:.1f}s (not counted in training time)")
    net = MLP([784, 128, 10], seed=0)
    opt = Adam(net.parameters(), lr=1e-3)

    batch, epochs = 128, 3
    rng = np.random.default_rng(0)
    losses = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            idx = order[s:s + batch]
            net.zero_grad()
            loss, _ = softmax_cross_entropy(net(Tensor(Xtr[idx])), ytr[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.data))
        print(f"epoch {epoch} test acc {accuracy(net, Xte, yte):.4f}")
    elapsed = time.perf_counter() - t0
    acc = accuracy(net, Xte, yte)
    print(f"\nFINAL: {acc:.4f} test accuracy in {elapsed:.1f}s")
    save_loss_curve(losses, steps_per_epoch=-(-len(Xtr) // batch))
    assert acc >= 0.95 and elapsed < 30, "target: >=95% in <30s"
    print("PASS")


if __name__ == "__main__":
    main()
