"""Train a tiny MLP on XOR using our own autograd. Target: loss < 0.01."""
from tiny_vec import MLP, Value
import random

random.seed(42)

# XOR is the canonical "needs a hidden layer" problem.
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = MLP(2, [8, 8, 1])
print(f"params: {len(model.parameters())}")

lr = 0.05
for epoch in range(2000):
    # forward: mean squared error over the 4 examples
    preds = [model([Value(x[0]), Value(x[1])]) for x in X]
    loss = sum(((p - y) ** 2 for p, y in zip(preds, Y)), Value(0.0)) * (1.0 / len(X))

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss {loss.data:.6f}")

final = loss.data
print(f"\nFINAL loss {final:.6f}")
for x, y in zip(X, Y):
    pred = model([Value(x[0]), Value(x[1])]).data
    print(f"  {x} -> {pred:+.3f}  (target {y})")

assert final < 0.01, f"XOR did not converge: {final}"
print("\nPASS: loss < 0.01")
