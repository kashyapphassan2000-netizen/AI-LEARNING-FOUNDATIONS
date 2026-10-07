# MNIST in 6 seconds with an autograd engine I wrote myself

`loss.backward()` used to be the line I trusted without understanding. This week I
wrote the thing behind it — twice.

**Version 1: scalars.** Every number is a `Value` object that remembers which
operation produced it and holds a tiny closure with that operation's local
derivative. `backward()` topologically sorts the graph and runs the closures in
reverse: the chain rule, mechanically. For `relu(a·b + c)` at a=2, b=−3, c=10 it
returns gradients −3, 2, 1, the same numbers I worked out on paper.

**Version 2: tensors.** Same design, but each node holds a whole NumPy array. Two
new ideas appear. The matmul gradient is `dA = dOut·Bᵀ`, `dB = Aᵀ·dOut`. And
broadcasting has to be undone: a bias added to a batch receives a gradient per row,
which must be summed back to the bias's shape.

**The speedup.** On the same small problem (a 16×64 by 64×16 matmul, relu, sum and
backward) the scalar engine needs ~130 ms and builds ~18,000 objects. The tensor
engine needs 0.34 ms cold and 0.03 ms warm: roughly **400x to 4000x**, depending on
whether you count warm-up. I report both, because the guide I followed quoted only one.

**The result.** A 784-128-10 MLP with my Adam optimizer and a fused
softmax-cross-entropy (whose gradient is just "predicted minus actual") reaches
**96.85% test accuracy on MNIST in 6.0 seconds** on a CPU. The loss starts at
2.30, which is ln(10), the loss of guessing at random, and drops under 0.5 within the
first eighth of an epoch.

**How I know it's right.** Every gradient is checked against finite differences.
To check the checks, I broke the engine on purpose: removing the divide-by-batch in the
softmax gradient failed a test, and swapping `+=` for `=` in matmul halved the gradient
of a value used twice.

Code: https://github.com/kashyapphassan2000-netizen/micrograd-fast
