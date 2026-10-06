# micrograd-fast

**A tiny autograd engine that trains MNIST in seconds on a CPU** — Project 03 of 12 (ZERO_TO_ANTHROPIC).

Built twice: first a **scalar** engine (every number is an object — simple, correct,
slow), then a **NumPy tensor** engine (every node is a whole array — same math, orders
of magnitude faster). On top: Linear layers, an MLP, Adam, a fused
softmax-cross-entropy, gradient checks, a real MNIST run, and a Graphviz visualizer.

Stack: Python 3.12, NumPy, scikit-learn (data loading), matplotlib, pytest, Graphviz. Cost: $0.

---

## Build log — the story, day by day

### Day 0 — setup

```bash
mkdir micrograd-fast && cd micrograd-fast
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pytest scikit-learn matplotlib
sudo apt-get install -y graphviz          # `dot -V` -> graphviz version 2.43.0
mkdir -p micrograd_fast tests examples docs && touch micrograd_fast/__init__.py
git init && git add -A && git commit -m 'chore: scaffold micrograd-fast'
```
I also checked up front that the MNIST sources are reachable from this machine
(openml.org and the Keras dataset bucket both returned HTTP 200) — the original
guide had to fall back to the small 8×8 digits set because its sandbox was offline.
Here I can run the **real** 28×28 MNIST target on Friday.

### Day 1 (Monday) — the scalar autograd engine (`micrograd_fast/engine.py`)

**Goal:** the simplest correct autograd, so the fast version has a reference.

**The worked example I built it against** — `L = relu(a·b + c)` with a = 2, b = −3, c = 10:
```text
forward:   e = a*b = -6     d = e + c = 4     L = relu(d) = 4
backward:  dL/dL = 1
           dL/dd = 1 * relu'(4) = 1          (d > 0, so relu passes gradient)
           dL/dc = dL/dd * 1 = 1
           dL/de = dL/dd * 1 = 1
           dL/da = dL/de * b = -3
           dL/db = dL/de * a =  2
```
Each op knows only its *local* derivative; backprop multiplies them along the chain.

**The code:** `Value` holds `data`, `grad`, its parents `_prev`, the op name, and a
`_backward` closure. `__add__` routes the incoming gradient unchanged to both inputs;
`__mul__` sends `other.data * out.grad` to `self` (the d(a·b)/da = b rule) and vice
versa; `__pow__` uses `p·xᵖ⁻¹`; `relu` passes gradient only where the output is
positive. `backward()` topologically sorts the graph (DFS, append a node after its
parents), seeds `grad = 1.0` at the output, and runs every `_backward` in reverse.
All gradient updates use `+=` so a value reused in two places sums both contributions.

**Sanity check — it reproduces the hand calculation exactly:**
```text
$ python -c "from micrograd_fast import Value as V; a=V(2); b=V(-3); c=V(10); \
             L=(a*b+c).relu(); L.backward(); print(a.grad, b.grad, c.grad)"
-3.0 2.0 1.0
```

**Commit:** `feat: scalar reverse-mode autograd engine`

### Day 2 (Tuesday) — the fast Tensor engine (`micrograd_fast/tensor.py`)

**Goal:** the same autograd design, but each node holds a whole NumPy array.

**What's new compared to the scalar engine:**
- **`matmul`** — for `out = A @ B` the gradients are `dA = dOut @ Bᵀ` and
  `dB = Aᵀ @ dOut`. Shapes force this: `dA` must be shaped like `A`.
- **`_unbroadcast`** — adding a bias of shape `(10,)` to a batch `(N, 10)` makes NumPy
  copy the bias across N rows. The gradient coming back is `(N, 10)`, but the bias has
  only 10 numbers, so the gradient must be **summed over the batch axis**.
  `_unbroadcast` removes extra leading axes by summing and sums any axis where the
  original size was 1.
- **`softmax_cross_entropy`** — a fused op. Forward: subtract the row max (prevents
  `exp` overflow, doesn't change the result), softmax, then mean of `−log p[y]`.
  Backward: the gradient w.r.t. the logits is `(softmax − onehot(y)) / N` —
  "predicted minus actual".

**Quick checks I ran before moving on:**
```text
grads [-3.] [2.] [1.]                     <- Monday's example, now as tensors
bias.grad shape (10,) values [4. 4. 4.]   <- (4,10)+(10,) bias grad summed over 4 rows
uniform-logit loss 2.3026 ln(10)= 2.3026  <- 10 equal logits = random guessing = ln 10
```

**Measuring the speedup — `examples/bench_speedup.py`:** the same computation,
`sum(relu(X @ W))` with X 16×64 and W 64×16, plus its backward pass, run on both
engines. The scalar version builds ~18,000 `Value` objects; the tensor version builds
4 nodes. The script asserts the two engines' gradients agree before timing anything.
```text
$ PYTHONPATH=. python examples/bench_speedup.py
problem: X 16x64 @ W 64x16, relu, sum, backward
scalar (Value) :   126.70 ms
tensor (NumPy) :    0.031 ms
speedup        :     4151x   (gradients match: True)
```
That's ~10x bigger than the guide's 432x, so I checked why. The script reports the
*best of 50* tensor runs (warm caches). A single cold tensor run took **0.340 ms →
410x**, which matches the guide's number. Both are true; they measure different
things. The honest summary: **400x cold, ~4000x warm**, and the gap keeps growing with
problem size, because the scalar cost is per number while NumPy does the arithmetic
in compiled code.

**Commit:** `feat: NumPy-backed Tensor engine + fused softmax-cross-entropy`
