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

### Day 3 (Wednesday) — layers, the MLP, and Adam (`micrograd_fast/nn.py`)

**`Linear`** is `y = x @ W + b`. Weights use **He initialization** —
`N(0, 1) · sqrt(2 / fan_in)` — because relu zeroes about half its inputs, and the
factor 2 compensates so activations don't shrink layer after layer.

**`MLP(sizes)`** stacks `Linear` layers with relu between them but **not after the
last one**: the final layer outputs raw logits for softmax. `zero_grad()` resets
every parameter's gradient (otherwise gradients from batch t leak into batch t+1,
because every op uses `+=`).

**`Adam`** keeps two running averages per parameter: momentum
`m = β₁m + (1−β₁)g` and variance `v = β₂v + (1−β₂)g²`, bias-corrects both (they start
at 0, so early on they are too small; dividing by `1 − βᵗ` fixes that), and updates
`w −= lr · m̂ / (√v̂ + ε)`.

**Sanity checks:**
```text
params: 101770                     <- 784*128 + 128 + 128*10 + 10 for MLP([784,128,10])
hidden pre-relu std 0.783          <- He init keeps activations O(1) on [0,1] pixel input
grad   1000.0: first Adam step moved weight by 0.001000
grad    0.001: first Adam step moved weight by 0.001000
```
The last two lines show what makes Adam useful: on step 1, `m̂/√v̂ = g/|g| = ±1`,
so the step is exactly `lr` **whether the gradient is 1000 or 0.001**. Step size is
set by the learning rate, not by the raw scale of the gradient — so one `lr = 1e-3`
works for every layer.

**Public API — `micrograd_fast/__init__.py`:** `Value, Tensor, softmax_cross_entropy,
Linear, MLP, Adam`, `__version__ = "0.1.0"`.

**Commit:** `feat: Linear, MLP, Adam optimizer`

### Day 4 (Thursday) — prove the gradients are correct (`tests/test_grad.py`)

**Goal:** never trust an autograd engine you haven't gradient-checked.

**Method:** `numeric_grad(f, x)` nudges each entry of `x` by ±h (h = 1e-6), re-runs the
forward pass, and estimates the derivative as `(f(x+h) − f(x−h)) / 2h` (central
difference: error O(h²) instead of O(h)). If backprop's analytic gradient matches
this to 1e-5, the engine is right.

| test | what it covers |
|---|---|
| `test_matmul_add_relu_gradients` | `A @ B + b`, relu, sum — matmul backward **and** bias unbroadcasting in one graph |
| `test_softmax_cross_entropy_gradient` | the fused op's "predicted minus actual" gradient |
| `test_mlp_forward_shape` | `MLP([8,16,3])` maps a (10, 8) batch to (10, 3) |

```text
$ pytest -q
...                                                                      [100%]
3 passed in 0.22s
```

**Do the tests catch real bugs? I checked by breaking the engine on a scratch copy:**
- Deleted the `d /= n` line in the softmax backward (a classic mistake) →
  `1 failed, 2 passed`. The gradient check caught it.
- Changed matmul's `+=` to `=` and used `A` in two matmuls:
  ```text
  with '=' bug, A.grad = [[2.0, 2.0], [2.0, 2.0]] (correct is all 4.0)
  ```
  The second path overwrote the first instead of adding to it — exactly the
  "missing `+=`" failure the guide warns about, made visible.

I also added `pyproject.toml` (`numpy` as the only runtime dependency, pytest
`pythonpath = ["."]`) so `pytest` runs from the repo root.

**Commit:** `test: gradient checks vs numerical gradients`
