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
