# blazetok

**A byte-level BPE tokenizer in Rust, with Python bindings** — Project 02 of 12 (ZERO_TO_ANTHROPIC).

A tokenizer is the first stage of every language model: it turns text into integer
ids and back. blazetok implements Byte-Pair Encoding (the family GPT and Claude
use) from scratch in Rust, exposes it to Python with PyO3/maturin, and measures
how many fewer tokens an Indic-trained BPE needs for Hindi and Tamil.

Stack: Rust (2021 edition), PyO3 0.20, maturin, cargo test, Python 3.12. Cost: $0.

---

## Build log — the story, day by day

### Day 0 — toolchain setup

**Goal:** Rust, cargo, and maturin on the PATH; an empty crate under git.

1. Rust toolchain: `cargo --version` → `cargo 1.97.0` (the guide asks for ≥ 1.70).
2. Python side: `pip install maturin tiktoken` in a Python **3.12** venv. I picked 3.12
   on purpose: PyO3 0.20 supports CPython up to 3.12, and this machine's default
   `python3` is 3.13, which PyO3 0.20 would refuse to build against.
3. Created the crate and folders, first commit:
   ```bash
   cargo new --lib blazetok && cd blazetok
   mkdir -p tests examples bench docs
   git add -A && git commit -m 'chore: scaffold blazetok crate'
   ```

### Day 1 (Monday) — understand BPE + write `Cargo.toml`

**Goal:** understand the algorithm *before* typing it.

**Hand-trace.** I traced BPE on `lower lowest slow slowest` (25 bytes) and checked
my trace with a 15-line script that runs the exact merge loop:
```text
start      tokens left 25
step 1: ('l', 'o') count 4 -> 'lo'  tokens left 21
step 2: ('lo', 'w') count 4 -> 'low'  tokens left 17
step 3: ('low', 'e') count 3 -> 'lowe'  tokens left 14
step 4: (' ', 's') count 2 -> ' s'  tokens left 12
step 5: ('s', 't') count 2 -> 'st'  tokens left 10
step 6: ('lowe', 'st') count 2 -> 'lowest'  tokens left 8
```
"low" emerges first because it is the most common chunk, and after six merges the
whole word "lowest" is a single token.

**The bug I almost shipped:** my first checking script broke ties by comparing the
*byte strings* of the pairs, and at step 5 it chose `(lowe, s)` instead of `(s, t)`.
The Rust code breaks ties by **token id** (`(115,116) < (258,115)`), which is what
the primer table shows. Same algorithm, different vocabulary — a reminder that
tie-breaking is part of the spec, and why I write it down.

**`Cargo.toml`:** edition 2021; `crate-type = ["rlib", "cdylib"]` builds both a normal
Rust library (for `cargo test` and Rust users) and a C-compatible dynamic library
(which Python loads as an extension module). PyO3 is an **optional** dependency
behind a `python` feature, so `cargo test` stays pure-Rust and fast.

**`docs/algorithm.md`:** the merge loop, encode and decode in my own words, plus the
hand-trace table and the tie-break gotcha.

**Commit:** `docs: BPE algorithm notes + Cargo.toml`
