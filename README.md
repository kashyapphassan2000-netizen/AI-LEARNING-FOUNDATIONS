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

### Day 2 (Tuesday) — BPE training, encode, decode in Rust (`src/lib.rs`)

**Goal:** the whole tokenizer in ~130 lines of Rust.

**The data:** a `Tokenizer` struct with two `HashMap`s —
`merges: (a, b) -> new_id` (the learned rules; a smaller id means "learned earlier")
and `vocab: id -> Vec<u8>` (what bytes each token expands to). `new()` seeds `vocab`
with the 256 single bytes.

**The three functions that *are* the algorithm:**
- `pair_counts(&[u32])` slides a window of 2 over the ids (`ids.windows(2)`) and
  counts each pair with `*counts.entry(pair).or_insert(0) += 1`.
- `merge(ids, pair, new_id)` walks left to right; when it sees the pair it emits
  `new_id` and skips 2, otherwise copies 1. `Vec::with_capacity(ids.len())` avoids
  re-allocations because the output can only shrink.
- `train(text, vocab_size)` loops `vocab_size − 256` times: count, pick the best pair
  with `max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))` — highest count, and on a tie
  the *smaller* pair (the reversed `b.0.cmp(a.0)` makes the smaller pair compare as
  "greater") — then merge, record the rule, and store the new token's bytes. It stops
  early if the best pair occurs fewer than 2 times.

**Encode / decode:**
- `encode` starts from bytes and repeatedly applies the merge with the **lowest id**
  among the pairs present. That replays training's order exactly, which is why the
  same text always encodes the same way.
- `decode` concatenates each id's bytes and converts with `String::from_utf8_lossy`.

**Rust things that clicked:** `&[u32]` borrows a slice without copying;
`self.vocab[&pair.0].clone()` copies because I need an owned `Vec` to extend;
`match best { Some(x) => x, None => break }` handles "no pairs left" explicitly — the
compiler will not let me forget that case.

**Proving it matches my Monday hand-trace — `tests/hand_trace.rs`:** train on
`lower lowest slow slowest` with 6 merges and assert the learned tokens are exactly
`["lo", "low", "lowe", " s", "st", "lowest"]` and that the sentence encodes to 8 tokens.
```text
$ cargo build
   Compiling blazetok v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.26s
$ cargo test --test hand_trace
test hand_trace_lower_lowest_slow_slowest ... ok
test result: ok. 1 passed; 0 failed
```

**Commit:** `feat: BPE training, encode, decode in Rust`

### Day 3 (Wednesday) — round-trip tests on English, Hindi, Tamil (`tests/roundtrip.rs`)

**Goal:** prove the defining property of a tokenizer — `decode(encode(x)) == x` —
for every script, including text the tokenizer never saw.

**The corpus** mixes English, Hindi (Devanagari) and Tamil sentences. Indic scripts
are where PDFs bite: fonts can reorder or substitute combining marks when you copy
text out. Before trusting the literals I checked their Unicode code points — e.g.
`नमस्ते` = NA, MA, SA, VIRAMA, TA, SIGN E and `வணக்கம்` = VA, NNA, KA, VIRAMA, KA,
MA, VIRAMA — logical order, nothing mangled.

**Six tests, each trains a 400-token vocabulary on the corpus:**
| test | what it proves |
|---|---|
| `roundtrip_english` | `"the quick brown fox"` survives encode → decode |
| `roundtrip_hindi` | `"नमस्ते दुनिया"` survives — 3-byte Devanagari characters are never split wrongly |
| `roundtrip_tamil` | `"வணக்கம் உலகம்"` survives |
| `roundtrip_unseen_text` | `"zany ☃ pangram 12345 ✓"` — emoji/symbols never seen in training still round-trip via **byte fallback** |
| `training_grows_vocab` | vocab starts at exactly 256 and ends in (256, 400] |
| `merges_reduce_token_count` | an English sentence encodes to fewer tokens than its raw byte count |

```text
$ cargo test
     Running tests/hand_trace.rs
test hand_trace_lower_lowest_slow_slowest ... ok
test result: ok. 1 passed; 0 failed
     Running tests/roundtrip.rs
test roundtrip_hindi ... ok
test roundtrip_english ... ok
test roundtrip_tamil ... ok
test merges_reduce_token_count ... ok
test roundtrip_unseen_text ... ok
test training_grows_vocab ... ok
test result: ok. 6 passed; 0 failed
```

**Why byte-level can't lose information:** every token's expansion is a byte string,
the 256 single bytes are always in the vocabulary, and merges only *concatenate*
bytes. So any byte sequence can be encoded and the concatenation on decode gives the
exact original bytes back.

**Commit:** `test: round-trip on en/hi/ta + unseen text`
