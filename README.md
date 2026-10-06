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

### Day 4 (Thursday) — PyO3 bindings + build the Python wheel

**Goal:** Rust speed behind a normal Python API: `import blazetok`.

**`src/py.rs`** wraps the core type: `#[pyclass(name = "Tokenizer")] struct PyTokenizer
{ inner: CoreTokenizer }` and a `#[pymethods]` block whose methods (`new`, `train`,
`encode`, `decode`, `vocab_size`) just forward to the Rust tokenizer. PyO3 converts
types at the boundary for me: Python `str` → `&str`, Python `list[int]` →
`Vec<u32>` and back. `#[pymodule] fn blazetok` registers the class. The file is only
compiled with `--features python` (`#[cfg(feature = "python")] mod py;` at the end of
`lib.rs`), so `cargo test` never needs Python.

**`pyproject.toml`** tells pip/maturin how to build: `build-backend = "maturin"`, and
`[tool.maturin] features = ["python"]` switches the bindings on.

**Build + install:**
```text
$ maturin build --release
warning: non-local `impl` definition ... the attribute macro `pymethods` may come from an old version of the `pyo3_macros` crate
    Finished `release` profile [optimized] target(s) in 10.53s
📦 Built wheel for CPython 3.12 to target/wheels/blazetok-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
$ pip install target/wheels/*.whl --force-reinstall
```
The one warning is a known lint: PyO3 0.20's macros (locked at 0.20.3) predate the
`non_local_definitions` lint in newer rustc. It is harmless; upgrading to PyO3 ≥ 0.21
silences it but changes the binding API (`Bound<PyModule>`), so I kept the version
the guide pins.

**Two snags and how I fixed them:**
1. PyO3 0.20 only supports CPython ≤ 3.12 → I build inside a 3.12 venv (see Day 0).
2. My first `pip install` failed with *"is not a supported wheel on this platform"* —
   `pip` on my PATH belonged to the system Python 3.13, not the 3.12 venv the wheel was
   built for (`cp312`). Installing with the venv's own installer fixed it. Lesson:
   when a wheel "isn't supported", check *which* interpreter's pip you're running.

**Verify from Python** (run from `/tmp`, so it imports the installed wheel, not the source):
```text
>>> t = blazetok.Tokenizer(); t.train('the quick brown fox '*200, 400)
vocab_size 282
the quick brown fox -> 1 tokens [273] roundtrip True
>>> # trained on the Wednesday test corpus instead:
corpus-trained vocab_size 338
नमस्ते दुनिया 37 bytes -> 7 tokens, roundtrip True
வணக்கம் உலகம் 37 bytes -> 5 tokens, roundtrip True
```
Two things worth noticing: (1) on the repetitive text, training **stopped at 282**,
not 400 — once the whole phrase is one token there are no pairs left that occur
twice, so the `count < 2` early-exit fires; (2) on Indic text BPE already turns 37
UTF-8 bytes into 5–7 tokens after a tiny amount of training. That's the Friday story.

**Commit:** `feat: PyO3 bindings + maturin build, importable from Python`

### Day 5 (Friday) — benchmark token efficiency vs tiktoken

**Goal:** the headline claim — an Indic-trained BPE needs far fewer tokens for Hindi
and Tamil than tiktoken's `cl100k_base` (the GPT-4 tokenizer), which was trained
mostly on English.

**`bench/compare_tiktoken.py`:** builds reproducible text (seed 0) by sampling from
word banks of common Hindi, Tamil and English words with occasional punctuation;
trains a 2000-token blazetok per language on 6000 sampled words; then encodes a
separate 800-word sample with both tokenizers and reports **tokens per character**
(the fair unit — it doesn't care how many bytes a script needs).

```text
$ python bench/compare_tiktoken.py
=== Hindi (test text: 4345 chars) ===
  blazetok (Indic-trained) :    578 tokens  (0.133 tok/char)
  tiktoken cl100k (GPT-4)  :   4430 tokens  (1.020 tok/char)
  -> tiktoken uses 7.66x as many tokens on this text

=== Tamil (test text: 4957 chars) ===
  blazetok (Indic-trained) :    553 tokens  (0.112 tok/char)
  tiktoken cl100k (GPT-4)  :   6788 tokens  (1.369 tok/char)
  -> tiktoken uses 12.27x as many tokens on this text

=== English (test text: 4782 chars) ===
  blazetok (Indic-trained) :    525 tokens  (0.110 tok/char)
  tiktoken cl100k (GPT-4)  :    965 tokens  (0.202 tok/char)
  -> tiktoken uses 1.84x as many tokens on this text
```

**`bench/compare_offline.py`** (no download needed): compares against the raw
byte baseline, and adds a **held-out** check on real sentences built from words
*outside* the word bank.
```text
$ cd bench && python compare_offline.py
script    UTF-8 bytes/char  blazetok tok/char  vs raw bytes
Hindi                 2.55              0.133          19x
Tamil                 2.62              0.112          23x

held-out sentences (words/sentence structure partly unseen):
Hindi     145 bytes ->   45 tokens (3.2x fewer), roundtrip True
Tamil     155 bytes ->   43 tokens (3.6x fewer), roundtrip True
```
And tiktoken on those same held-out sentences: Hindi 60 tokens (1.09 tok/char),
Tamil 80 tokens (1.36 tok/char).

**Reading these numbers honestly — this is the part that matters:**
1. **The 7.7x / 12.3x headline is inflated by the setup.** Train and test text come
   from the *same* 25-word bank, so blazetok simply learns whole words. Proof: the
   **English** row — blazetok "beats" GPT-4's tokenizer by 1.84x on *English*, which
   is obviously not a real win; it just memorized 15 words.
2. **The honest held-out number is much smaller:** on sentences with unseen words,
   blazetok uses 45 vs 60 tokens on Hindi (**1.33x fewer** than cl100k) and 43 vs 80
   on Tamil (**1.86x fewer**) — after training on nothing but a tiny word bank.
3. **cl100k is not pure byte fallback.** The guide's primer suggests English-first
   tokenizers spend ~2.6 tokens per Devanagari character; that's the *raw-byte*
   rate (2.55 here). cl100k actually spends **1.02 tok/char on Hindi** and **1.37 on
   Tamil** — it has some Indic merges, just far fewer than English ones (0.20 tok/char).
4. **Direction is solid, magnitude needs real data:** Tamil is penalized more than
   Hindi in cl100k, and a tokenizer that has seen the script compresses it. The real
   measurement requires training on a large Indic corpus (AI4Bharat/IndicCorp) and
   testing on held-out text — that's the next step, not a claim I can make today.

**Commit:** `bench: token efficiency vs tiktoken + offline baseline`
