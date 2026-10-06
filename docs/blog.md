# BPE in Rust: lost on English, won on Tamil

A language model never sees text. It sees integers, and the tokenizer decides which
ones. This week I wrote one from scratch: **blazetok**, a byte-level Byte-Pair
Encoding tokenizer in Rust with Python bindings.

**BPE in three sentences.** Start with the 256 possible bytes as your vocabulary.
Repeatedly find the most frequent adjacent pair of tokens in the training text and
merge it into a new token. Encoding replays those merges in the order they were
learned; decoding concatenates the bytes each token stands for — so any UTF-8 string
round-trips exactly, with no "unknown token".

**Why Rust + PyO3.** Training is a tight loop of pair counting and rewriting
arrays — exactly the kind of code where an interpreter's per-operation overhead
dominates. In Rust it's ~130 lines with no garbage collector; PyO3 and maturin turn it
into a normal `pip install`-able module, so Python users get `blazetok.Tokenizer()` and
never see Rust.

**The result, honestly.** Devanagari and Tamil characters take ~2.6 bytes each in
UTF-8, and GPT-4's `cl100k_base` tokenizer, trained mostly on English, learned few
merges for them: I measured 1.02 tokens per character on Hindi and 1.37 on Tamil,
versus 0.20 on English. A BPE trained on Indic text does better — but by how much
depends on how you measure. Trained and tested on the same small word bank, blazetok
used 7.7x fewer tokens than cl100k on Hindi and 12.3x on Tamil. That number is
inflated: with the same setup blazetok also "beat" GPT-4 on English by 1.8x, which
just means it memorized the word bank. On held-out sentences with unseen words the
honest gap is **1.3x fewer tokens on Hindi and 1.9x on Tamil**.

So: lost on English (in any fair test), won on Tamil — modestly, and for a reason I
can explain. Fewer tokens means cheaper and faster inference for Indian-language users,
which is why tokenizer choice is a product decision, not a detail.

Code: https://github.com/kashyapphassan2000-netizen/blazetok
