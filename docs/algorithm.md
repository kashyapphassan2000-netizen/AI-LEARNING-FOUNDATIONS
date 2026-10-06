# BPE — the merge loop in my own words

1. Turn the training text into bytes. Every byte (0–255) is already a token, so the
   starting vocabulary is 256 tokens and *any* UTF-8 text is representable.
2. Count every adjacent pair of tokens in the sequence.
3. Take the most frequent pair. (Ties: pick the pair with the smaller token ids, so
   training is deterministic — HashMap iteration order is random in Rust.)
4. Give that pair a brand-new id (256, 257, …), record the rule `(a, b) -> new_id`,
   and record that `new_id` expands to `bytes(a) + bytes(b)`.
5. Replace every occurrence of the pair in the sequence with the new id
   (left-to-right, non-overlapping).
6. Repeat 2–5 until the vocabulary hits the target size or no pair occurs twice.

**Encode** = start from bytes, then repeatedly apply the *earliest-learned* merge that
is present, until none applies. This replays training's merge order exactly.

**Decode** = look up the bytes each id expands to, concatenate, interpret as UTF-8.
Byte-level means decode(encode(x)) == x for every string — no unknown token.

## Hand trace: `lower lowest slow slowest`

| step | most frequent pair | count | new token | tokens left |
|---|---|---|---|---|
| start | — | — | — | 25 |
| 1 | (l, o) | 4 | `lo` | 21 |
| 2 | (lo, w) | 4 | `low` | 17 |
| 3 | (low, e) | 3 | `lowe` | 14 |
| 4 | (space, s) | 2 | `␠s` | 12 |
| 5 | (s, t) | 2 | `st` | 10 |
| 6 | (lowe, st) | 2 | `lowest` | 8 |

Gotcha I hit while checking this: the tie-break must compare **token ids**, not byte
strings. At step 5 both `(s, t)` and `(lowe, s)` have count 2. By id, `(115, 116)` <
`(258, 115)`, so `(s, t)` wins. Comparing byte strings instead picks `(lowe, s)` and
produces a different (still valid, but different) vocabulary.
