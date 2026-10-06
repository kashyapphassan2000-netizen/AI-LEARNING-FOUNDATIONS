# kalman-rag

**A niche Retrieval-Augmented Generation system that cites its sources** — Project 04 of 12 (ZERO_TO_ANTHROPIC).

RAG makes a language model answer from *your* documents instead of from memory.
kalman-rag does it over control-theory notes: chunk → dense + lexical search →
Reciprocal Rank Fusion → (rerank) → a grounded, cited answer from Claude.

The retrieval core runs fully offline (TF-IDF stands in for neural embeddings). The
production pieces — BGE-M3 embeddings in pgvector, Cohere rerank, Claude answering —
are included as reference code and need your API keys / a database.

Stack: Python 3.12, scikit-learn, rank_bm25, NumPy, Gradio; production: FlagEmbedding, pgvector, Cohere, Anthropic. Cost: $0 offline.

---

## Build log — the story, day by day

### Day 0 — setup

```bash
mkdir kalman-rag && cd kalman-rag
python3 -m venv .venv && source .venv/bin/activate
pip install rank_bm25 scikit-learn numpy gradio
mkdir -p kalman_rag data eval tests app docs && touch kalman_rag/__init__.py
git init && git add -A && git commit -m 'chore: scaffold kalman-rag'
```
Production extras, used only with keys: `pip install FlagEmbedding cohere anthropic psycopg2-binary pgvector`.

### Day 1 (Monday) — build the corpus (`data/corpus.py`)

**Goal:** a document collection that is (a) about a domain I know, (b) legally
clean for a public repo, and (c) big enough that retrieval has to work for it.

**The sourcing decision.** The real project would ingest public-domain control
texts and NASA technical reports. Modern textbook PDFs "found online" are not
acceptable in a public repo. For a runnable, copyright-clean demo I wrote **10 original
notes** myself, one topic each, with a `source` label and a page-like id so answers
can cite `[notes/kalman_gain p.2]`:

| page | source | topic |
|---|---|---|
| 1 | notes/kalman_basics | what the Kalman filter is; predict + update |
| 2 | notes/kalman_gain | how the gain is derived and what it trades off |
| 3 | notes/state_space | state transition A, input B, measurement H |
| 4 | notes/noise_covariance | tuning Q and R |
| 5 | notes/observability | observability matrix, rank test |
| 6 | notes/controllability | controllability matrix, duality |
| 7 | notes/lqr | LQR, Riccati equation, LQG duality |
| 8 | notes/pid | P, I, D terms; windup; noise |
| 9 | notes/ekf | extended / unscented Kalman filter |
| 10 | notes/stability | eigenvalue tests, Lyapunov |

601 words total; each note is 51–66 words. I made the topics **deliberately
overlap** (Kalman gain vs. noise covariance vs. LQR all talk about "gain", Q and R;
observability and state-space both mention the measurement and transition matrices)
so that retrieval faces real ambiguity instead of a toy where every query has one
obvious keyword.

**Commit:** `feat: original control-theory corpus (10 notes)`

### Day 2 (Tuesday) — chunking (`kalman_rag/retrieve.py`, part 1)

**Goal:** split documents into pieces small enough to embed and retrieve precisely.

You can't embed a whole textbook as one vector: one vector for 300 pages averages
every topic into mush. The choices:

| strategy | what it does | trade-off |
|---|---|---|
| fixed-size | every N words | simple; can cut mid-idea |
| sentence / semantic | split on meaning boundaries | cleaner; more work |
| overlap (10–20%) | neighbouring chunks share edge text | nothing lost at a cut; some duplication |

**What I wrote:** a `Chunk` dataclass (`text, source, page` — the citation travels
with the text), and `chunk_documents(corpus, max_words=60, overlap=12)`: documents of
≤ 60 words stay whole; longer ones are cut into 60-word windows that advance by
48 words, so consecutive chunks share 12 words (20%). Plus `_tokenize()`, a
lowercase alphanumeric tokenizer for BM25 tomorrow.

```text
10 docs -> 17 chunks
  [notes/kalman_basics p.1] 58 words: The Kalman filter is a recursive estimator for lin...
  [notes/kalman_gain p.2] 60 words: The Kalman gain determines how much the measuremen...
  [notes/kalman_gain p.2] 14 words: trusts the model prediction. Deriving the Kalman g...
  [notes/state_space p.3] 60 words: A discrete-time state-space model writes the next ...
  [notes/state_space p.3] 14 words: Kalman filter's predict step uses A and B, and its...
  ...
  [notes/stability p.10] 51 words: A continuous-time linear system is asymptotically ...
```

**What I noticed:** my notes are 51–66 words, just over the 60-word window, so
each long note produces a **tiny 14–18-word tail chunk that is mostly overlap** (12
of its 14 words are repeated from the previous chunk). That's harmless for retrieval
quality here (the tail still carries the right citation), but it means the same note
can occupy two of the top-5 slots. On a real corpus I'd merge tails shorter than
~half a window into the previous chunk. I kept the guide's settings so the numbers
are comparable, and wrote the issue down instead.

**Commit:** `feat: overlapping word-window chunker`
