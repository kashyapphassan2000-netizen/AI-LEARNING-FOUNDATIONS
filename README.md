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
