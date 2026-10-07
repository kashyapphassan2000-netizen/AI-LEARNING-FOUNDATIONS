"""Goldset evaluation: does the right source rank #1?

Each question is paired with the note that should answer it. We report top-1
accuracy, the rank of the gold source, and mean reciprocal rank (MRR). A second,
harder set uses paraphrases that share few words with the notes -- the case
where lexical/TF-IDF retrieval is expected to struggle and BGE-M3 should help.

Run: PYTHONPATH=.:data python eval/goldset.py
"""
from corpus import CORPUS
from kalman_rag.retrieve import chunk_documents, HybridRetriever

GOLDSET = [
    ("derive the Kalman gain", "notes/kalman_gain"),
    ("two steps predict and update", "notes/kalman_basics"),
    ("state transition and measurement matrix", "notes/state_space"),
    ("reconstruct internal state from outputs", "notes/observability"),
    ("how to tune process noise and measurement noise", "notes/noise_covariance"),
    ("drive the system to any state with an input", "notes/controllability"),
    ("Riccati equation optimal feedback", "notes/lqr"),
    ("integral windup and derivative noise", "notes/pid"),
    ("nonlinear measurements Jacobian linearization", "notes/ekf"),
    ("eigenvalues inside the unit circle", "notes/stability"),
]

PARAPHRASES = [
    ("how much should I trust the sensor versus the model?", "notes/kalman_gain"),
    ("my estimate is too jumpy, what knob makes it smoother?", "notes/noise_covariance"),
    ("can I figure out hidden variables just by watching the sensors?", "notes/observability"),
    ("my controller keeps overshooting the target", "notes/pid"),
    ("will this system blow up or settle down?", "notes/stability"),
]


def gold_rank(retriever, question, gold, k):
    sources = [c.source for c, _ in retriever.retrieve(question, k=k)]
    return sources.index(gold) + 1 if gold in sources else None


def run(retriever, items, title):
    k = len(retriever.chunks)
    print(f"\n{title}")
    hits, rr = 0, 0.0
    for q, gold in items:
        rank = gold_rank(retriever, q, gold, k)
        top = retriever.retrieve(q, k=1)[0][0].source
        ok = top == gold
        hits += ok
        rr += 1.0 / rank if rank else 0.0
        print(f"  {'OK  ' if ok else 'MISS'} {q[:52]:<52} -> {top:<24} (gold rank {rank})")
    n = len(items)
    print(f"top-1 accuracy: {hits}/{n} = {hits / n:.0%}   MRR: {rr / n:.3f}")
    return hits, n


def main():
    retriever = HybridRetriever(chunk_documents(CORPUS))
    run(retriever, GOLDSET, "goldset (keyword-style questions)")
    run(retriever, PARAPHRASES, "paraphrase set (few shared words; hard for TF-IDF/BM25)")


if __name__ == "__main__":
    main()
