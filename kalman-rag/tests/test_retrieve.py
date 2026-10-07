from types import SimpleNamespace as NS
from corpus import CORPUS
from kalman_rag.retrieve import chunk_documents, HybridRetriever
from kalman_rag.answer import answer, build_prompt, to_candidates


def test_chunks_overlap_and_keep_citations():
    doc = {"source": "s", "page": 3, "text": " ".join(f"w{i}" for i in range(100))}
    chunks = chunk_documents([doc], max_words=60, overlap=12)
    assert len(chunks) == 3                                   # windows start at 0, 48, 96
    assert chunks[0].text.split()[-12:] == chunks[1].text.split()[:12]
    assert all(c.source == "s" and c.page == 3 for c in chunks)


def test_rrf_scores_follow_the_formula():
    r = HybridRetriever(chunk_documents(CORPUS))
    q = "derive the Kalman gain"
    dense, lexical = r._dense_ranking(q), r._lexical_ranking(q)
    for chunk, score in r.retrieve(q, k=5):
        i = r.chunks.index(chunk)
        assert abs(score - (1 / (60 + dense.index(i)) + 1 / (60 + lexical.index(i)))) < 1e-12


def test_kalman_gain_question_retrieves_kalman_gain_note():
    r = HybridRetriever(chunk_documents(CORPUS))
    top_chunk, _ = r.retrieve("derive the Kalman gain", k=1)[0]
    assert (top_chunk.source, top_chunk.page) == ("notes/kalman_gain", 2)


def test_grounded_prompt_carries_citations_and_refusal_rule():
    r = HybridRetriever(chunk_documents(CORPUS))
    cands = to_candidates(r.retrieve("derive the Kalman gain", k=3))
    prompt = build_prompt("derive the Kalman gain", cands)
    assert "[notes/kalman_gain p.2]" in prompt and "ONLY the context" in prompt

    class FakeClaude:
        messages = None
        def create(self, **kw):
            return NS(stop_reason="end_turn", content=[NS(type="text", text="cited [notes/kalman_gain p.2]")])
    fake = FakeClaude(); fake.messages = fake
    assert answer("q", cands, client=fake) == "cited [notes/kalman_gain p.2]"
