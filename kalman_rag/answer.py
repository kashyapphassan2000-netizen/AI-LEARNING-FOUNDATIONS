"""PRODUCTION rerank + answer step (needs Cohere + Anthropic API keys).

Rerank narrows 20 candidates to the 5 most relevant with a cross-encoder, then
Claude writes an answer grounded ONLY in those chunks, with citations. Needs
COHERE_API_KEY and ANTHROPIC_API_KEY. Both clients can be injected, which is how
the tests exercise this module offline.
"""
from __future__ import annotations
import os


def rerank(query: str, candidates: list[dict], top_n: int = 5, client=None) -> list[dict]:
    """Cohere cross-encoder rerank. A cross-encoder reads query+doc together, so
    it is more accurate than the bi-encoder used for the first-stage search."""
    if client is None:
        import cohere
        client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    docs = [c["body"] for c in candidates]
    res = client.rerank(model="rerank-english-v3.0", query=query,
                        documents=docs, top_n=top_n)
    return [candidates[r.index] for r in res.results]


GROUNDED_PROMPT = """You are a control-theory tutor. Answer the question using \
ONLY the context below. Cite every claim as [source p.N]. If the context does \
not contain the answer, say so -- never invent facts.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source']} p.{c['page']}] {c['body']}" for c in chunks)
    return GROUNDED_PROMPT.format(context=context, question=question)


def answer(question: str, chunks: list[dict], client=None) -> str:
    """Ask Claude for a grounded, cited answer."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}])
    if msg.stop_reason == "refusal":
        return "The model declined to answer this question."
    return "".join(b.text for b in msg.content if b.type == "text")


def to_candidates(retrieved) -> list[dict]:
    """Adapt HybridRetriever output [(Chunk, score)] to the dicts used here."""
    return [{"source": c.source, "page": c.page, "body": c.text, "score": s}
            for c, s in retrieved]
