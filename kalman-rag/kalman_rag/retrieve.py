"""The retrieval core of kalman-rag, runnable fully offline.

Production kalman-rag uses BGE-M3 neural embeddings in pgvector and a Cohere
cross-encoder reranker. Those need GPU/network/API keys. To make the *mechanics*
verifiable offline, this module stands in TF-IDF for the dense embedding (both
turn text into vectors you compare by cosine similarity) and keeps the exact
same hybrid + fusion logic the production system uses. The interfaces match, so
swapping TF-IDF for BGE-M3 later is a one-line change.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def chunk_documents(corpus, max_words: int = 60, overlap: int = 12) -> List[Chunk]:
    """Split each document into overlapping word windows.

    Overlap means a sentence cut at a boundary still appears whole in the next
    chunk, so retrieval never loses context that straddles a split.
    """
    chunks: List[Chunk] = []
    for doc in corpus:
        words = doc["text"].split()
        if len(words) <= max_words:
            chunks.append(Chunk(doc["text"], doc["source"], doc["page"]))
            continue
        step = max_words - overlap
        for start in range(0, len(words), step):
            window = words[start:start + max_words]
            if not window:
                break
            chunks.append(Chunk(" ".join(window), doc["source"], doc["page"]))
    return chunks


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """Dense (TF-IDF cosine) + lexical (BM25), fused by Reciprocal Rank Fusion."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        # dense: TF-IDF vectors (stand-in for BGE-M3 embeddings)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)
        # lexical: BM25 over tokenized text
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def _dense_ranking(self, query: str) -> List[int]:
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix)[0]
        return list(np.argsort(scores)[::-1])

    def _lexical_ranking(self, query: str) -> List[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return list(np.argsort(scores)[::-1])

    def retrieve(self, query: str, k: int = 5, rrf_k: int = 60):
        """Reciprocal Rank Fusion: each retriever votes by 1/(rrf_k + rank)."""
        dense = self._dense_ranking(query)
        lexical = self._lexical_ranking(query)
        fused = {}
        for ranking in (dense, lexical):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        return [(self.chunks[i], fused[i]) for i in order]


def grounded_answer(query: str, retrieved) -> str:
    """A placeholder for the LLM step.

    Production calls Claude with the retrieved chunks and asks for a cited
    answer. Offline, we return the top chunks with their citations so you can
    see exactly what would be fed to the model -- and verify it is grounded.
    """
    lines = [f"Question: {query}", "", "Grounded context (what the LLM would cite):"]
    for chunk, score in retrieved:
        lines.append(f"  [{chunk.source} p.{chunk.page}] (score {score:.4f})")
        lines.append(f"     {chunk.text[:160]}...")
    return "\n".join(lines)
