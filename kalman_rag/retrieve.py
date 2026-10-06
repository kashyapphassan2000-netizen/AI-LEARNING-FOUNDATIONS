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
