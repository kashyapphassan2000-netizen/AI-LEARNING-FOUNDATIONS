"""PRODUCTION version of the dense retriever (needs network + a Postgres/pgvector
instance). The offline guide verifies the same logic with TF-IDF in retrieve.py;
this file is the real thing you run on your own machine. It is not executed in
CI (model download + database are not available there).
"""
from __future__ import annotations
import numpy as np

# pip install FlagEmbedding psycopg2-binary pgvector
# docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16


def embed_texts(texts: list[str]) -> np.ndarray:
    """BGE-M3 produces 1024-dim multilingual embeddings, good for technical text."""
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    out = model.encode(texts, batch_size=16, max_length=512)
    return out["dense_vecs"]


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            " id serial primary key, source text, page int,"
            " body text, embedding vector(1024));")
        # HNSW index for approximate nearest-neighbour search at scale
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops);")
    conn.commit()


def upsert_chunks(conn, chunks, vectors):
    with conn.cursor() as cur:
        for c, v in zip(chunks, vectors):
            cur.execute(
                "INSERT INTO chunks (source, page, body, embedding) "
                "VALUES (%s, %s, %s, %s)",
                (c.source, c.page, c.text, list(map(float, v))))
    conn.commit()


def vector_search(conn, query_vec, k=20):
    """pgvector cosine search: <=> is the cosine-distance operator."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source, page, body, 1 - (embedding <=> %s::vector) AS score "
            "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            (list(map(float, query_vec)), list(map(float, query_vec)), k))
        return cur.fetchall()
