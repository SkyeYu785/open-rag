"""Hybrid retrieval over child chunks, with parent blocks as context.

Children are indexed (dense + BM25, fused with Reciprocal Rank Fusion); the API
returns the matched child plus its parent block, which is what a generator
should actually read (small-to-big retrieval).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np

from .. import db
from .bm25 import BM25Index
from .embedding import get_embedder

RRF_K = 60


class IndexMismatch(RuntimeError):
    """Raised when stored vectors were produced by a different embedder."""


@dataclass
class KBIndex:
    version: int
    size: int
    dim: int
    chunk_ids: list[str]
    texts: list[str]
    doc_ids: list[str]
    doc_names: list[str]
    ordinals: list[int]
    pages: list[int | None]
    tokens: list[int]
    starts: list[int | None]
    ends: list[int | None]
    metas: list[dict]
    parent_ids: list[str | None]
    parent_texts: list[str | None]
    matrix: np.ndarray
    bm25: BM25Index


_cache: dict[str, KBIndex] = {}
_cache_lock = threading.Lock()


def invalidate(kb_id: str | None = None) -> None:
    with _cache_lock:
        if kb_id is None:
            _cache.clear()
        else:
            _cache.pop(kb_id, None)


def _build(kb_id: str, version: int) -> KBIndex:
    rows = db.query_all(
        """
        SELECT c.id, c.doc_id, c.ordinal, c.text, c.page, c.token_count,
               c.start_offset, c.end_offset, c.meta, c.parent_id,
               c.embedding, c.dim, c.embedding_sig,
               d.name AS doc_name, p.text AS parent_text
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        LEFT JOIN chunks p ON p.id = c.parent_id
        WHERE c.kb_id = ? AND c.level = 'child'
        ORDER BY c.doc_id, c.ordinal
        """,
        (kb_id,),
    )

    embedder = get_embedder()
    current = embedder.signature
    stored = {row["embedding_sig"] for row in rows if row["embedding_sig"]}
    if stored and current not in stored:
        raise IndexMismatch(
            f"知识库索引由 {sorted(stored)[0]} 生成，当前 embedding 配置为 {current}。"
            "请新建知识库重新导入文档，或清空该知识库后重新上传。"
        )

    if not rows:
        return KBIndex(
            version=version,
            size=0,
            dim=0,
            chunk_ids=[],
            texts=[],
            doc_ids=[],
            doc_names=[],
            ordinals=[],
            pages=[],
            tokens=[],
            starts=[],
            ends=[],
            metas=[],
            parent_ids=[],
            parent_texts=[],
            matrix=np.zeros((0, 0), dtype=np.float32),
            bm25=BM25Index([]),
        )

    dim = int(rows[0]["dim"]) or embedder.dim
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    for index, row in enumerate(rows):
        if row["embedding"]:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size == dim:
                matrix[index] = vector

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    texts = [row["text"] for row in rows]
    metas = []
    for row in rows:
        try:
            metas.append(json.loads(row["meta"]) if row["meta"] else {})
        except (TypeError, ValueError):
            metas.append({})

    return KBIndex(
        version=version,
        size=len(rows),
        dim=dim,
        chunk_ids=[row["id"] for row in rows],
        texts=texts,
        doc_ids=[row["doc_id"] for row in rows],
        doc_names=[row["doc_name"] for row in rows],
        ordinals=[int(row["ordinal"]) for row in rows],
        pages=[row["page"] for row in rows],
        tokens=[int(row["token_count"] or 0) for row in rows],
        starts=[row["start_offset"] for row in rows],
        ends=[row["end_offset"] for row in rows],
        metas=metas,
        parent_ids=[row["parent_id"] for row in rows],
        parent_texts=[row["parent_text"] for row in rows],
        matrix=matrix,
        bm25=BM25Index(texts),
    )


def get_index(kb_id: str) -> KBIndex:
    kb = db.query_one("SELECT version FROM knowledge_bases WHERE id = ?", (kb_id,))
    if kb is None:
        raise KeyError(kb_id)
    with _cache_lock:
        cached = _cache.get(kb_id)
        if cached is not None and cached.version == kb["version"]:
            return cached
    index = _build(kb_id, int(kb["version"]))
    with _cache_lock:
        _cache[kb_id] = index
    return index


def _top_indices(scores: np.ndarray, limit: int) -> list[tuple[int, float]]:
    if scores.size == 0 or limit <= 0:
        return []
    limit = min(limit, scores.size)
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    ordered = candidates[np.argsort(-scores[candidates])]
    return [(int(i), float(scores[i])) for i in ordered if scores[i] > 0]


def retrieve(
    kb_id: str,
    query: str,
    top_k: int = 6,
    mode: str = "hybrid",
    *,
    index: KBIndex | None = None,
    include_context: bool = True,
) -> tuple[list[dict], int, float]:
    """Return ``(hits, total_child_chunks, latency_ms)``."""
    started = time.perf_counter()
    index = index if index is not None else get_index(kb_id)
    if index.size == 0:
        return [], 0, (time.perf_counter() - started) * 1000

    query_vector = get_embedder().embed_query(query)
    query_vector = np.asarray(query_vector, dtype=np.float32)
    norm = float(np.linalg.norm(query_vector))
    if norm:
        query_vector = query_vector / norm
    if query_vector.size != index.dim:
        raise IndexMismatch("查询向量维度与索引维度不一致，请检查 embedding 模型配置")

    vector_scores = index.matrix @ query_vector
    keyword_scores = index.bm25.scores(query)

    pool = min(index.size, max(top_k * 4, 40))
    vector_ranked = _top_indices(vector_scores, pool)
    keyword_ranked = _top_indices(keyword_scores, pool)

    if mode == "vector":
        ranked = vector_ranked[:top_k]
        fused: dict[int, float] = {i: score for i, score in ranked}
    elif mode == "keyword":
        ranked = keyword_ranked[:top_k]
        fused = {i: score for i, score in ranked}
    else:
        scores: dict[int, float] = {}
        for rank, (i, _) in enumerate(vector_ranked):
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (i, _) in enumerate(keyword_ranked):
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(((i, s) for i, s in scores.items()), key=lambda item: item[1], reverse=True)[:top_k]
        fused = dict(ranked)

    hits: list[dict] = []
    for rank, (i, score) in enumerate(ranked, start=1):
        hits.append(
            {
                "chunk_id": index.chunk_ids[i],
                "doc_id": index.doc_ids[i],
                "doc_name": index.doc_names[i],
                "ordinal": index.ordinals[i],
                "text": index.texts[i],
                "page": index.pages[i],
                "parent_id": index.parent_ids[i],
                "context": index.parent_texts[i] if include_context else None,
                "tokens": index.tokens[i],
                "meta": index.metas[i],
                "start_offset": index.starts[i],
                "end_offset": index.ends[i],
                "score": float(fused.get(i, score)),
                "rank": rank,
                "scores": {
                    "vector": float(vector_scores[i]),
                    "keyword": float(keyword_scores[i]),
                    "fused": float(fused.get(i, score)),
                },
            }
        )

    return hits, index.size, (time.perf_counter() - started) * 1000
