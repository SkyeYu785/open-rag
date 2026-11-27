"""Hybrid retrieval over child chunks, with parent blocks as context.

Children are indexed (dense + BM25, fused with Reciprocal Rank Fusion); the API
returns the matched child plus its parent block, which is what a generator
should actually read (small-to-big retrieval).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .. import db
from ..config import get_settings
from .bm25 import BM25Index
from .embedding import get_embedder
from .rerank import get_reranker, reranker_status, validate_scores

logger = logging.getLogger(__name__)

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


@dataclass
class Filters:
    min_score: float
    relative_cutoff: float
    max_per_doc: int
    dedupe_parent: bool


def _resolve_filters(
    min_score: float | None,
    relative_cutoff: float | None,
    max_per_doc: int | None,
    dedupe_parent: bool | None,
) -> Filters:
    settings = get_settings()
    return Filters(
        min_score=settings.min_score if min_score is None else max(0.0, min_score),
        relative_cutoff=(
            settings.relative_cutoff
            if relative_cutoff is None
            else min(1.0, max(0.0, relative_cutoff))
        ),
        max_per_doc=settings.max_per_doc if max_per_doc is None else max(0, max_per_doc),
        dedupe_parent=settings.dedupe_parent if dedupe_parent is None else dedupe_parent,
    )


def _apply_filters(hits: list[dict], top_k: int, filters: Filters, score_key: str) -> tuple[list[dict], int]:
    """Drop irrelevant tail, de-duplicate parents/documents, then truncate.

    Returns the kept hits and how many were rejected by the filters (hits cut
    off by ``top_k`` are truncation, not rejection).
    """
    if not hits:
        return [], 0

    top_score = max((hit["scores"].get(score_key, 0.0) for hit in hits), default=0.0)
    cutoff = top_score * filters.relative_cutoff if top_score > 0 else 0.0

    kept: list[dict] = []
    dropped = 0
    parent_seen: set[str] = set()
    per_doc: dict[str, int] = {}

    for hit in hits:
        if len(kept) >= top_k:
            break

        score = hit["scores"].get(score_key, 0.0)
        if filters.min_score and score < filters.min_score:
            dropped += 1
            continue
        if filters.relative_cutoff and cutoff and score < cutoff:
            dropped += 1
            continue

        parent_id = hit.get("parent_id")
        if filters.dedupe_parent and parent_id:
            if parent_id in parent_seen:
                dropped += 1
                continue
            parent_seen.add(parent_id)

        doc_id = hit["doc_id"]
        if filters.max_per_doc and per_doc.get(doc_id, 0) >= filters.max_per_doc:
            dropped += 1
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1

        kept.append(hit)

    return kept, dropped


def retrieve(
    kb_id: str,
    query: str,
    top_k: int = 6,
    mode: str = "hybrid",
    *,
    index: KBIndex | None = None,
    include_context: bool = True,
    min_score: float | None = None,
    relative_cutoff: float | None = None,
    max_per_doc: int | None = None,
    dedupe_parent: bool | None = None,
) -> tuple[list[dict], int, float, dict]:
    """Return ``(hits, total_child_chunks, latency_ms, stats)``."""
    started = time.perf_counter()
    index = index if index is not None else get_index(kb_id)
    if index.size == 0:
        return [], 0, (time.perf_counter() - started) * 1000, {
            "recalled": 0,
            "returned": 0,
            "dropped": 0,
            "rerank_applied": False,
            "rerank_warning": None,
            "filter_score": "",
        }

    query_vector = get_embedder().embed_query(query)
    query_vector = np.asarray(query_vector, dtype=np.float32)
    norm = float(np.linalg.norm(query_vector))
    if norm:
        query_vector = query_vector / norm
    if query_vector.size != index.dim:
        raise IndexMismatch("查询向量维度与索引维度不一致，请检查 embedding 模型配置")

    filters = _resolve_filters(min_score, relative_cutoff, max_per_doc, dedupe_parent)

    reranker = None
    rerank_warning: str | None = None
    try:
        candidate = get_reranker()
    except Exception as exc:  # noqa: BLE001 - a broken rerank config must not break search
        candidate = None
        rerank_warning = f"reranker 配置错误：{exc}"
        logger.warning("reranker unavailable, falling back to hybrid ranking: %s", exc)

    if candidate is not None:
        usable, warning = reranker_status(candidate)
        if usable:
            reranker = candidate
        else:
            rerank_warning = warning
            logger.warning("reranker self-test failed, falling back to hybrid ranking: %s", warning)

    vector_scores = index.matrix @ query_vector
    keyword_scores = index.bm25.scores(query)

    recall_limit = max(top_k, get_settings().rerank_candidates) if reranker else max(top_k * 4, 40)
    recall_limit = min(recall_limit, index.size)

    vector_ranked = _top_indices(vector_scores, recall_limit)
    keyword_ranked = _top_indices(keyword_scores, recall_limit)

    if mode == "vector":
        candidates = vector_ranked[:recall_limit]
        fused: dict[int, float] = {i: score for i, score in candidates}
    elif mode == "keyword":
        candidates = keyword_ranked[:recall_limit]
        fused = {i: score for i, score in candidates}
    else:
        scores: dict[int, float] = {}
        for rank, (i, _) in enumerate(vector_ranked):
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (i, _) in enumerate(keyword_ranked):
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:recall_limit]
        fused = dict(candidates)

    hits: list[dict] = []
    for i, score in candidates:
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
                "scores": {
                    "vector": float(vector_scores[i]),
                    "keyword": float(keyword_scores[i]),
                    "fused": float(fused.get(i, score)),
                },
            }
        )

    rerank_applied = False
    if reranker is not None and hits:
        try:
            rerank_scores = reranker.rerank(query, [hit["text"] for hit in hits])
            validate_scores(rerank_scores)
            for hit, rerank_score in zip(hits, rerank_scores):
                hit["scores"]["rerank"] = float(rerank_score)
            hits.sort(key=lambda hit: hit["scores"].get("rerank", -1e9), reverse=True)
            rerank_applied = True
        except Exception as exc:  # noqa: BLE001 - degrade to hybrid ranking
            rerank_warning = str(exc)
            logger.warning("rerank failed, keeping hybrid order: %s", exc)

    if rerank_applied:
        score_key = "rerank"
    elif mode == "keyword":
        score_key = "keyword"
    else:
        score_key = "vector"

    recalled = len(hits)
    selected, dropped = _apply_filters(hits, top_k, filters, score_key)
    for rank, hit in enumerate(selected, start=1):
        hit["rank"] = rank
        hit["relevance"] = hit["scores"].get(score_key, 0.0)

    stats = {
        "recalled": recalled,
        "returned": len(selected),
        "dropped": dropped,
        "rerank_applied": rerank_applied,
        "rerank_warning": rerank_warning,
        "filter_score": score_key,
    }
    return selected, index.size, (time.perf_counter() - started) * 1000, stats
