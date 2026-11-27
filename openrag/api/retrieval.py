"""Retrieval endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import db
from ..models import RetrieveRequest, RetrieveResponse
from ..services import retrieval
from .serializers import get_kb_or_404

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest) -> dict:
    get_kb_or_404(payload.kb_id)
    try:
        hits, total, latency_ms, stats = retrieval.retrieve(
            kb_id=payload.kb_id,
            query=payload.query,
            top_k=payload.top_k,
            mode=payload.mode,
            include_context=payload.include_context,
            min_score=payload.min_score,
            relative_cutoff=payload.relative_cutoff,
            max_per_doc=payload.max_per_doc,
            dedupe_parent=payload.dedupe_parent,
        )
    except retrieval.IndexMismatch as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "query": payload.query,
        "mode": payload.mode,
        "kb_id": payload.kb_id,
        "total_chunks": total,
        "latency_ms": round(latency_ms, 2),
        "recalled": stats["recalled"],
        "returned": stats["returned"],
        "dropped": stats["dropped"],
        "rerank_applied": stats["rerank_applied"],
        "rerank_warning": stats.get("rerank_warning"),
        "filter_score": stats["filter_score"],
        "hits": hits,
    }
