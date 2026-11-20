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
        hits, total, latency_ms = retrieval.retrieve(
            kb_id=payload.kb_id,
            query=payload.query,
            top_k=payload.top_k,
            mode=payload.mode,
        )
    except retrieval.IndexMismatch as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "query": payload.query,
        "mode": payload.mode,
        "kb_id": payload.kb_id,
        "total_chunks": total,
        "latency_ms": round(latency_ms, 2),
        "hits": hits,
    }
