"""Knowledge base CRUD."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import db
from ..models import KnowledgeBaseCreate, KnowledgeBaseOut, KnowledgeBaseUpdate
from .serializers import get_kb_or_404, kb_summary

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])

_SUMMARY_SQL = """
SELECT kb.*,
       (SELECT COUNT(*) FROM documents d WHERE d.kb_id = kb.id) AS document_count,
       (SELECT COUNT(*) FROM chunks c WHERE c.kb_id = kb.id) AS chunk_count
FROM knowledge_bases kb
"""


@router.get("", response_model=list[KnowledgeBaseOut])
def list_knowledge_bases() -> list[dict]:
    rows = db.query_all(_SUMMARY_SQL + " ORDER BY kb.created_at DESC")
    return [kb_summary(row) for row in rows]


@router.post("", response_model=KnowledgeBaseOut, status_code=201)
def create_knowledge_base(payload: KnowledgeBaseCreate) -> dict:
    kb_id = uuid.uuid4().hex
    timestamp = db.now()
    try:
        db.execute(
            "INSERT INTO knowledge_bases (id, name, description, version, created_at, updated_at)"
            " VALUES (?, ?, ?, 0, ?, ?)",
            (kb_id, payload.name.strip(), payload.description.strip(), timestamp, timestamp),
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="同名知识库已存在") from exc
    return kb_summary(get_kb_or_404(kb_id) | {"document_count": 0, "chunk_count": 0})


@router.get("/{kb_id}", response_model=KnowledgeBaseOut)
def get_knowledge_base(kb_id: str) -> dict:
    row = db.query_one(_SUMMARY_SQL + " WHERE kb.id = ?", (kb_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb_summary(row)


@router.patch("/{kb_id}", response_model=KnowledgeBaseOut)
def update_knowledge_base(kb_id: str, payload: KnowledgeBaseUpdate) -> dict:
    get_kb_or_404(kb_id)
    updates: list[str] = []
    params: list[str] = []
    if payload.name is not None:
        updates.append("name = ?")
        params.append(payload.name.strip())
    if payload.description is not None:
        updates.append("description = ?")
        params.append(payload.description.strip())
    if updates:
        updates.append("updated_at = ?")
        params.append(db.now())
        params.append(kb_id)
        try:
            db.execute(f"UPDATE knowledge_bases SET {', '.join(updates)} WHERE id = ?", params)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="同名知识库已存在") from exc
    return get_knowledge_base(kb_id)


@router.delete("/{kb_id}", status_code=204)
def delete_knowledge_base(kb_id: str) -> None:
    get_kb_or_404(kb_id)
    paths = [row["path"] for row in db.query_all("SELECT path FROM documents WHERE kb_id = ?", (kb_id,))]
    db.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
    for raw in paths:
        try:
            Path(raw).unlink(missing_ok=True)
        except OSError:
            pass
