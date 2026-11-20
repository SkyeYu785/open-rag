"""Small helpers shared by the API routers."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from .. import db


def get_kb_or_404(kb_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return row


def kb_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row.get("description") or "",
        "document_count": row.get("document_count", 0),
        "chunk_count": row.get("chunk_count", 0),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def document_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kb_id": row["kb_id"],
        "name": row["name"],
        "ext": row["ext"],
        "size": row["size"],
        "status": row["status"],
        "error": row.get("error"),
        "chunk_count": row["chunk_count"],
        "char_count": row["char_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def dataset_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kb_id": row["kb_id"],
        "name": row["name"],
        "description": row.get("description") or "",
        "case_count": row.get("case_count", 0),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def case_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "question": row["question"],
        "reference_doc_ids": db.loads(row["reference_docs"], []),
        "keywords": db.loads(row["keywords"], []),
        "created_at": row["created_at"],
    }
