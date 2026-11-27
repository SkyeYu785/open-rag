"""Document upload / listing / chunk inspection."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile

from .. import db
from ..config import get_settings
from ..models import ChunkPage, DocumentOut, UploadResponse
from ..services import ingestion
from ..services.parsing import SUPPORTED_EXTENSIONS
from .serializers import document_payload, get_kb_or_404

router = APIRouter(tags=["documents"])

MAX_FILES_PER_REQUEST = 20


def _sanitize(name: str) -> str:
    cleaned = Path(name).name.replace("\x00", "").strip()
    return cleaned or "untitled"


@router.get("/knowledge-bases/{kb_id}/documents", response_model=list[DocumentOut])
def list_documents(kb_id: str) -> list[dict]:
    get_kb_or_404(kb_id)
    rows = db.query_all(
        "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at DESC", (kb_id,)
    )
    return [document_payload(row) for row in rows]


@router.post("/knowledge-bases/{kb_id}/documents", response_model=UploadResponse, status_code=202)
async def upload_documents(
    kb_id: str,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> dict:
    get_kb_or_404(kb_id)
    settings = get_settings()
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_FILES_PER_REQUEST} 个文件")

    accept: list[dict] = []
    rejected: list[dict[str, str]] = []

    for upload in files:
        name = _sanitize(upload.filename or "untitled")
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            rejected.append({"name": name, "reason": f"不支持的文件类型 {ext or '（无扩展名）'}"})
            continue

        data = await upload.read()
        if not data:
            rejected.append({"name": name, "reason": "文件为空"})
            continue
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            rejected.append({"name": name, "reason": f"超过大小限制（{settings.max_upload_mb}MB）"})
            continue

        doc_id = uuid.uuid4().hex
        path = settings.upload_dir / f"{doc_id}{ext}"
        path.write_bytes(data)

        timestamp = db.now()
        db.execute(
            """
            INSERT INTO documents
                (id, kb_id, name, path, ext, size, status, error, chunk_count, char_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, 0, 0, ?, ?)
            """,
            (doc_id, kb_id, name, str(path), ext, len(data), timestamp, timestamp),
        )
        background.add_task(ingestion.process_document, doc_id)
        accept.append(document_payload(db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))))

    return {"accepted": accept, "rejected": rejected}


@router.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str) -> dict:
    row = db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document_payload(row)


@router.post("/documents/{doc_id}/reindex", response_model=DocumentOut, status_code=202)
def reindex_document(doc_id: str, background: BackgroundTasks) -> dict:
    row = db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.execute(
        "UPDATE documents SET status = 'pending', error = NULL, updated_at = ? WHERE id = ?",
        (db.now(), doc_id),
    )
    background.add_task(ingestion.process_document, doc_id)
    return document_payload(db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,)))


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    if not ingestion.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")


@router.get("/documents/{doc_id}/chunks", response_model=ChunkPage)
def list_chunks(
    doc_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict:
    if db.query_one("SELECT id FROM documents WHERE id = ?", (doc_id,)) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    total = db.query_one(
        "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ? AND level = 'child'", (doc_id,)
    )["n"]
    rows = db.query_all(
        "SELECT id, doc_id, ordinal, text, page, token_count AS tokens, level,"
        " start_offset, end_offset FROM chunks"
        " WHERE doc_id = ? AND level = 'child' ORDER BY ordinal LIMIT ? OFFSET ?",
        (doc_id, limit, offset),
    )
    return {"total": total, "items": rows}
