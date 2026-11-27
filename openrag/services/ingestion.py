"""Document ingestion pipeline: parse -> chunk (parent/child) -> embed -> store."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np

from .. import db
from ..config import get_settings
from . import retrieval
from .chunking import Chunk, chunk_sections
from .embedding import get_embedder
from .parsing import ParseError, parse_file
from .tokens import count_tokens

logger = logging.getLogger(__name__)


def assert_embedding_compatible(kb_id: str, signature: str) -> None:
    row = db.query_one(
        "SELECT embedding_sig FROM chunks WHERE kb_id = ? AND level = 'child' LIMIT 1", (kb_id,)
    )
    if row and row["embedding_sig"] and row["embedding_sig"] != signature:
        raise RuntimeError(
            f"该知识库已使用 {row['embedding_sig']} 建立索引，与当前模型 {signature} 不一致。"
            "请新建知识库后重新上传文档。"
        )


def process_document(doc_id: str) -> None:
    """Parse, chunk and embed one document. Safe to call from a worker thread."""
    document = db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))
    if document is None:
        return

    kb_id = document["kb_id"]
    settings = get_settings()
    db.execute(
        "UPDATE documents SET status = 'processing', error = NULL, updated_at = ? WHERE id = ?",
        (db.now(), doc_id),
    )

    try:
        path = Path(document["path"])
        if not path.exists():
            raise ParseError("文件不存在，可能已被删除")

        parsed = parse_file(path, document["ext"])
        chunks = chunk_sections(
            parsed.sections,
            child_tokens=settings.child_tokens,
            parent_tokens=settings.parent_tokens,
            overlap_tokens=settings.overlap_tokens,
        )
        parents = [chunk for chunk in chunks if chunk.level == "parent"]
        children = [chunk for chunk in chunks if chunk.level == "child"]
        if not parents or not children:
            raise ParseError("没有生成任何文本分块，请检查文件内容")

        embedder = get_embedder()
        assert_embedding_compatible(kb_id, embedder.signature)

        # parents first: children reference them through parent_id
        parent_ids = {chunk.key: uuid.uuid4().hex for chunk in parents if chunk.key}
        parent_rows = [
            _chunk_row(parent_ids[chunk.key], kb_id, doc_id, ordinal, chunk, embedding=None, signature="")
            for ordinal, chunk in enumerate(parents)
        ]
        matrix = embedder.embed_documents([chunk.text for chunk in children])
        if len(matrix) != len(children):
            raise RuntimeError("embedding 返回的向量数量与分块数量不一致")
        child_rows = [
            _chunk_row(
                uuid.uuid4().hex,
                kb_id,
                doc_id,
                ordinal,
                chunk,
                embedding=matrix[ordinal],
                signature=embedder.signature,
                parent_id=parent_ids.get(chunk.parent_key or ""),
            )
            for ordinal, chunk in enumerate(children)
        ]

        with db.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks
                    (id, kb_id, doc_id, ordinal, level, parent_id, text, page, token_count,
                     start_offset, end_offset, embedding, dim, meta, embedding_sig)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                parent_rows + child_rows,
            )
            conn.execute(
                """
                UPDATE documents
                SET status = 'ready', error = NULL, chunk_count = ?, char_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    len(children),
                    sum(len(chunk.text) for chunk in children),
                    db.now(),
                    doc_id,
                ),
            )
            conn.execute(
                "UPDATE knowledge_bases SET version = version + 1, updated_at = ? WHERE id = ?",
                (db.now(), kb_id),
            )

        retrieval.invalidate(kb_id)
        logger.info(
            "indexed %s: %d parents, %d children", document["name"], len(parents), len(children)
        )

    except Exception as exc:  # noqa: BLE001 - surfaced to the UI through documents.error
        logger.warning("failed to index %s: %s", document["name"], exc)
        db.execute(
            "UPDATE documents SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
            (str(exc)[:1000], db.now(), doc_id),
        )


def _chunk_row(
    chunk_id: str,
    kb_id: str,
    doc_id: str,
    ordinal: int,
    chunk: Chunk,
    embedding: np.ndarray | None,
    signature: str,
    parent_id: str | None = None,
) -> tuple:
    vector = None if embedding is None else np.asarray(embedding, dtype=np.float32)
    return (
        chunk_id,
        kb_id,
        doc_id,
        ordinal,
        chunk.level,
        parent_id,
        chunk.text,
        chunk.page,
        count_tokens(chunk.text),
        chunk.start_offset,
        chunk.end_offset,
        None if vector is None else vector.tobytes(),
        0 if vector is None else int(vector.shape[0]),
        json.dumps(chunk.meta, ensure_ascii=False),
        signature,
    )


def delete_document(doc_id: str) -> bool:
    document = db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))
    if document is None:
        return False

    path = Path(document["path"])
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best effort cleanup
        logger.warning("could not remove %s", path)

    db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    db.bump_kb_version(document["kb_id"])
    retrieval.invalidate(document["kb_id"])
    return True
