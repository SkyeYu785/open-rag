"""SQLite persistence layer.

Everything lives in a single SQLite file (metadata + chunk embeddings), so the
project can be started with zero external services. Embeddings are stored as
raw float32 blobs and are searched in memory by :mod:`openrag.services.retrieval`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT NOT NULL DEFAULT '',
    version       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    kb_id         TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    path          TEXT NOT NULL,
    ext           TEXT NOT NULL DEFAULT '',
    size          INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    char_count    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_id, created_at DESC);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    kb_id         TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    page          INTEGER,
    embedding     BLOB,
    dim           INTEGER NOT NULL DEFAULT 0,
    meta          TEXT NOT NULL DEFAULT '{}',
    embedding_sig TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(kb_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);

CREATE TABLE IF NOT EXISTS eval_datasets (
    id            TEXT PRIMARY KEY,
    kb_id         TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_kb ON eval_datasets(kb_id);

CREATE TABLE IF NOT EXISTS eval_cases (
    id              TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL REFERENCES eval_datasets(id) ON DELETE CASCADE,
    question        TEXT NOT NULL,
    reference_docs  TEXT NOT NULL DEFAULT '[]',
    keywords        TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_dataset ON eval_cases(dataset_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id            TEXT PRIMARY KEY,
    dataset_id    TEXT NOT NULL REFERENCES eval_datasets(id) ON DELETE CASCADE,
    kb_id         TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    config        TEXT NOT NULL DEFAULT '{}',
    metrics       TEXT NOT NULL DEFAULT '{}',
    error         TEXT,
    created_at    TEXT NOT NULL,
    finished_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON eval_runs(dataset_id, created_at DESC);

CREATE TABLE IF NOT EXISTS eval_results (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    case_id       TEXT NOT NULL,
    question      TEXT NOT NULL,
    metrics       TEXT NOT NULL DEFAULT '{}',
    hits          TEXT NOT NULL DEFAULT '[]',
    latency_ms    REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_results_run ON eval_results(run_id);
"""

_local = threading.local()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_conn() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        settings = get_settings()
        conn = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()


def close_conn() -> None:
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


def query_all(sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    rows = get_conn().execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def query_one(sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
    row = get_conn().execute(sql, params).fetchone()
    return dict(row) if row is not None else None


def execute(sql: str, params: Sequence[Any] = ()) -> int:
    conn = get_conn()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.rowcount


def execute_many(sql: str, rows: Sequence[Sequence[Any]]) -> None:
    conn = get_conn()
    conn.executemany(sql, rows)
    conn.commit()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def bump_kb_version(kb_id: str) -> None:
    execute(
        "UPDATE knowledge_bases SET version = version + 1, updated_at = ? WHERE id = ?",
        (now(), kb_id),
    )
