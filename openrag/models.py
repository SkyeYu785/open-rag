"""Pydantic models shared by the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["hybrid", "vector", "keyword"]
DocStatus = Literal["pending", "processing", "ready", "failed"]


# --------------------------------------------------------------------------- #
# knowledge bases
# --------------------------------------------------------------------------- #
class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)


class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    description: str = ""
    document_count: int = 0
    chunk_count: int = 0
    created_at: str
    updated_at: str


# --------------------------------------------------------------------------- #
# documents & chunks
# --------------------------------------------------------------------------- #
class DocumentOut(BaseModel):
    id: str
    kb_id: str
    name: str
    ext: str = ""
    size: int = 0
    status: DocStatus = "pending"
    error: str | None = None
    chunk_count: int = 0
    char_count: int = 0
    created_at: str
    updated_at: str


class ChunkOut(BaseModel):
    id: str
    doc_id: str
    ordinal: int
    text: str
    page: int | None = None
    tokens: int = 0
    level: str = "child"
    start_offset: int | None = None
    end_offset: int | None = None


class ChunkPage(BaseModel):
    total: int
    items: list[ChunkOut]


class UploadResponse(BaseModel):
    accepted: list[DocumentOut]
    rejected: list[dict[str, str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
class RetrieveRequest(BaseModel):
    kb_id: str
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=50)
    mode: RetrievalMode = "hybrid"
    include_context: bool = True
    # filtering overrides; null means "use the server defaults"
    min_score: float | None = Field(default=None, ge=0)
    relative_cutoff: float | None = Field(default=None, ge=0, le=1)
    max_per_doc: int | None = Field(default=None, ge=0, le=50)
    dedupe_parent: bool | None = None


class RetrievalHit(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    ordinal: int
    text: str
    page: int | None = None
    context: str | None = None
    parent_id: str | None = None
    tokens: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)
    start_offset: int | None = None
    end_offset: int | None = None
    score: float
    relevance: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    mode: RetrievalMode
    kb_id: str
    total_chunks: int
    latency_ms: float
    recalled: int = 0
    returned: int = 0
    dropped: int = 0
    rerank_applied: bool = False
    filter_score: str = ""
    hits: list[RetrievalHit]

# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
class DatasetCreate(BaseModel):
    kb_id: str
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class DatasetOut(BaseModel):
    id: str
    kb_id: str
    name: str
    description: str = ""
    case_count: int = 0
    created_at: str
    updated_at: str


class CaseCreate(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    reference_doc_ids: list[str] = Field(default_factory=list)
    reference_chunk_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class CaseOut(BaseModel):
    id: str
    dataset_id: str
    question: str
    reference_doc_ids: list[str] = Field(default_factory=list)
    reference_chunk_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    created_at: str


class RunCreate(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)
    mode: RetrievalMode = "hybrid"


class CaseResult(BaseModel):
    case_id: str
    question: str
    metrics: dict[str, float]
    latency_ms: float
    hits: list[dict[str, Any]] = Field(default_factory=list)


class RunOut(BaseModel):
    id: str
    dataset_id: str
    kb_id: str
    status: Literal["pending", "running", "done", "failed"]
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    created_at: str
    finished_at: str | None = None
    case_count: int = 0
    results: list[CaseResult] | None = None
