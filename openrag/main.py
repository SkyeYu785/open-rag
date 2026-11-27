"""FastAPI application: JSON API under ``/v1`` and the web UI at ``/``."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import db
from .api import documents, evaluation, knowledge_bases, retrieval
from .config import get_settings
from .services import embedding, rerank

settings = get_settings()

API_DESCRIPTION = """
**open-rag** 是一个最小可用的自托管 RAG 后端。

* 文档入库：上传 → 解析 → 分块 → 向量化（支持 PDF / DOCX / XLSX / CSV / Markdown / TXT / HTML）
* 检索：向量 + BM25 混合检索，RRF 融合，返回可引用的分块
* 评测：用标注好的问答集评估检索质量（hit rate / recall / MRR / nDCG）
"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="open-rag",
    version=settings.version,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/health", tags=["system"])
def health() -> dict:
    provider = embedding.configured_provider()
    return {
        "status": "ok",
        "version": settings.version,
        "embedding": {
            "provider": provider,
            "model": settings.embedding_model if provider != "hash" else f"hashing-{settings.embedding_dim}",
        },
        "chunking": {
            "unit": "tokens",
            "child_tokens": settings.child_tokens,
            "parent_tokens": settings.parent_tokens,
            "overlap_tokens": settings.overlap_tokens,
        },
        "rerank": rerank.rerank_info(),
        "retrieval": {
            "min_score": settings.min_score,
            "relative_cutoff": settings.relative_cutoff,
            "max_per_doc": settings.max_per_doc,
            "dedupe_parent": settings.dedupe_parent,
        },
        "max_upload_mb": settings.max_upload_mb,
    }


for router in (knowledge_bases.router, documents.router, retrieval.router, evaluation.router):
    app.include_router(router, prefix="/v1")

WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
