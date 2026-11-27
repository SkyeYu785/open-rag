"""Configuration is read from environment variables (optionally via `.env`)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # optional, only used for local development convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

VERSION = "0.1.0"


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = _str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = _str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    upload_dir: Path
    embedding_provider: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_dim: int
    child_tokens: int
    parent_tokens: int
    overlap_tokens: int
    max_upload_mb: int
    # reranker
    rerank_provider: str
    rerank_model: str
    rerank_base_url: str
    rerank_path: str
    rerank_api_key: str
    rerank_timeout: float
    rerank_candidates: int
    # retrieval filtering
    min_score: float
    relative_cutoff: float
    max_per_doc: int
    dedupe_parent: bool
    cors_origins: tuple[str, ...]
    version: str = VERSION

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(_str("OPENRAG_DATA_DIR", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()

    origins = tuple(o.strip() for o in _str("OPENRAG_CORS_ORIGINS", "*").split(",") if o.strip())

    # OPENRAG_CHUNK_SIZE is the pre-v0.2 name and is kept as an alias.
    child_tokens = max(48, _int("OPENRAG_CHILD_TOKENS", _int("OPENRAG_CHUNK_SIZE", 320)))
    parent_tokens = max(child_tokens * 2, _int("OPENRAG_PARENT_TOKENS", 1400))
    overlap_tokens = max(0, min(_int("OPENRAG_CHUNK_OVERLAP", 48), child_tokens // 3))

    settings = Settings(
        data_dir=data_dir,
        db_path=data_dir / "openrag.db",
        upload_dir=data_dir / "uploads",
        embedding_provider=_str("OPENRAG_EMBEDDING_PROVIDER", "auto").lower(),
        embedding_model=_str("OPENRAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        embedding_base_url=_str("OPENRAG_EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        embedding_api_key=_str("OPENRAG_EMBEDDING_API_KEY"),
        embedding_dim=_int("OPENRAG_EMBEDDING_DIM", 512),
        child_tokens=child_tokens,
        parent_tokens=parent_tokens,
        overlap_tokens=overlap_tokens,
        max_upload_mb=max(1, _int("OPENRAG_MAX_UPLOAD_MB", 50)),
        rerank_provider=_str("OPENRAG_RERANK_PROVIDER", "none").lower(),
        rerank_model=_str("OPENRAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_base_url=_str("OPENRAG_RERANK_BASE_URL", "http://127.0.0.1:8082/v1").rstrip("/"),
        rerank_path=_str("OPENRAG_RERANK_PATH", "/rerank"),
        rerank_api_key=_str("OPENRAG_RERANK_API_KEY"),
        rerank_timeout=_float("OPENRAG_RERANK_TIMEOUT", 30.0),
        rerank_candidates=max(1, _int("OPENRAG_RERANK_CANDIDATES", 30)),
        min_score=_float("OPENRAG_MIN_SCORE", 0.0),
        relative_cutoff=min(1.0, max(0.0, _float("OPENRAG_RELATIVE_CUTOFF", 0.75))),
        max_per_doc=max(0, _int("OPENRAG_MAX_PER_DOC", 0)),
        dedupe_parent=_bool("OPENRAG_DEDUPE_PARENT", True),
        cors_origins=origins or ("*",),
    )
    settings.ensure_dirs()
    return settings
