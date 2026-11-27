"""Pluggable rerankers.

Vector search answers "what is semantically close"; a cross-encoder answers
"does this chunk actually answer the question". Production pipelines rerank a
wider candidate set (30-100) and only then cut it down to the top-k sent to the
LLM.

Two backends are supported:

* ``http``  - any service exposing the common ``/rerank`` contract used by
              Cohere, Jina, vLLM, Xinference and TEI::

                  POST {base}/{path}
                  {"model": "...", "query": "...", "documents": ["..."], "top_n": 30}
                  -> {"results": [{"index": 0, "relevance_score": 0.92}, ...]}

* ``local`` - ``sentence_transformers.CrossEncoder`` (e.g. BAAI/bge-reranker-v2-m3).
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from ..config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    name = "base"
    model = "base"

    def rerank(self, query: str, documents: list[str]) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.model}"


def _normalise(scores: list[float]) -> list[float]:
    """Map raw cross-encoder logits into 0-1 when the service returns logits."""
    if not scores:
        return scores
    if all(0.0 <= score <= 1.0 for score in scores):
        return scores
    return [1.0 / (1.0 + math.exp(-max(min(score, 60.0), -60.0))) for score in scores]


class HttpReranker(Reranker):
    name = "http"

    def __init__(self, model: str, base_url: str, path: str, api_key: str, timeout: float) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.path = path if path.startswith("/") else f"/{path}"
        self.api_key = api_key
        self.timeout = timeout

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.path}"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.url,
                headers=headers,
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
            )
            response.raise_for_status()
            payload = response.json()

        # {"results": [...]} (Cohere / Jina / vLLM) or bare [...] (TEI)
        items = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise RuntimeError("rerank 服务返回格式无法识别")

        raw = [-1e9] * len(documents)
        for item in items:
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if isinstance(index, int) and 0 <= index < len(documents) and score is not None:
                raw[index] = float(score)
        return _normalise(raw)


class LocalReranker(Reranker):
    name = "local"

    def __init__(self, model: str) -> None:
        self.model = model
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading reranker %s (first call may take a while)", self.model)
            self._encoder = CrossEncoder(self.model)
        return self._encoder

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self._load().predict([(query, document) for document in documents])
        return _normalise([float(score) for score in scores])


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    settings = get_settings()
    provider = settings.rerank_provider

    if provider in {"", "none", "off", "disabled"}:
        return None
    if provider == "http":
        if not settings.rerank_base_url:
            raise RuntimeError("OPENRAG_RERANK_BASE_URL is required for the http reranker")
        return HttpReranker(
            model=settings.rerank_model,
            base_url=settings.rerank_base_url,
            path=settings.rerank_path,
            api_key=settings.rerank_api_key,
            timeout=settings.rerank_timeout,
        )
    if provider in {"local", "sentence-transformers"}:
        return LocalReranker(settings.rerank_model)
    raise RuntimeError(f"unknown rerank provider: {provider}")


def rerank_info() -> dict[str, object]:
    settings = get_settings()
    return {
        "provider": settings.rerank_provider,
        "model": settings.rerank_model,
        "candidates": settings.rerank_candidates,
    }
