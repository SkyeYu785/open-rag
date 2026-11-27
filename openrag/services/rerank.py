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
import threading
import time
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


class RerankUnusable(RuntimeError):
    """The service answered, but the scores cannot rank anything."""


# A cross-encoder must score a query against itself near the top of the scale.
# Generative rerankers served by the wrong runtime (e.g. Qwen3-Reranker on
# llama.cpp, which only implements classification-head rerankers) return ~1e-7
# for *every* pair, including identical text, and would silently scramble the
# ranking. A self-test separates "broken service" from "no relevant document".
SELF_TEST_QUERY = "rerank 自检：设备最大工作压力是多少"
SELF_TEST_MIN_SCORE = 0.5
_SELF_TEST_RETRY_SECONDS = 30.0

_health: dict[str, tuple[bool, float, str | None]] = {}
_health_lock = threading.Lock()


def _normalise(scores: list[float]) -> list[float]:
    """Map raw cross-encoder logits into 0-1 when the service returns logits."""
    if not scores:
        return scores
    if all(0.0 <= score <= 1.0 for score in scores):
        return scores
    return [1.0 / (1.0 + math.exp(-max(min(score, 60.0), -60.0))) for score in scores]


def validate_scores(scores: list[float]) -> None:
    """Reject empty or non-finite rerank output."""
    if not scores:
        raise RerankUnusable("rerank 服务没有返回任何分数")
    if not all(math.isfinite(score) for score in scores):
        raise RerankUnusable("rerank 服务返回了非法分数")


def reranker_status(reranker: Reranker) -> tuple[bool, str | None]:
    """Self-test the reranker once (with a retry window) and cache the verdict."""
    key = getattr(reranker, "signature", None) or (
        f"{getattr(reranker, 'name', type(reranker).__name__)}:{getattr(reranker, 'model', '')}"
    )
    now = time.monotonic()
    with _health_lock:
        cached = _health.get(key)
        if cached is not None:
            ok, expires, message = cached
            if ok or now < expires:
                return ok, message

    message: str | None = None
    try:
        scores = reranker.rerank(SELF_TEST_QUERY, [SELF_TEST_QUERY])
        top = max(scores) if scores else 0.0
    except Exception as exc:  # noqa: BLE001
        top = 0.0
        message = f"rerank 服务调用失败（{exc}），已回退到混合检索。"

    if top >= SELF_TEST_MIN_SCORE:
        message = None
    else:
        message = message or (
            f"rerank 服务自检未通过：完全相同的文本只得到 {top:.4f} 分。"
            "通常是该服务不支持这个模型——llama.cpp 的 /v1/rerank 只实现 classification-head "
            "模型（bge-reranker-v2-m3 等），Qwen3-Reranker 这类生成式模型需要 vLLM / SGLang。"
            "已回退到混合检索。"
        )

    ok = top >= SELF_TEST_MIN_SCORE
    with _health_lock:
        _health[key] = (ok, float("inf") if ok else now + _SELF_TEST_RETRY_SECONDS, message)
    logger.info("reranker self-test %s (%s)", "passed" if ok else "failed", key)
    return ok, message


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
