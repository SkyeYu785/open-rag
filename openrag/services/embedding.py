"""Pluggable embedding providers.

Three implementations ship with the project:

* ``hash``                  - deterministic feature hashing, zero dependencies,
                              always available (default when nothing else is
                              installed). Works offline; quality is keyword-ish.
* ``sentence-transformers`` - real local embeddings, e.g. BAAI/bge-small-zh-v1.5.
* ``openai``                - any OpenAI compatible ``/embeddings`` endpoint
                              (OpenAI, DashScope, DeepSeek, Ollama, vLLM, ...).
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from collections import Counter
from functools import lru_cache

import numpy as np

from ..config import get_settings
from .text import tokenize

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    name = "base"
    model = "base"

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.model}:{self.dim}"

    @property
    def dim(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class HashEmbedder(EmbeddingProvider):
    """Hashing trick over word / CJK n-gram tokens.

    Deterministic across processes (blake2b, not ``hash()``), so indexes stay
    valid after a restart.
    """

    name = "hash"

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim
        self.model = f"hashing-{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + np.log(count))
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])


class SentenceTransformerEmbedder(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, model: str) -> None:
        self.model = model
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s (first call may take a while)", self.model)
            self._encoder = SentenceTransformer(self.model)
        return self._encoder

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._load().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(matrix, dtype=np.float32)


class OpenAIEmbedder(EmbeddingProvider):
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @property
    def dim(self) -> int:
        return get_settings().embedding_dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        vectors: list[list[float]] = []
        batch_size = 64
        with httpx.Client(timeout=120.0) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()
                data = sorted(payload["data"], key=lambda item: item.get("index", 0))
                vectors.extend(item["embedding"] for item in data)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def _has_sentence_transformers() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def configured_provider() -> str:
    """Resolved provider name, without loading any model."""
    provider = get_settings().embedding_provider
    if provider == "auto":
        return "sentence-transformers" if _has_sentence_transformers() else "hash"
    return provider


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    settings = get_settings()
    provider = configured_provider()

    if provider == "hash":
        return HashEmbedder(settings.embedding_dim)
    if provider in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(settings.embedding_model)
    if provider == "openai":
        if not settings.embedding_base_url:
            raise RuntimeError("OPENRAG_EMBEDDING_BASE_URL is required for the openai provider")
        return OpenAIEmbedder(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
        )
    raise RuntimeError(f"unknown embedding provider: {provider}")


def embedder_info() -> dict[str, object]:
    try:
        embedder = get_embedder()
        return {"name": embedder.name, "model": embedder.model, "signature": embedder.signature}
    except Exception as exc:  # pragma: no cover - configuration errors surface in /health
        return {"name": "unavailable", "model": "", "error": str(exc)}
