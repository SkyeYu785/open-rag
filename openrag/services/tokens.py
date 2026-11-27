"""Token counting.

The project supports BERT-style Chinese tokenizers (bge-*, most local embedding
models) and OpenAI-style tokenizers. For chunk sizing we only need a stable,
cheap estimate, so the default counter is a heuristic calibrated on the two:

* every CJK character (including full-width punctuation) counts as 1 token,
* every 4 latin/digit characters count as 1 token,
* whitespace is free.

If ``tiktoken`` is installed and ``OPENRAG_TOKENIZER=tiktoken`` is set, the
cl100k_base tokenizer is used instead (useful for OpenAI embedding models).
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3000-\u303f\uff00-\uffef]")


def _heuristic_count(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    latin = 0
    for char in text:
        if char.isspace() or _CJK.match(char):
            continue
        latin += 1
    return cjk + math.ceil(latin / 4)


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``."""
    if not text:
        return 0
    if os.getenv("OPENRAG_TOKENIZER", "").strip().lower() == "tiktoken":
        encoder = _tiktoken_encoder()
        if encoder is not None:
            return len(encoder.encode(text))
    return _heuristic_count(text)


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Return the longest suffix of ``text`` within ``max_tokens``."""
    if max_tokens <= 0 or not text:
        return ""
    total = count_tokens(text)
    if total <= max_tokens:
        return text
    ratio = len(text) / max(total, 1)
    length = max(1, int(max_tokens * ratio))
    return text[-length:]
