"""Tokenisation helpers shared by the hashing embedder and BM25.

Chinese is indexed with character unigrams + bigrams (no external tokeniser such
as jieba required), latin text with words.
"""

from __future__ import annotations

import re

LATIN_RE = re.compile(r"[a-z0-9]+(?:[._+#-][a-z0-9]+)*")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
SPACE_RE = re.compile(r"[ \t\u3000]+")


def normalize(text: str) -> str:
    """Normalise whitespace and remove invisible characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = SPACE_RE.sub(" ", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = LATIN_RE.findall(lowered)
    for run in CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        tokens.extend(run)  # unigrams improve recall for single-character queries
    return tokens


def snippet(text: str, limit: int = 240) -> str:
    text = normalize(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
