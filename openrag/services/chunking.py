"""Structure aware chunking.

The splitter keeps paragraphs and headings together, splits oversized blocks on
sentence boundaries and adds a small character overlap between neighbours so
that answers crossing a boundary are still retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .parsing import Section

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SPLIT_RE = re.compile(r"(?<=[。！？；])|(?<=[.!?])\s+|\n+")


@dataclass
class Chunk:
    text: str
    page: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text into ``(heading_path, body)`` blocks."""
    blocks: list[tuple[str, str]] = []
    headings: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            blocks.append((" > ".join(headings), body))
        buffer.clear()

    for line in text.split("\n"):
        match = _HEADING_RE.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            headings = headings[: level - 1] + [match.group(2).strip()]
        else:
            buffer.append(line)
    flush()
    return blocks or [("", text)]


def _split_parts(text: str, size: int) -> list[str]:
    """Break text into paragraph/sentence level parts, each at most ``size``."""
    parts: list[str] = []
    for raw in _SPLIT_RE.split(text):
        part = (raw or "").strip()
        if not part:
            continue
        if len(part) <= size:
            parts.append(part)
        else:
            parts.extend(part[i : i + size] for i in range(0, len(part), size))
    return parts


def _pack(parts: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge parts into chunks of <= ``size`` with a character overlap."""
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if not buffer:
            buffer = part
            continue
        if len(buffer) + 1 + len(part) <= size:
            buffer = f"{buffer}\n{part}"
            continue
        chunks.append(buffer)
        carry = buffer[-overlap:] if overlap else ""
        if carry and len(carry) + 1 + len(part) <= size:
            buffer = f"{carry}\n{part}"
        else:
            buffer = part
    if buffer.strip():
        chunks.append(buffer)
    return chunks


def chunk_sections(sections: list[Section], size: int = 700, overlap: int = 100) -> list[Chunk]:
    size = max(120, size)
    overlap = max(0, min(overlap, size // 2))

    chunks: list[Chunk] = []
    for section in sections:
        text = (section.text or "").strip()
        if not text:
            continue
        for heading, body in _split_by_headings(text):
            prefix = f"[{heading}]\n" if heading else ""
            parts = _split_parts(prefix + body, size)
            meta = {"heading": heading} if heading else {}
            for piece in _pack(parts, size, overlap):
                cleaned = piece.strip()
                if cleaned:
                    chunks.append(Chunk(text=cleaned, page=section.page, meta=dict(meta)))

    return _merge_short(chunks, size)


def _merge_short(chunks: list[Chunk], size: int) -> list[Chunk]:
    """Merge trailing chunks that ended up far below the target size."""
    merged: list[Chunk] = []
    target = size // 4
    for chunk in chunks:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and len(chunk.text) < target
            and previous.page == chunk.page
            and previous.meta.get("heading") == chunk.meta.get("heading")
            and len(previous.text) + len(chunk.text) + 1 <= size
        ):
            previous.text = f"{previous.text}\n{chunk.text}"
            continue
        merged.append(chunk)
    return merged
