"""Hierarchical, structure aware chunking.

Two levels are produced for every document:

* **parent**  - a structural block (a heading section, or a whole table) that is
                large enough to answer a question on its own,
* **child**   - the retrieval unit. Small chunks are indexed and searched, and
                the API returns their parent as ``context`` (small-to-big).

Sizing is done in **tokens** (:mod:`openrag.services.tokens`), never in raw
characters, so Chinese and English documents end up with comparable chunk
sizes. Tables are rendered row by row as ``列名: 值`` pairs and every chunk
repeats the header line, so a single row is still interpretable out of context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from .parsing import Section, Table
from .tokens import count_tokens, trim_to_tokens

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SPLIT_RE = re.compile(r"(?<=[。！？；])|(?<=[.!?])\s+|\n+")


@dataclass
class Part:
    text: str
    start: int | None = None
    end: int | None = None


@dataclass
class Chunk:
    text: str
    page: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    level: str = "child"
    key: str | None = None
    parent_key: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass
class Block:
    heading: str
    text: str
    page: int | None
    offset: int
    table: Table | None
    section: int
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# block extraction
# --------------------------------------------------------------------------- #
def _flush_lines(
    lines: list[tuple[str, int, int]], headings: list[str], page: int | None, section: int
) -> Iterator[Block]:
    if not lines:
        return
    first = 0
    while first < len(lines) and not lines[first][0].strip():
        first += 1
    last = len(lines) - 1
    while last >= first and not lines[last][0].strip():
        last -= 1
    if last < first:
        return

    selected = lines[first : last + 1]
    body = "\n".join(item[0] for item in selected)
    yield Block(
        heading=" > ".join(headings),
        text=body,
        page=page,
        offset=selected[0][1],
        table=None,
        section=section,
    )


def _iter_blocks(section: Section, headings: list[str], section_index: int) -> Iterator[Block]:
    if section.table is not None:
        yield Block(
            heading=" > ".join(headings),
            text="",
            page=section.page,
            offset=0,
            table=section.table,
            section=section_index,
        )
        return

    text = section.text or ""
    buffer: list[tuple[str, int, int]] = []
    cursor = 0
    for line in text.split("\n"):
        start, end = cursor, cursor + len(line)
        cursor = end + 1
        match = _HEADING_RE.match(line.strip())
        if match:
            yield from _flush_lines(buffer, headings, section.page, section_index)
            buffer = []
            level = len(match.group(1))
            headings[:] = headings[: level - 1] + [match.group(2).strip()]
        else:
            buffer.append((line, start, end))
    yield from _flush_lines(buffer, headings, section.page, section_index)


def _block_prefix(block: Block) -> str:
    lines: list[str] = []
    if block.heading:
        lines.append(f"[{block.heading}]")
    if block.table is not None:
        if block.table.title:
            lines.append(f"[{block.table.title}]")
        header = block.table.header_line()
        if header:
            lines.append(header)
    return "\n".join(lines) + "\n" if lines else ""


# --------------------------------------------------------------------------- #
# parts (sentence / row level units that never exceed a child chunk)
# --------------------------------------------------------------------------- #
def _make_part(segment: str, base: int, max_tokens: int) -> list[Part]:
    stripped = segment.strip()
    if not stripped:
        return []
    lead = len(segment) - len(segment.lstrip())
    start = base + lead
    if count_tokens(stripped) <= max_tokens:
        return [Part(stripped, start, start + len(stripped))]
    return _hard_split(stripped, max_tokens, start)


def _hard_split(text: str, max_tokens: int, base: int | None) -> list[Part]:
    """Split on character boundaries with a binary search on token count."""
    parts: list[Part] = []
    index = 0
    length = len(text)
    while index < length:
        low, high = index + 1, length
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(text[index:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        end = max(low, index + 1)
        segment = text[index:end]
        stripped = segment.strip()
        if stripped:
            lead = len(segment) - len(segment.lstrip())
            start = None if base is None else base + index + lead
            parts.append(
                Part(stripped, start, None if start is None else start + len(stripped))
            )
        index = end
    return parts


def _split_parts(text: str, max_tokens: int, base_offset: int = 0) -> list[Part]:
    parts: list[Part] = []
    previous = 0
    for match in _SPLIT_RE.finditer(text):
        if match.start() > previous:
            parts.extend(_make_part(text[previous : match.start()], base_offset + previous, max_tokens))
        previous = max(previous, match.end())
    if previous < len(text):
        parts.extend(_make_part(text[previous:], base_offset + previous, max_tokens))
    return [part for part in parts if part.text]


def _table_parts(table: Table, max_tokens: int) -> list[Part]:
    parts: list[Part] = []
    for row in table.rows:
        pairs: list[str] = []
        for index, cell in enumerate(row):
            value = (cell or "").strip()
            if not value:
                continue
            name = table.headers[index].strip() if index < len(table.headers) else ""
            name = name or f"列{index + 1}"
            pairs.append(f"{name}: {value}")
        if not pairs:
            continue
        line = "；".join(pairs)
        if count_tokens(line) <= max_tokens:
            parts.append(Part(line))
        else:
            parts.extend(_hard_split(line, max_tokens, None))
    return parts


def _group_text(group: list[Part]) -> str:
    return "\n".join(part.text for part in group)


def _group_bounds(group: list[Part]) -> tuple[int, int] | None:
    starts = [part.start for part in group if part.start is not None]
    ends = [part.end for part in group if part.end is not None]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _pack_parts(parts: list[Part], budget: int, overlap_tokens: int) -> list[list[Part]]:
    """Greedily pack parts into groups no larger than ``budget`` tokens."""
    if not parts:
        return []

    groups: list[list[Part]] = []
    current: list[Part] = []
    current_tokens = 0

    for part in parts:
        tokens = count_tokens(part.text)
        if not current:
            current = [part]
            current_tokens = tokens
            continue
        if current_tokens + tokens + 1 <= budget:
            current.append(part)
            current_tokens += tokens + 1
            continue

        groups.append(current)
        carry: Part | None = None
        if overlap_tokens > 0:
            tail = trim_to_tokens(_group_text(current), overlap_tokens)
            if tail and count_tokens(tail) + tokens + 1 <= budget:
                carry = Part(tail)
        current = [carry, part] if carry is not None else [part]
        current_tokens = tokens + (count_tokens(carry.text) + 1 if carry is not None else 0)

    if current:
        groups.append(current)
    return groups


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def chunk_sections(
    sections: list[Section],
    child_tokens: int = 320,
    parent_tokens: int = 1400,
    overlap_tokens: int = 48,
) -> list[Chunk]:
    """Return parents and their children, parents first.

    Parents carry ``key``; children carry ``parent_key`` matching it.
    """
    child_tokens = max(48, child_tokens)
    parent_tokens = max(child_tokens * 2, parent_tokens)
    overlap_tokens = max(0, min(overlap_tokens, child_tokens // 3))

    chunks: list[Chunk] = []
    headings: list[str] = []

    for section_index, section in enumerate(sections):
        for block_index, block in enumerate(_iter_blocks(section, headings, section_index)):
            prefix = _block_prefix(block)
            prefix_tokens = count_tokens(prefix)
            parts = (
                _table_parts(block.table, child_tokens)
                if block.table is not None
                else _split_parts(block.text, child_tokens, base_offset=block.offset)
            )
            if not parts:
                continue

            meta: dict[str, Any] = {"section": section_index}
            if block.heading:
                meta["heading"] = block.heading
            if block.table is not None and block.table.title:
                meta["table"] = block.table.title

            parent_budget = max(child_tokens, parent_tokens - prefix_tokens)
            for group_index, group in enumerate(_pack_parts(parts, parent_budget, 0)):
                key = f"{section_index}:{block_index}:{group_index}"
                chunks.append(_build_chunk(block, prefix, meta, group, "parent", key, None))

                child_budget = max(32, child_tokens - prefix_tokens)
                child_overlap = 0 if block.table is not None else overlap_tokens
                for child_group in _pack_parts(group, child_budget, child_overlap):
                    chunks.append(_build_chunk(block, prefix, meta, child_group, "child", None, key))

    return chunks


def _build_chunk(
    block: Block,
    prefix: str,
    meta: dict[str, Any],
    group: list[Part],
    level: str,
    key: str | None,
    parent_key: str | None,
) -> Chunk:
    bounds = _group_bounds(group)
    return Chunk(
        text=prefix + _group_text(group),
        page=block.page,
        meta=dict(meta),
        level=level,
        key=key,
        parent_key=parent_key,
        start_offset=bounds[0] if bounds else None,
        end_offset=bounds[1] if bounds else None,
    )
