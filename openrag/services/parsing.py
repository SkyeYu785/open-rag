"""File parsers: turn an uploaded file into ordered text sections.

Sections keep a page number when the source format has pages, which lets the
retrieval API return citable locations.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .text import normalize

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
    ".xlsx",
    ".csv",
    ".html",
    ".htm",
}


class ParseError(Exception):
    """Raised when a file cannot be turned into text."""


@dataclass
class Section:
    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    sections: list[Section]
    meta: dict[str, Any] = field(default_factory=dict)


def parse_file(path: Path, ext: str | None = None) -> ParsedDocument:
    suffix = (ext or path.suffix).lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"不支持的文件类型: {suffix or '未知'}")
    handler = {
        ".txt": _parse_text,
        ".md": _parse_text,
        ".markdown": _parse_text,
        ".csv": _parse_csv,
        ".html": _parse_html,
        ".htm": _parse_html,
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".xlsx": _parse_xlsx,
    }[suffix]
    return handler(path)


# --------------------------------------------------------------------------- #
# plain text like formats
# --------------------------------------------------------------------------- #
def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


def _parse_text(path: Path) -> ParsedDocument:
    text = normalize(_read_text(path))
    if not text:
        raise ParseError("文件内容为空")
    return ParsedDocument(sections=[Section(text)])


def _parse_csv(path: Path) -> ParsedDocument:
    text = _read_text(path)
    reader = csv.reader(io.StringIO(text))
    lines: list[str] = []
    for index, row in enumerate(reader):
        if index > 20000:
            lines.append("…（内容过长，已截断）")
            break
        cells = [cell.strip() for cell in row if cell.strip()]
        if cells:
            lines.append(" | ".join(cells))
    if not lines:
        raise ParseError("CSV 中没有可读取的行")
    return ParsedDocument(sections=[Section(normalize("\n".join(lines)))], meta={"rows": len(lines)})


class _HTMLTextExtractor(HTMLParser):
    skip_tags = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._block_tags = {
            "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.skip_tags:
            self._skip_depth += 1
        elif tag in self._block_tags:
            self._parts.append("\n")
        elif tag in {"td", "th"}:
            self._parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.skip_tags and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return normalize("".join(self._parts))


def _parse_html(path: Path) -> ParsedDocument:
    parser = _HTMLTextExtractor()
    parser.feed(_read_text(path))
    text = parser.text()
    if not text:
        raise ParseError("HTML 中没有可提取的文本")
    return ParsedDocument(sections=[Section(text)])


# --------------------------------------------------------------------------- #
# pdf
# --------------------------------------------------------------------------- #
def _parse_pdf(path: Path) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ParseError("缺少 pypdf 依赖，无法解析 PDF") from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ParseError("PDF 已加密，无法读取") from exc

    sections: list[Section] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        text = normalize(raw)
        if text:
            sections.append(Section(text, page=number))

    if not sections:
        raise ParseError("没有从 PDF 中提取到文本，可能是扫描件（当前版本不支持 OCR）")
    return ParsedDocument(sections=sections, meta={"pages": len(reader.pages)})


# --------------------------------------------------------------------------- #
# office formats
# --------------------------------------------------------------------------- #
def _parse_docx(path: Path) -> ParsedDocument:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ParseError("缺少 python-docx 依赖，无法解析 DOCX") from exc

    document = docx.Document(str(path))
    blocks: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "") if paragraph.style is not None else ""
        if style.lower().startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "1"
            blocks.append("#" * min(int(level), 6) + " " + text)
        else:
            blocks.append(text)

    for table_index, table in enumerate(document.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append(f"[表格 {table_index}]\n" + "\n".join(rows))

    text = normalize("\n\n".join(blocks))
    if not text:
        raise ParseError("DOCX 中没有可提取的文本")
    return ParsedDocument(sections=[Section(text)], meta={"tables": len(document.tables)})


def _parse_xlsx(path: Path) -> ParsedDocument:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover
        raise ParseError("缺少 openpyxl 依赖，无法解析 XLSX") from exc

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    sections: list[Section] = []
    try:
        for sheet in workbook.worksheets:
            lines: list[str] = []
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                if index > 5000:
                    lines.append("…（内容过长，已截断）")
                    break
                cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                if cells:
                    lines.append(" | ".join(cells))
            if lines:
                sections.append(Section(f"[工作表: {sheet.title}]\n" + "\n".join(lines)))
    finally:
        workbook.close()

    if not sections:
        raise ParseError("XLSX 中没有可读取的内容")
    return ParsedDocument(sections=sections, meta={"sheets": len(sections)})
