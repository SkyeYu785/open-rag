"""File parsers: turn an uploaded file into ordered blocks.

A parsed document is a list of :class:`Section` objects, in original order.
A section is either free text (with an optional page number) or a structured
table, which lets the chunker keep table headers attached to every row chunk.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

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
class Table:
    headers: list[str]
    rows: list[list[str]]
    title: str = ""

    def header_line(self) -> str:
        return " | ".join(header for header in self.headers if header)


@dataclass
class Section:
    text: str = ""
    page: int | None = None
    table: Table | None = None


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
# helpers
# --------------------------------------------------------------------------- #
def _split_header(rows: list[list[str]], prefix: str) -> Table | None:
    """Treat the first non-empty row as the header row."""
    cleaned = []
    for row in rows:
        cells = [" ".join(cell.split()) for cell in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return None

    headers = cleaned[0]
    width = len(headers)
    for index, cell in enumerate(headers):
        if not cell:
            headers[index] = f"{prefix}列{index + 1}"

    data = []
    for row in cleaned[1:]:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        data.append(row[:width] if width else row)
    return Table(headers=headers, rows=data)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# plain text like formats
# --------------------------------------------------------------------------- #
def _parse_text(path: Path) -> ParsedDocument:
    text = normalize(_read_text(path))
    if not text:
        raise ParseError("文件内容为空")
    return ParsedDocument(sections=[Section(text=text)])


def _parse_csv(path: Path) -> ParsedDocument:
    reader = csv.reader(io.StringIO(_read_text(path)))
    rows: list[list[str]] = []
    for index, row in enumerate(reader):
        if index > 20000:
            break
        if any(cell.strip() for cell in row):
            rows.append(row)

    table = _split_header(rows, "列")
    if table is None:
        raise ParseError("CSV 中没有可读取的行")
    table.title = path.stem
    return ParsedDocument(sections=[Section(table=table)], meta={"rows": len(table.rows)})


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
    return ParsedDocument(sections=[Section(text=text)])


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
            sections.append(Section(text=text, page=number))

    if not sections:
        raise ParseError("没有从 PDF 中提取到文本，可能是扫描件（当前版本不支持 OCR）")
    return ParsedDocument(sections=sections, meta={"pages": len(reader.pages)})


# --------------------------------------------------------------------------- #
# office formats
# --------------------------------------------------------------------------- #
def _iter_docx_blocks(document) -> Iterator[Any]:
    from docx.oxml.ns import qn
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, document)


def _docx_table(table, title: str) -> Table | None:
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = " ".join(cell.text.split())
            # merged cells are reported once per spanned column
            if not cells or cells[-1] != text:
                cells.append(text)
        if any(cells):
            rows.append(cells)

    parsed = _split_header(rows, "列")
    if parsed is not None:
        parsed.title = title
    return parsed


def _parse_docx(path: Path) -> ParsedDocument:
    try:
        import docx
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover
        raise ParseError("缺少 python-docx 依赖，无法解析 DOCX") from exc

    document = docx.Document(str(path))
    sections: list[Section] = []
    buffer: list[str] = []
    table_count = 0

    def flush() -> None:
        nonlocal buffer
        text = normalize("\n\n".join(buffer))
        if text:
            sections.append(Section(text=text))
        buffer = []

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name or "") if block.style is not None else ""
            if style.lower().startswith("heading"):
                level = int("".join(ch for ch in style if ch.isdigit()) or "1")
                buffer.append("#" * min(level, 6) + " " + text)
            else:
                buffer.append(text)
        else:
            table_count += 1
            table = _docx_table(block, title=f"表格 {table_count}")
            if table is not None and table.rows:
                flush()
                sections.append(Section(table=table))

    flush()

    if not sections:
        raise ParseError("DOCX 中没有可提取的文本")
    return ParsedDocument(sections=sections, meta={"tables": table_count})


def _parse_xlsx(path: Path) -> ParsedDocument:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover
        raise ParseError("缺少 openpyxl 依赖，无法解析 XLSX") from exc

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    sections: list[Section] = []
    try:
        for sheet in workbook.worksheets:
            rows: list[list[str]] = []
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                if index > 5000:
                    break
                cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                if cells:
                    rows.append(cells)
            table = _split_header(rows, f"{sheet.title}列")
            if table is not None:
                table.title = f"工作表 {sheet.title}"
                sections.append(Section(table=table))
    finally:
        workbook.close()

    if not sections:
        raise ParseError("XLSX 中没有可读取的内容")
    return ParsedDocument(sections=sections, meta={"sheets": len(sections)})
