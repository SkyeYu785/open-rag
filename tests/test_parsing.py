from __future__ import annotations

import csv

import docx
from openpyxl import Workbook

from openrag.services.chunking import chunk_sections
from openrag.services.parsing import parse_file


def _write_docx(path):
    document = docx.Document()
    document.add_heading("备件台账", level=1)
    document.add_paragraph("以下为常用备件清单。")

    table = document.add_table(rows=3, cols=3)
    for column, header in enumerate(["型号", "压力", "库存"]):
        table.cell(0, column).text = header
    for row_index, row in enumerate([["E-201", "0.8MPa", "12"], ["E-202", "1.2MPa", "4"]], start=1):
        for column, value in enumerate(row):
            table.cell(row_index, column).text = value

    document.add_paragraph("盘点周期为每季度一次。")
    document.save(path)


def test_docx_tables_become_structured_rows(tmp_path):
    path = tmp_path / "备件台账.docx"
    _write_docx(path)

    parsed = parse_file(path)
    tables = [section.table for section in parsed.sections if section.table is not None]

    assert len(tables) == 1
    assert tables[0].headers == ["型号", "压力", "库存"]
    assert tables[0].rows[0] == ["E-201", "0.8MPa", "12"]
    assert tables[0].title == "表格 1"


def test_docx_table_chunks_repeat_the_header(tmp_path):
    path = tmp_path / "备件台账.docx"
    _write_docx(path)

    chunks = chunk_sections(parse_file(path).sections, child_tokens=60, parent_tokens=300)
    children = [chunk for chunk in chunks if chunk.level == "child"]
    text = "\n".join(chunk.text for chunk in children)

    assert "型号 | 压力 | 库存" in text
    assert "型号: E-201；压力: 0.8MPa；库存: 12" in text
    assert "型号: E-202；压力: 1.2MPa；库存: 4" in text


def test_heading_is_kept_after_a_table(tmp_path):
    path = tmp_path / "备件台账.docx"
    _write_docx(path)

    chunks = chunk_sections(parse_file(path).sections, child_tokens=80, parent_tokens=300)
    after_table = [chunk for chunk in chunks if "盘点周期" in chunk.text]

    assert after_table
    assert after_table[0].meta["heading"] == "备件台账"


def test_xlsx_sheet_becomes_a_table(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "台账"
    sheet.append(["型号", "压力"])
    sheet.append(["E-201", "0.8MPa"])
    path = tmp_path / "台账.xlsx"
    workbook.save(path)

    parsed = parse_file(path)
    table = parsed.sections[0].table

    assert table is not None
    assert table.headers == ["型号", "压力"]
    assert table.rows == [["E-201", "0.8MPa"]]
    assert table.title == "工作表 台账"


def test_csv_first_row_is_the_header(tmp_path):
    path = tmp_path / "备件.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["型号", "压力"])
        writer.writerow(["E-201", "0.8MPa"])

    parsed = parse_file(path)
    table = parsed.sections[0].table

    assert table is not None
    assert table.headers == ["型号", "压力"]
    assert table.rows == [["E-201", "0.8MPa"]]


def test_plain_text_has_no_table(tmp_path):
    path = tmp_path / "说明.md"
    path.write_text("# 说明\n\n这是一段普通文本。", encoding="utf-8")

    parsed = parse_file(path)
    assert parsed.sections[0].table is None
    assert "普通文本" in parsed.sections[0].text
