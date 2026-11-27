from __future__ import annotations

from openrag.services.bm25 import BM25Index
from openrag.services.chunking import chunk_sections
from openrag.services.parsing import Section, Table
from openrag.services.text import normalize, tokenize
from openrag.services.tokens import count_tokens


def _children(chunks):
    return [chunk for chunk in chunks if chunk.level == "child"]


def _parents(chunks):
    return [chunk for chunk in chunks if chunk.level == "parent"]


def test_token_counter_is_language_aware():
    assert count_tokens("") == 0
    assert count_tokens("最大工作压力") == 6
    assert count_tokens("hello world") <= 4
    assert count_tokens("E-201") <= 3


def test_chunking_produces_parents_and_children():
    text = "# 技术参数\n\n" + "额定流量 120 立方米每小时，最大工作压力 0.8MPa。" * 30
    chunks = chunk_sections([Section(text)], child_tokens=60, parent_tokens=200, overlap_tokens=10)

    parents = _parents(chunks)
    children = _children(chunks)

    assert parents and children
    assert len(children) > len(parents)
    assert all(count_tokens(child.text) <= 60 for child in children)
    assert all(count_tokens(parent.text) <= 200 for parent in parents)

    keys = {parent.key for parent in parents}
    assert all(child.parent_key in keys for child in children)


def test_chunking_keeps_heading_prefix():
    text = "# E-201 离心泵\n\n## 技术参数\n\n型号 E-201，最大工作压力 0.8MPa。"
    chunks = chunk_sections([Section(text)], child_tokens=120, parent_tokens=300)

    child = _children(chunks)[0]
    assert child.text.startswith("[E-201 离心泵 > 技术参数]")
    assert child.meta["heading"] == "E-201 离心泵 > 技术参数"


def test_offsets_point_into_the_section_text():
    body = "额定流量 120 立方米每小时。"
    text = "# 参数\n\n" + body
    chunks = chunk_sections([Section(text)], child_tokens=60, parent_tokens=200)

    child = _children(chunks)[0]
    assert child.start_offset is not None and child.end_offset is not None
    assert text[child.start_offset : child.end_offset] == body


def test_chunking_preserves_page_numbers():
    chunks = chunk_sections(
        [Section("第一页的内容。", page=1), Section("第二页的内容。", page=2)],
        child_tokens=60,
        parent_tokens=200,
    )
    assert [chunk.page for chunk in _children(chunks)] == [1, 2]


def test_table_rows_are_rendered_as_key_value_pairs():
    table = Table(
        headers=["型号", "最大压力"],
        rows=[["E-201", "0.8MPa"], ["E-202", "1.2MPa"]],
        title="台账",
    )
    chunks = chunk_sections([Section(table=table)], child_tokens=50, parent_tokens=200)
    children = _children(chunks)

    assert children
    for child in children:
        assert "型号 | 最大压力" in child.text  # header injected into every chunk
    joined = "\n".join(child.text for child in children)
    assert "型号: E-201；最大压力: 0.8MPa" in joined
    assert "型号: E-202；最大压力: 1.2MPa" in joined


def test_table_parent_contains_all_rows():
    table = Table(headers=["型号", "压力"], rows=[["E-201", "0.8MPa"], ["E-202", "1.2MPa"]])
    parents = _parents(chunk_sections([Section(table=table)], child_tokens=50, parent_tokens=200))

    assert len(parents) == 1
    assert "E-201" in parents[0].text and "E-202" in parents[0].text


def test_bm25_ranks_the_matching_document_first():
    index = BM25Index(["离心泵的额定流量是 120 立方米每小时", "更换机械密封需要 2000 小时", "完全无关的文本"])
    results = index.search("机械密封", 3)
    assert results[0][0] == 1
    assert results[0][1] > 0


def test_tokenize_handles_chinese_and_english():
    tokens = tokenize("E-201 泵 pressure")
    assert "pressure" in tokens
    assert "泵" in tokens


def test_normalize_collapses_blank_lines():
    assert normalize("a\r\n\r\n\r\n\r\nb") == "a\n\nb"
