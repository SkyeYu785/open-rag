from __future__ import annotations

from openrag.services.bm25 import BM25Index
from openrag.services.chunking import chunk_sections
from openrag.services.parsing import Section
from openrag.services.text import normalize, tokenize


def test_chunking_respects_size_and_keeps_headings():
    text = "# 参数\n\n" + ("额定流量 120 立方米每小时。" * 60) + "\n\n## 维护\n\n每 2000 小时更换密封。"
    chunks = chunk_sections([Section(text)], size=300, overlap=40)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 340 for chunk in chunks)
    assert any("维护" in chunk.text for chunk in chunks)
    assert chunks[0].meta.get("heading") == "参数"


def test_chunking_preserves_page_numbers():
    chunks = chunk_sections(
        [Section("第一页的内容。", page=1), Section("第二页的内容。", page=2)], size=300
    )
    assert [chunk.page for chunk in chunks] == [1, 2]


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
