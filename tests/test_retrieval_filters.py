from __future__ import annotations

from conftest import SAMPLE, upload

from openrag.services.retrieval import Filters, _apply_filters


def _hit(chunk_id: str, doc_id: str, parent_id: str, vector: float, keyword: float = 0.0) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "parent_id": parent_id,
        "text": "",
        "scores": {"vector": vector, "keyword": keyword},
    }


def test_relative_cutoff_drops_the_tail():
    hits = [_hit("a", "d1", "p1", 0.9), _hit("b", "d2", "p2", 0.7), _hit("c", "d3", "p3", 0.2)]
    kept, dropped = _apply_filters(hits, 10, Filters(0.0, 0.75, 0, False), "vector")

    assert [hit["chunk_id"] for hit in kept] == ["a", "b"]
    assert dropped == 1


def test_min_score_is_absolute():
    hits = [_hit("a", "d1", "p1", 0.4), _hit("b", "d2", "p2", 0.3)]
    kept, dropped = _apply_filters(hits, 10, Filters(0.35, 0.0, 0, False), "vector")

    assert [hit["chunk_id"] for hit in kept] == ["a"]
    assert dropped == 1


def test_dedupe_parent_keeps_the_best_child():
    hits = [
        _hit("a1", "d1", "p1", 0.9),
        _hit("a2", "d1", "p1", 0.8),
        _hit("b1", "d2", "p2", 0.7),
    ]
    kept, dropped = _apply_filters(hits, 10, Filters(0.0, 0.0, 0, True), "vector")

    assert [hit["chunk_id"] for hit in kept] == ["a1", "b1"]
    assert dropped == 1


def test_max_per_doc_caps_results_per_document():
    hits = [
        _hit("a1", "d1", "p1", 0.9),
        _hit("a2", "d1", "p2", 0.8),
        _hit("a3", "d1", "p3", 0.7),
        _hit("b1", "d2", "p4", 0.6),
    ]
    kept, dropped = _apply_filters(hits, 10, Filters(0.0, 0.0, 2, False), "vector")

    assert [hit["chunk_id"] for hit in kept] == ["a1", "a2", "b1"]
    assert dropped == 1


def test_truncation_is_not_counted_as_dropped():
    hits = [_hit(f"c{index}", f"d{index}", f"p{index}", 0.9) for index in range(5)]
    kept, dropped = _apply_filters(hits, 2, Filters(0.0, 0.0, 0, False), "vector")

    assert len(kept) == 2
    assert dropped == 0


class FakeReranker:
    name = "fake"
    model = "fake"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [0.9] + [0.1] * (len(documents) - 1)


def test_reranker_reorders_and_filters(client, kb, monkeypatch):
    upload(client, kb["id"], "E-201.md", SAMPLE)
    monkeypatch.setattr("openrag.services.retrieval.get_reranker", lambda: FakeReranker())

    payload = client.post(
        "/v1/retrieve",
        json={
            "kb_id": kb["id"],
            "query": "离心泵",
            "top_k": 5,
            "min_score": 0.3,
            "relative_cutoff": 0.5,
            "dedupe_parent": False,
        },
    ).json()

    assert payload["rerank_applied"] is True
    assert payload["filter_score"] == "rerank"
    assert payload["recalled"] > 1
    assert payload["returned"] == 1
    assert payload["dropped"] >= 1
    top = payload["hits"][0]
    assert top["scores"]["rerank"] == 0.9
    assert top["relevance"] == 0.9


def test_broken_reranker_falls_back_to_hybrid(client, kb, monkeypatch):
    upload(client, kb["id"], "E-201.md", SAMPLE)

    class BrokenReranker:
        name = "broken"
        model = "broken"

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            raise RuntimeError("service down")

    monkeypatch.setattr("openrag.services.retrieval.get_reranker", lambda: BrokenReranker())

    payload = client.post(
        "/v1/retrieve", json={"kb_id": kb["id"], "query": "机械密封", "top_k": 3}
    ).json()

    assert payload["rerank_applied"] is False
    assert payload["hits"]
