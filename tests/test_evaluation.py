from __future__ import annotations

from conftest import SAMPLE, upload

from openrag.services.evaluation import _case_metrics


def _dataset(client, kb_id: str) -> dict:
    response = client.post(
        "/v1/eval/datasets", json={"kb_id": kb_id, "name": "运维问答", "description": "回归集"}
    )
    assert response.status_code == 201, response.text
    return response.json()


def _case(client, dataset_id: str, payload: dict) -> dict:
    response = client.post(f"/v1/eval/datasets/{dataset_id}/cases", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_full_evaluation_flow(client, kb):
    upload(client, kb["id"], "E-201.md", SAMPLE)
    dataset = _dataset(client, kb["id"])
    case = _case(client, dataset["id"], {"question": "最大工作压力是多少", "keywords": ["0.8MPa"]})

    assert case["keywords"] == ["0.8MPa"]
    assert client.get(f"/v1/eval/datasets/{dataset['id']}/cases").json()[0]["id"] == case["id"]

    run = client.post(f"/v1/eval/datasets/{dataset['id']}/runs", json={"top_k": 3, "mode": "hybrid"})
    assert run.status_code == 202, run.text

    detail = client.get(f"/v1/eval/runs/{run.json()['id']}").json()
    assert detail["status"] == "done", detail
    assert detail["metrics"]["hit_rate"] == 1.0
    assert detail["metrics"]["recall"] == 1.0
    assert detail["metrics"]["mrr"] > 0
    assert len(detail["results"]) == 1
    assert detail["results"][0]["hits"]


def test_case_requires_labels(client, kb):
    dataset = _dataset(client, kb["id"])
    response = client.post(
        f"/v1/eval/datasets/{dataset['id']}/cases", json={"question": "没有标注的问题"}
    )
    assert response.status_code == 400


def test_case_rejects_foreign_document(client, kb):
    dataset = _dataset(client, kb["id"])
    response = client.post(
        f"/v1/eval/datasets/{dataset['id']}/cases",
        json={"question": "问题", "reference_doc_ids": ["unknown-doc"]},
    )
    assert response.status_code == 400


def test_run_on_empty_dataset_fails(client, kb):
    upload(client, kb["id"], "E-201.md", SAMPLE)
    dataset = _dataset(client, kb["id"])
    run = client.post(f"/v1/eval/datasets/{dataset['id']}/runs", json={"top_k": 3})
    detail = client.get(f"/v1/eval/runs/{run.json()['id']}").json()
    assert detail["status"] == "failed"
    assert "用例" in detail["error"]


def test_ndcg_is_bounded_when_a_keyword_repeats():
    hits = [{"doc_id": f"doc-{index}", "text": "最大工作压力 0.8MPa"} for index in range(5)]
    metrics, flags = _case_metrics(hits, set(), set(), ["0.8MPa"], k=5)

    assert metrics["hit"] == 1.0
    assert metrics["ndcg"] <= 1.0
    assert all(flags)


def test_ndcg_uses_each_reference_document_once():
    hits = [
        {"doc_id": "a", "text": "x"},
        {"doc_id": "a", "text": "x"},
        {"doc_id": "b", "text": "y"},
    ]
    metrics, _ = _case_metrics(hits, {"a", "b"}, set(), [], k=3)

    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    # 相关文档出现在第 1、3 位，低于理想排序（第 1、2 位）
    assert round(metrics["ndcg"], 3) == 0.92


def test_chunk_level_labels_are_strict():
    hits = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "无关"},
        {"chunk_id": "c2", "doc_id": "d1", "text": "答案在这里"},
    ]
    metrics, flags = _case_metrics(hits, {"d1"}, {"c2"}, [], k=2)

    assert flags == [False, True]
    assert metrics["mrr"] == 0.5
    assert metrics["recall"] == 1.0


def test_chunk_level_labels_are_strict_in_a_run(client, kb):
    result = upload(client, kb["id"], "E-201.md", SAMPLE)
    doc_id = result["accepted"][0]["id"]
    items = client.get(f"/v1/documents/{doc_id}/chunks").json()["items"]
    target = next(item for item in items if "0.8MPa" in item["text"])

    dataset = _dataset(client, kb["id"])
    case = client.post(
        f"/v1/eval/datasets/{dataset['id']}/cases",
        json={"question": "E-201 最大工作压力是多少", "reference_chunk_ids": [target["id"]]},
    )
    assert case.status_code == 201, case.text
    assert case.json()["reference_chunk_ids"] == [target["id"]]

    run = client.post(f"/v1/eval/datasets/{dataset['id']}/runs", json={"top_k": 5}).json()
    detail = client.get(f"/v1/eval/runs/{run['id']}").json()

    assert detail["status"] == "done", detail
    assert detail["metrics"]["chunk_labeled"] == 1.0
    assert detail["metrics"]["doc_labeled"] == 0.0
    assert detail["metrics"]["hit_rate"] == 1.0
    assert detail["results"][0]["hits"][0]["chunk_id"] == target["id"]


def test_unknown_chunk_id_is_rejected(client, kb):
    dataset = _dataset(client, kb["id"])
    response = client.post(
        f"/v1/eval/datasets/{dataset['id']}/cases",
        json={"question": "问题", "reference_chunk_ids": ["unknown-chunk"]},
    )
    assert response.status_code == 400


def test_dataset_cascade_delete(client, kb):
    upload(client, kb["id"], "E-201.md", SAMPLE)
    dataset = _dataset(client, kb["id"])
    _case(client, dataset["id"], {"question": "压力", "keywords": ["0.8MPa"]})
    run = client.post(f"/v1/eval/datasets/{dataset['id']}/runs", json={"top_k": 3}).json()

    assert client.delete(f"/v1/eval/datasets/{dataset['id']}").status_code == 204
    assert client.get(f"/v1/eval/runs/{run['id']}").status_code == 404
