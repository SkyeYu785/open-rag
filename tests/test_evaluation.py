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
    metrics, flags = _case_metrics(hits, set(), ["0.8MPa"], k=5)

    assert metrics["hit"] == 1.0
    assert metrics["ndcg"] <= 1.0
    assert all(flags)


def test_ndcg_uses_each_reference_document_once():
    hits = [
        {"doc_id": "a", "text": "x"},
        {"doc_id": "a", "text": "x"},
        {"doc_id": "b", "text": "y"},
    ]
    metrics, _ = _case_metrics(hits, {"a", "b"}, [], k=3)

    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    # 相关文档出现在第 1、3 位，低于理想排序（第 1、2 位）
    assert round(metrics["ndcg"], 3) == 0.92


def test_dataset_cascade_delete(client, kb):
    upload(client, kb["id"], "E-201.md", SAMPLE)
    dataset = _dataset(client, kb["id"])
    _case(client, dataset["id"], {"question": "压力", "keywords": ["0.8MPa"]})
    run = client.post(f"/v1/eval/datasets/{dataset['id']}/runs", json={"top_k": 3}).json()

    assert client.delete(f"/v1/eval/datasets/{dataset['id']}").status_code == 204
    assert client.get(f"/v1/eval/runs/{run['id']}").status_code == 404
