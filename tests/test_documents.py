from __future__ import annotations

from conftest import SAMPLE, upload


def test_health(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["embedding"]["provider"] == "hash"


def test_knowledge_base_crud(client, kb):
    listing = client.get("/v1/knowledge-bases").json()
    assert [item["name"] for item in listing] == ["设备手册"]

    duplicate = client.post("/v1/knowledge-bases", json={"name": "设备手册"})
    assert duplicate.status_code == 409

    patched = client.patch(f"/v1/knowledge-bases/{kb['id']}", json={"description": "泵类"})
    assert patched.status_code == 200
    assert patched.json()["description"] == "泵类"

    assert client.delete(f"/v1/knowledge-bases/{kb['id']}").status_code == 204
    assert client.get(f"/v1/knowledge-bases/{kb['id']}").status_code == 404


def test_upload_parse_and_retrieve(client, kb):
    result = upload(client, kb["id"], "E-201.md", SAMPLE)
    assert result["rejected"] == []

    documents = client.get(f"/v1/knowledge-bases/{kb['id']}/documents").json()
    assert len(documents) == 1
    assert documents[0]["status"] == "ready", documents[0]
    assert documents[0]["chunk_count"] > 0

    chunks = client.get(f"/v1/documents/{documents[0]['id']}/chunks").json()
    assert chunks["total"] == documents[0]["chunk_count"]
    assert all(item["level"] == "child" for item in chunks["items"])
    assert all(item["tokens"] > 0 for item in chunks["items"])
    assert any("0.8MPa" in item["text"] for item in chunks["items"])

    response = client.post(
        "/v1/retrieve",
        json={"kb_id": kb["id"], "query": "E-201 最大工作压力是多少", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["hits"], payload
    assert payload["total_chunks"] >= 1
    assert any("0.8MPa" in hit["text"] for hit in payload["hits"])
    assert payload["hits"][0]["doc_name"] == "E-201.md"

    top = payload["hits"][0]
    assert top["tokens"] > 0
    assert top["parent_id"]
    assert top["context"]
    assert len(top["context"]) >= len(top["text"])
    assert top["meta"]["heading"].startswith("E-201 离心泵")


def test_retrieve_can_skip_parent_context(client, kb):
    upload(client, kb["id"], "E-201.md", SAMPLE)
    payload = client.post(
        "/v1/retrieve",
        json={"kb_id": kb["id"], "query": "机械密封", "include_context": False},
    ).json()

    assert payload["hits"]
    assert all(hit["context"] is None for hit in payload["hits"])


def test_retrieval_modes(client, kb):
    upload(client, kb["id"], "E-201.md", SAMPLE)
    for mode in ("hybrid", "vector", "keyword"):
        response = client.post(
            "/v1/retrieve",
            json={"kb_id": kb["id"], "query": "机械密封更换", "top_k": 3, "mode": mode},
        )
        assert response.status_code == 200, response.text
        assert response.json()["hits"], mode


def test_retrieve_unknown_kb(client):
    response = client.post("/v1/retrieve", json={"kb_id": "nope", "query": "x"})
    assert response.status_code == 404


def test_unsupported_file_is_rejected(client, kb):
    result = upload(client, kb["id"], "virus.exe", "MZ")
    assert result["accepted"] == []
    assert result["rejected"][0]["name"] == "virus.exe"


def test_delete_document_removes_chunks(client, kb):
    result = upload(client, kb["id"], "E-201.md", SAMPLE)
    doc_id = result["accepted"][0]["id"]

    assert client.delete(f"/v1/documents/{doc_id}").status_code == 204
    assert client.get(f"/v1/documents/{doc_id}").status_code == 404
    assert client.get(f"/v1/documents/{doc_id}/chunks").status_code == 404

    hits = client.post("/v1/retrieve", json={"kb_id": kb["id"], "query": "压力"}).json()["hits"]
    assert hits == []
