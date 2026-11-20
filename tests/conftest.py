"""Test fixtures: run everything against a throwaway data directory."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["OPENRAG_DATA_DIR"] = tempfile.mkdtemp(prefix="openrag-test-")
os.environ["OPENRAG_EMBEDDING_PROVIDER"] = "hash"
os.environ["OPENRAG_EMBEDDING_DIM"] = "256"
os.environ["OPENRAG_CHUNK_SIZE"] = "320"
os.environ["OPENRAG_CHUNK_OVERLAP"] = "40"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from openrag import db  # noqa: E402
from openrag.main import app  # noqa: E402

TABLES = (
    "eval_results",
    "eval_runs",
    "eval_cases",
    "eval_datasets",
    "chunks",
    "documents",
    "knowledge_bases",
)


@pytest.fixture(autouse=True)
def clean_database():
    db.init_db()
    conn = db.get_conn()
    for table in TABLES:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    yield


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def kb(client):
    response = client.post("/v1/knowledge-bases", json={"name": "设备手册", "description": "测试"})
    assert response.status_code == 201, response.text
    return response.json()


SAMPLE = """# E-201 离心泵

## 技术参数

型号 E-201，额定流量 120 m³/h，最大工作压力 0.8MPa，电机功率 45kW。
适用介质温度范围 -20℃ 至 120℃。

## 维护说明

每运行 2000 小时需要更换机械密封，并检查联轴器的同轴度。
若出口压力持续高于 0.8MPa，应立即停机并检查出口阀门。

## 常见故障

轴承温度超过 75℃ 时，优先检查润滑油位和冷却水流量。
"""


def upload(client, kb_id: str, name: str, content: str):
    response = client.post(
        f"/v1/knowledge-bases/{kb_id}/documents",
        files=[("files", (name, content.encode("utf-8"), "text/plain"))],
    )
    assert response.status_code == 202, response.text
    return response.json()
