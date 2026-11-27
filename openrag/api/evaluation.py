"""Evaluation datasets, cases and runs."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from .. import db
from ..models import (
    CaseCreate,
    CaseOut,
    DatasetCreate,
    DatasetOut,
    RunCreate,
    RunOut,
)
from ..services import evaluation
from .serializers import case_payload, dataset_payload, get_kb_or_404

router = APIRouter(prefix="/eval", tags=["evaluation"])

_DATASET_SQL = """
SELECT ds.*, (SELECT COUNT(*) FROM eval_cases c WHERE c.dataset_id = ds.id) AS case_count
FROM eval_datasets ds
"""

_RUN_SQL = """
SELECT r.*, (SELECT COUNT(*) FROM eval_results er WHERE er.run_id = r.id) AS case_count
FROM eval_runs r
"""


def _get_dataset(dataset_id: str) -> dict:
    row = db.query_one(_DATASET_SQL + " WHERE ds.id = ?", (dataset_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="评测集不存在")
    return row


def _run_payload(row: dict, include_results: bool = False) -> dict:
    payload = {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "kb_id": row["kb_id"],
        "status": row["status"],
        "config": db.loads(row["config"], {}),
        "metrics": db.loads(row["metrics"], {}),
        "error": row.get("error"),
        "created_at": row["created_at"],
        "finished_at": row.get("finished_at"),
        "case_count": row.get("case_count", 0),
        "results": None,
    }
    if include_results:
        results = db.query_all(
            "SELECT * FROM eval_results WHERE run_id = ? ORDER BY rowid", (row["id"],)
        )
        payload["results"] = [
            {
                "case_id": item["case_id"],
                "question": item["question"],
                "metrics": db.loads(item["metrics"], {}),
                "latency_ms": round(item["latency_ms"], 2),
                "hits": db.loads(item["hits"], []),
            }
            for item in results
        ]
    return payload


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(kb_id: str | None = Query(default=None)) -> list[dict]:
    if kb_id:
        rows = db.query_all(_DATASET_SQL + " WHERE ds.kb_id = ? ORDER BY ds.created_at DESC", (kb_id,))
    else:
        rows = db.query_all(_DATASET_SQL + " ORDER BY ds.created_at DESC")
    return [dataset_payload(row) for row in rows]


@router.post("/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate) -> dict:
    get_kb_or_404(payload.kb_id)
    dataset_id = uuid.uuid4().hex
    timestamp = db.now()
    db.execute(
        "INSERT INTO eval_datasets (id, kb_id, name, description, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (dataset_id, payload.kb_id, payload.name.strip(), payload.description.strip(), timestamp, timestamp),
    )
    return dataset_payload(_get_dataset(dataset_id))


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str) -> None:
    _get_dataset(dataset_id)
    db.execute("DELETE FROM eval_datasets WHERE id = ?", (dataset_id,))


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #
@router.get("/datasets/{dataset_id}/cases", response_model=list[CaseOut])
def list_cases(dataset_id: str) -> list[dict]:
    _get_dataset(dataset_id)
    rows = db.query_all(
        "SELECT * FROM eval_cases WHERE dataset_id = ? ORDER BY created_at, rowid", (dataset_id,)
    )
    return [case_payload(row) for row in rows]


@router.post("/datasets/{dataset_id}/cases", response_model=CaseOut, status_code=201)
def create_case(dataset_id: str, payload: CaseCreate) -> dict:
    dataset = _get_dataset(dataset_id)
    keywords = list(dict.fromkeys(k.strip() for k in payload.keywords if k.strip()))
    if not payload.reference_doc_ids and not payload.reference_chunk_ids and not keywords:
        raise HTTPException(status_code=400, detail="请至少提供参考文档、参考分块或关键词标注")

    valid_docs = {
        row["id"] for row in db.query_all("SELECT id FROM documents WHERE kb_id = ?", (dataset["kb_id"],))
    }
    unknown_docs = [doc_id for doc_id in payload.reference_doc_ids if doc_id not in valid_docs]
    if unknown_docs:
        raise HTTPException(status_code=400, detail="参考文档不属于该知识库")

    valid_chunks = {
        row["id"]
        for row in db.query_all(
            "SELECT id FROM chunks WHERE kb_id = ? AND level = 'child'", (dataset["kb_id"],)
        )
    }
    unknown_chunks = [chunk_id for chunk_id in payload.reference_chunk_ids if chunk_id not in valid_chunks]
    if unknown_chunks:
        raise HTTPException(status_code=400, detail="参考分块不属于该知识库")

    case_id = uuid.uuid4().hex
    db.execute(
        "INSERT INTO eval_cases"
        " (id, dataset_id, question, reference_docs, reference_chunks, keywords, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            case_id,
            dataset_id,
            payload.question.strip(),
            json.dumps(sorted(set(payload.reference_doc_ids))),
            json.dumps(sorted(set(payload.reference_chunk_ids))),
            json.dumps(keywords, ensure_ascii=False),
            db.now(),
        ),
    )
    db.execute("UPDATE eval_datasets SET updated_at = ? WHERE id = ?", (db.now(), dataset_id))
    row = db.query_one("SELECT * FROM eval_cases WHERE id = ?", (case_id,))
    return case_payload(row)


@router.delete("/cases/{case_id}", status_code=204)
def delete_case(case_id: str) -> None:
    row = db.query_one("SELECT dataset_id FROM eval_cases WHERE id = ?", (case_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="用例不存在")
    db.execute("DELETE FROM eval_cases WHERE id = ?", (case_id,))
    db.execute("UPDATE eval_datasets SET updated_at = ? WHERE id = ?", (db.now(), row["dataset_id"]))


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
@router.post("/datasets/{dataset_id}/runs", response_model=RunOut, status_code=202)
def create_run(dataset_id: str, payload: RunCreate, background: BackgroundTasks) -> dict:
    dataset = _get_dataset(dataset_id)
    run_id = uuid.uuid4().hex
    db.execute(
        "INSERT INTO eval_runs (id, dataset_id, kb_id, status, config, metrics, created_at)"
        " VALUES (?, ?, ?, 'pending', ?, '{}', ?)",
        (
            run_id,
            dataset_id,
            dataset["kb_id"],
            json.dumps({"top_k": payload.top_k, "mode": payload.mode}),
            db.now(),
        ),
    )
    background.add_task(evaluation.execute_run, run_id)
    return _run_payload(db.query_one(_RUN_SQL + " WHERE r.id = ?", (run_id,)))


@router.get("/runs", response_model=list[RunOut])
def list_runs(dataset_id: str | None = Query(default=None)) -> list[dict]:
    if dataset_id:
        rows = db.query_all(
            _RUN_SQL + " WHERE r.dataset_id = ? ORDER BY r.created_at DESC, r.rowid DESC", (dataset_id,)
        )
    else:
        rows = db.query_all(_RUN_SQL + " ORDER BY r.created_at DESC, r.rowid DESC")
    return [_run_payload(row) for row in rows]


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str) -> dict:
    row = db.query_one(_RUN_SQL + " WHERE r.id = ?", (run_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return _run_payload(row, include_results=True)
