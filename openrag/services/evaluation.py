"""Retrieval evaluation.

A case is a question plus ground truth in either of two forms:

* ``reference_doc_ids`` – the documents that contain the answer (doc level).
* ``keywords``           – strings that must appear in the retrieved context
                           (answer level, useful when a document is long).

A retrieved chunk counts as relevant when it belongs to a reference document or
contains one of the keywords. Metrics are then computed on the top-k ranking.
"""

from __future__ import annotations

import json
import logging
import math
import uuid

import numpy as np

from .. import db
from . import retrieval
from .text import snippet

logger = logging.getLogger(__name__)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _case_metrics(
    hits: list[dict],
    references: set[str],
    reference_chunks: set[str],
    keywords: list[str],
    k: int,
) -> tuple[dict, list[bool]]:
    lowercase_keywords = [kw.strip().lower() for kw in keywords if kw and kw.strip()]

    # ``flags`` marks every chunk that touches the ground truth (used for hit/MRR
    # and for highlighting in the UI). ``gains`` counts each annotated unit only
    # once, otherwise a single keyword repeated in many chunks would push nDCG
    # above 1.
    flags: list[bool] = []
    gains: list[float] = []
    covered: set[tuple[str, str]] = set()

    for hit in hits:
        if reference_chunks:
            units: set[tuple[str, str]] = (
                {("chunk", hit["chunk_id"])} if hit["chunk_id"] in reference_chunks else set()
            )
        else:
            # judge what the generator would actually read: the matched child plus
            # its parent context (small-to-big retrieval)
            text = (hit["text"] + "\n" + (hit.get("context") or "")).lower()
            units = set()
            if hit["doc_id"] in references:
                units.add(("doc", hit["doc_id"]))
            for keyword in lowercase_keywords:
                if keyword in text:
                    units.add(("keyword", keyword))

        flags.append(bool(units))
        gains.append(1.0 if units - covered else 0.0)
        covered |= units

    if reference_chunks:
        gold_count = len(reference_chunks)
        retrieved_chunks = {hit["chunk_id"] for hit in hits}
        recall = len(retrieved_chunks & reference_chunks) / gold_count
    elif references:
        gold_count = len(references)
        retrieved_docs = {hit["doc_id"] for hit in hits}
        recall = len(retrieved_docs & references) / gold_count
    else:
        gold_count = len(lowercase_keywords)
        joined = "\n".join(
            hit["text"] + "\n" + (hit.get("context") or "") for hit in hits
        ).lower()
        recall = sum(1 for kw in lowercase_keywords if kw in joined) / max(gold_count, 1)

    hit_at_k = 1.0 if any(flags) else 0.0

    mrr = 0.0
    for position, relevant in enumerate(flags, start=1):
        if relevant:
            mrr = 1.0 / position
            break

    dcg = sum(1.0 / math.log2(position + 1) for position, gain in enumerate(gains, start=1) if gain)
    ideal = min(gold_count, k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal + 1))
    ndcg = dcg / idcg if idcg else 0.0

    return {"hit": hit_at_k, "recall": recall, "mrr": mrr, "ndcg": min(ndcg, 1.0)}, flags


def execute_run(run_id: str) -> None:
    """Run an evaluation job. Meant to be executed in a background thread."""
    run = db.query_one("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
    if run is None:
        return

    dataset = db.query_one("SELECT * FROM eval_datasets WHERE id = ?", (run["dataset_id"],))
    db.execute("UPDATE eval_runs SET status = 'running' WHERE id = ?", (run_id,))

    try:
        if dataset is None:
            raise RuntimeError("评测集不存在")

        cases = db.query_all(
            "SELECT * FROM eval_cases WHERE dataset_id = ? ORDER BY created_at, rowid",
            (dataset["id"],),
        )
        if not cases:
            raise RuntimeError("评测集中还没有用例")

        config = db.loads(run["config"], {})
        top_k = int(config.get("top_k", 5))
        mode = str(config.get("mode", "hybrid"))
        kb_id = dataset["kb_id"]

        index = retrieval.get_index(kb_id)
        if index.size == 0:
            raise RuntimeError("知识库还没有可检索的分块，请先上传文档")

        rows: list[tuple] = []
        scored: list[dict] = []
        latencies: list[float] = []
        skipped = 0
        chunk_labeled = 0
        doc_labeled = 0
        keyword_labeled = 0

        for case in cases:
            references = set(db.loads(case["reference_docs"], []))
            reference_chunks = set(db.loads(case.get("reference_chunks"), []))
            keywords = db.loads(case["keywords"], [])
            if not references and not reference_chunks and not keywords:
                skipped += 1
                continue

            chunk_labeled += 1 if reference_chunks else 0
            doc_labeled += 1 if references else 0
            keyword_labeled += 1 if keywords else 0

            hits, _, latency_ms, _ = retrieval.retrieve(
                kb_id, case["question"], top_k=top_k, mode=mode, index=index
            )
            metrics, flags = _case_metrics(hits, references, reference_chunks, keywords, top_k)
            latencies.append(latency_ms)

            serialised_hits = []
            for hit, relevant in zip(hits, flags):
                serialised_hits.append(
                    {
                        "chunk_id": hit["chunk_id"],
                        "doc_id": hit["doc_id"],
                        "doc_name": hit["doc_name"],
                        "page": hit["page"],
                        "ordinal": hit["ordinal"],
                        "tokens": hit.get("tokens", 0),
                        "score": round(hit["score"], 6),
                        "relevant": bool(relevant),
                        "preview": snippet(hit["text"], 280),
                    }
                )

            scored.append(metrics)
            rows.append(
                (
                    uuid.uuid4().hex,
                    run_id,
                    case["id"],
                    case["question"],
                    json.dumps({key: round(value, 6) for key, value in metrics.items()}),
                    json.dumps(serialised_hits, ensure_ascii=False),
                    latency_ms,
                )
            )

        if not scored:
            raise RuntimeError("所有用例都缺少标注（reference_doc_ids / keywords），无法评测")

        keys = ("hit", "recall", "mrr", "ndcg")
        aggregate = {key: float(np.mean([item[key] for item in scored])) for key in keys}
        metrics_payload = {
            "k": float(top_k),
            "cases": float(len(scored)),
            "skipped": float(skipped),
            "chunk_labeled": float(chunk_labeled),
            "doc_labeled": float(doc_labeled),
            "keyword_labeled": float(keyword_labeled),
            "hit_rate": round(aggregate["hit"], 4),
            "recall": round(aggregate["recall"], 4),
            "mrr": round(aggregate["mrr"], 4),
            "ndcg": round(aggregate["ndcg"], 4),
            "latency_p50": round(_percentile(latencies, 50), 2),
            "latency_p95": round(_percentile(latencies, 95), 2),
        }

        with db.transaction() as conn:
            conn.execute("DELETE FROM eval_results WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT INTO eval_results (id, run_id, case_id, question, metrics, hits, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.execute(
                "UPDATE eval_runs SET status = 'done', metrics = ?, finished_at = ? WHERE id = ?",
                (json.dumps(metrics_payload), db.now(), run_id),
            )
        logger.info("eval run %s finished: %s", run_id, metrics_payload)

    except Exception as exc:  # noqa: BLE001 - reported through the run record
        logger.warning("eval run %s failed: %s", run_id, exc)
        db.execute(
            "UPDATE eval_runs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
            (str(exc)[:1000], db.now(), run_id),
        )
