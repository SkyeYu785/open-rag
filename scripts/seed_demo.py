"""Create a demo knowledge base, upload sample documents and build an eval set.

Usage::

    python scripts/seed_demo.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openrag import db  # noqa: E402
from openrag.config import get_settings  # noqa: E402
from openrag.services import ingestion  # noqa: E402

KB_NAME = "演示 · 工厂设备资料"

DOCUMENTS = {
    "E-201离心泵手册.md": """# E-201 离心泵

## 技术参数

型号 E-201，额定流量 120 m³/h，扬程 32 m，最大工作压力 0.8MPa，电机功率 45kW。
适用介质温度范围 -20℃ 至 120℃，介质密度不超过 1.2 g/cm³。

## 维护说明

每运行 2000 小时需要更换机械密封，并检查联轴器的同轴度，偏差不得超过 0.05mm。
若出口压力持续高于 0.8MPa，应立即停机并检查出口阀门是否误关闭。

## 常见故障

轴承温度超过 75℃ 时，优先检查润滑油位和冷却水流量。
泵体异常振动时，依次排查地脚螺栓松动、叶轮结垢和轴承磨损。
""",
    "螺杆空压机运维规程.md": """# 螺杆空压机运维规程

## 运行参数

型号 GA-75，额定排气压力 0.8MPa，排气量 12.5 m³/min。
排气温度正常范围为 75℃ 至 95℃，超过 100℃ 属于异常工况。

## 排气温度过高

当排气温度超过 100℃ 时，依次检查：冷却器是否积灰堵塞、冷却风扇是否运转、
润滑油位是否过低、温控阀是否卡滞。确认冷却器压差大于 0.1MPa 时应安排清洗。

## 保养周期

每 2000 小时更换空气滤芯，每 4000 小时更换油过滤器，每 8000 小时更换润滑油。
保养前必须停机、泄压并执行上锁挂牌。
""",
    "车间安全作业规范.md": """# 车间安全作业规范

## 上锁挂牌

检修任何带能量的设备前，必须执行上锁挂牌（LOTO）：切断能源、上锁、挂签、验证零能量。
多人作业时，每人使用独立锁具，钥匙由本人保管。

## 有限空间

进入有限空间前必须办理作业许可，检测氧含量在 19.5% 至 23.5% 之间，
可燃气体浓度低于爆炸下限的 10%，并安排监护人在外持续监护。

## 应急响应

发生人员伤害时，立即拨打厂内应急电话 8119，同时保护现场并上报安全科。
""",
}

CASES = [
    {"question": "E-201 离心泵的最大工作压力是多少？", "keywords": ["0.8MPa"]},
    {"question": "机械密封应该多久更换一次？", "keywords": ["2000 小时"]},
    {"question": "空压机排气温度超过 100℃ 要检查哪些部件？", "keywords": ["冷却器", "润滑油位"]},
    {"question": "进入有限空间作业前需要办理什么手续？", "keywords": ["作业许可"]},
]


def main() -> None:
    db.init_db()
    settings = get_settings()

    existing = db.query_one("SELECT * FROM knowledge_bases WHERE name = ?", (KB_NAME,))
    if existing:
        print(f"知识库「{KB_NAME}」已存在，跳过创建。")
        print(f"打开 http://127.0.0.1:8000 查看。")
        return

    kb_id = uuid.uuid4().hex
    timestamp = db.now()
    db.execute(
        "INSERT INTO knowledge_bases (id, name, description, version, created_at, updated_at)"
        " VALUES (?, ?, ?, 0, ?, ?)",
        (kb_id, KB_NAME, "示例数据：设备手册、运维规程与安全规范", timestamp, timestamp),
    )

    for name, content in DOCUMENTS.items():
        doc_id = uuid.uuid4().hex
        path = settings.upload_dir / f"{doc_id}.md"
        path.write_text(content, encoding="utf-8")
        db.execute(
            """
            INSERT INTO documents
                (id, kb_id, name, path, ext, size, status, error, chunk_count, char_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, '.md', ?, 'pending', NULL, 0, 0, ?, ?)
            """,
            (doc_id, kb_id, name, str(path), path.stat().st_size, timestamp, timestamp),
        )
        ingestion.process_document(doc_id)
        status = db.query_one("SELECT status, chunk_count FROM documents WHERE id = ?", (doc_id,))
        print(f"  {name}: {status['status']} · {status['chunk_count']} 个分块")

    dataset_id = uuid.uuid4().hex
    db.execute(
        "INSERT INTO eval_datasets (id, kb_id, name, description, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (dataset_id, kb_id, "设备问答 · 冒烟测试", "4 条标注好的问题，用来验证检索效果", timestamp, timestamp),
    )

    import json

    for case in CASES:
        db.execute(
            "INSERT INTO eval_cases (id, dataset_id, question, reference_docs, keywords, created_at)"
            " VALUES (?, ?, ?, '[]', ?, ?)",
            (
                uuid.uuid4().hex,
                dataset_id,
                case["question"],
                json.dumps(case["keywords"], ensure_ascii=False),
                db.now(),
            ),
        )

    print(f"已创建评测集「设备问答 · 冒烟测试」，共 {len(CASES)} 条用例。")
    print("启动服务：python -m openrag  →  http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
