# open-rag

一个最小可用的**自托管 RAG 检索后端**：上传文档 → 解析分块 → 混合检索 → 检索评测。自带 Web UI，单进程、零外部依赖即可跑起来。

> 只做 RAG 里的 **R**（Retrieval），不绑定任何 LLM。这样它既能被 Dify / LangChain / 自研 Agent 直接当检索层用，也能独立做召回质量评测。

```
文档入库  PDF / DOCX / XLSX / CSV / Markdown / TXT / HTML
检索      向量（dense）+ BM25（sparse），RRF 融合，返回可引用的分块与页码
评测      Hit@k / Recall@k / MRR / nDCG@k，逐条可追溯到检索结果
```

## 特性

- **零外部服务**：SQLite 存元数据与向量，进程内暴力检索；不需要 Milvus / ES / Redis
- **中文友好**：BM25 使用汉字 uni/bi-gram，无需 jieba；混合检索对中文短查询更稳
- **可插拔 embedding**：内置离线 hash 向量兜底，支持 sentence-transformers（BGE 等）与任意 OpenAI 兼容接口
- **可评测**：自带标注数据管理与评测页，改 chunk 策略 / 换模型后能立刻看到指标变化
- **两个页面**：知识库文档管理、检索评测；无构建步骤（原生 ES Module + CSS）

## 架构

```
                    ┌──────────────────────────────────────────┐
   Web UI (/)  ───► │                                          │
                    │   FastAPI  /v1/*                         │
   HTTP 客户端 ───► │                                          │
                    └───────┬──────────────────────┬───────────┘
                            │                      │
                    ┌───────▼────────┐     ┌───────▼────────┐
                    │  ingestion     │     │  retrieval     │
                    │  parse → chunk │     │  vector + BM25 │
                    │  → embed       │     │  → RRF 融合     │
                    └───────┬────────┘     └───────┬────────┘
                            │                      │
                    ┌───────▼──────────────────────▼────────┐
                    │  SQLite：documents / chunks / 向量 BLOB │
                    └───────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 环境（Python 3.10+）
conda create -n open-rag python=3.12 -y
conda activate open-rag
pip install -e ".[dev]"

# 2. 启动
python -m openrag
# → http://127.0.0.1:8000        Web UI
# → http://127.0.0.1:8000/docs    OpenAPI 文档

# 3.（可选）灌入演示数据：3 篇设备资料 + 1 个评测集
python scripts/seed_demo.py
```

## Embedding 配置

默认 `auto`：装了 `sentence-transformers` 就用本地模型，否则退化为内置 hash 向量（离线可用，但只有词面相似度）。

| Provider | 说明 | 适合 |
| --- | --- | --- |
| `hash` | 内置，零依赖，512 维特征哈希 | 快速体验 / 离线环境 / 单元测试 |
| `sentence-transformers` | 推荐 `BAAI/bge-small-zh-v1.5`（约 100MB） | 生产、中文语义检索 |
| `openai` | 任意 OpenAI 兼容 `/embeddings`（OpenAI、DashScope、DeepSeek、Ollama、vLLM） | 已有模型服务 |

```bash
# 本地模型
pip install -e ".[local]"
export OPENRAG_EMBEDDING_PROVIDER=sentence-transformers
export OPENRAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 或复用已有的模型服务
export OPENRAG_EMBEDDING_PROVIDER=openai
export OPENRAG_EMBEDDING_BASE_URL=http://localhost:11434/v1
export OPENRAG_EMBEDDING_MODEL=bge-m3

python -m openrag
```

> ⚠️ 同一个知识库必须始终使用同一个 embedding 模型。切换模型后新建知识库重新导入，否则检索会明确报错（而不是静默返回错误结果）。

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/v1/health` | 健康检查与当前 embedding 配置 |
| GET/POST | `/v1/knowledge-bases` | 知识库列表 / 创建 |
| GET/PATCH/DELETE | `/v1/knowledge-bases/{id}` | 知识库详情 / 修改 / 删除 |
| GET/POST | `/v1/knowledge-bases/{id}/documents` | 文档列表 / 上传（multipart，字段名 `files`） |
| GET/DELETE | `/v1/documents/{id}` | 文档详情 / 删除 |
| POST | `/v1/documents/{id}/reindex` | 重新解析与向量化 |
| GET | `/v1/documents/{id}/chunks` | 分块预览（`offset` / `limit`） |
| POST | `/v1/retrieve` | 检索：`{kb_id, query, top_k, mode}` |
| GET/POST | `/v1/eval/datasets` | 评测集列表 / 创建 |
| GET/POST | `/v1/eval/datasets/{id}/cases` | 用例列表 / 添加 |
| POST | `/v1/eval/datasets/{id}/runs` | 发起一次评测（异步） |
| GET | `/v1/eval/runs/{id}` | 评测状态、指标与逐条结果 |

```bash
# 创建知识库
curl -s -X POST localhost:8000/v1/knowledge-bases \
  -H 'Content-Type: application/json' \
  -d '{"name":"设备手册","description":"工厂设备资料"}'

# 上传文档
curl -s -X POST localhost:8000/v1/knowledge-bases/<KB_ID>/documents \
  -F "files=@E-201离心泵手册.md"

# 检索
curl -s -X POST localhost:8000/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"kb_id":"<KB_ID>","query":"E-201 最大工作压力","top_k":5,"mode":"hybrid"}'
```

检索返回示例：

```json
{
  "query": "E-201 最大工作压力",
  "mode": "hybrid",
  "total_chunks": 6,
  "latency_ms": 1.8,
  "hits": [
    {
      "chunk_id": "…",
      "doc_id": "…",
      "doc_name": "E-201离心泵手册.md",
      "page": null,
      "text": "[E-201 离心泵 > 技术参数]\n型号 E-201，额定流量 120 m³/h……最大工作压力 0.8MPa……",
      "score": 0.0328,
      "scores": {"vector": 0.61, "keyword": 4.27, "fused": 0.0328}
    }
  ]
}
```

`mode` 可选 `hybrid`（默认）/ `vector` / `keyword`。

## 评测怎么用

评测回答一个问题：**改完解析、分块或模型，检索到底变好了还是变差了？**

1. 在「评测」页新建评测集（绑定一个知识库）
2. 添加用例，每条用例提供至少一种标注：
   - **关键词**：答案里必须出现的字符串（比如 `0.8MPa`），适合答案可核验的事实型问题
   - **参考文档**：答案所在的文档，适合"这个问题该从哪份资料里找"
3. 点「开始评测」，得到：

| 指标 | 含义 |
| --- | --- |
| Hit@k | Top-k 里至少有一条相关结果；衡量"能不能找到" |
| Recall@k | 相关文档 / 关键词被召回的比例；衡量"找全了没有" |
| MRR | 第一条相关结果排名的倒数均值；衡量"排得靠不靠前" |
| nDCG@k | 考虑排序位置的相关性质量；衡量"整体排序好不好" |

建议先用 **20–50 条真实问题**建立起基线，再调 `OPENRAG_CHUNK_SIZE`、`top_k`、embedding 模型。指标不动就不要继续加功能。

## 配置

所有配置通过环境变量（或项目根目录的 `.env`，参考 `.env.example`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENRAG_DATA_DIR` | `data` | 数据目录（SQLite、上传文件） |
| `OPENRAG_EMBEDDING_PROVIDER` | `auto` | `auto` / `hash` / `sentence-transformers` / `openai` |
| `OPENRAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 模型名 |
| `OPENRAG_EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | 仅 `openai` provider |
| `OPENRAG_EMBEDDING_API_KEY` | – | 仅 `openai` provider |
| `OPENRAG_EMBEDDING_DIM` | `512` | 仅 `hash` provider 的维度 |
| `OPENRAG_CHUNK_SIZE` | `700` | 分块目标长度（字符） |
| `OPENRAG_CHUNK_OVERLAP` | `100` | 相邻分块重叠字符数 |
| `OPENRAG_MAX_UPLOAD_MB` | `50` | 单文件大小上限 |
| `OPENRAG_HOST` / `OPENRAG_PORT` | `127.0.0.1` / `8000` | 监听地址 |
| `OPENRAG_CORS_ORIGINS` | `*` | 逗号分隔的允许来源 |

分块策略：按 Markdown 标题切段 → 按句子边界切分 → 合并到目标长度 → 相邻块保留重叠；标题路径会作为前缀写入分块文本，便于检索和引用。

## 目录结构

```
openrag/
  main.py              FastAPI 应用与静态页面挂载
  config.py            环境变量配置
  db.py                SQLite schema 与访问层
  models.py            Pydantic 请求/响应模型
  api/                 knowledge_bases · documents · retrieval · evaluation
  services/
    parsing.py         PDF / DOCX / XLSX / CSV / HTML / 文本解析
    chunking.py        标题感知的分块
    embedding.py       hash / sentence-transformers / openai
    bm25.py            中文 uni/bi-gram BM25
    retrieval.py       向量 + BM25 + RRF
    ingestion.py       解析 → 分块 → 向量化 → 入库
    evaluation.py      指标计算与评测任务
  web/                 零构建 Web UI（index.html / app.js / styles.css）
scripts/seed_demo.py   演示数据
tests/                 API 与单元测试
```

## 测试

```bash
pytest
```

测试使用临时数据目录与 hash embedder，不依赖网络或模型下载。

## 设计取舍与已知限制

- **单机 SQLite + 内存暴力检索**：目标是单机 10 万级分块以内、开箱即用。再大请换 Milvus / Qdrant（`retrieval.py` 的 `get_index` 是唯一入口）
- **不做 OCR**：扫描件 PDF 会明确报错，而不是返回空结果
- **没有鉴权与多租户**：请只在内网或反向代理之后暴露
- **没有 LLM 生成**：本项目的产出是「带分数的上下文」，生成交给调用方，避免把模型选择绑定进检索层
- **BM25 索引在进程内构建**：知识库发生变化时按 `version` 自动重建，重启后首次查询会重建一次

## Roadmap

- [ ] 可插拔向量后端（Qdrant / pgvector）
- [ ] 扫描件 OCR（PaddleOCR / MinerU）与表格结构化
- [ ] 分块策略可配置（语义分块、父子块）
- [ ] 评测：LLM-as-judge 的答案质量指标、A/B 对比视图
- [ ] API Key 鉴权与知识库级权限

## License

MIT
