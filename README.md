# dental-menual-kb-ingestion

这是一个面向牙医领域专业论文的知识库摄取项目。仓库把原始文档经过离线处理流水线，逐步转换为结构化块、语义块和向量索引，最后提供一个可供 AI 助手调用的在线混合检索服务。

目前的整体链路是：

1. 结构感知切块：从 PDF 中恢复章节结构、清洗页面噪声、生成结构块。
2. 语义切块：在结构边界内继续细分，得到更适合检索的语义 chunk。
3. 向量化入库：把可索引 chunk 写入 Milvus，并同步到 Elasticsearch 做词法检索。
4. 在线检索服务：通过 FastAPI 暴露检索接口，融合向量召回、BM25 和 rerank。

## 项目结构

```text
.
├─ src/
│  ├─ 01-structure_aware_chunking/   # Phase 01，PDF 结构恢复与规范化
│  ├─ 02-semantic_chunking/          # Phase 02，结构边界内的语义切块
│  ├─ 03-vectorization/              # Phase 03，向量化并写入 Milvus / Elasticsearch
│  └─ 04-online_rag_service/         # Phase 04，在线混合检索服务
├─ requirements/                     # 每个阶段单独的依赖文件
├─ data/
│  ├─ raw/                           # 原始 PDF
│  ├─ processed/                     # 各阶段输出产物
│  ├─ milvus/                        # 本地 Milvus 数据目录
│  └─ elasticsearch/                 # 本地 Elasticsearch 数据目录
├─ docs/                             # Phase 04 的设计与任务说明
├─ compose.yaml                      # 本地依赖服务与各阶段运行入口
└─ Dockerfile                        # 通用运行镜像
```

## 四个阶段

### Phase 01: `src/01-structure_aware_chunking`

- 使用 Marker 解析 PDF，恢复结构信息。
- 输出 `normalized_blocks`、`structural_chunks` 和 manifest。
- 整理与解析文档结构。

### Phase 02: `src/02-semantic_chunking`

- 读取 Phase 01 结果。
- 在已有结构边界内部做语义切块。
- 保留 `heading_path`、页码、来源 block 等 lineage 信息，方便后续检索与引用。

### Phase 03: `src/03-vectorization`

- 读取 Phase 02 语义块。
- 使用 OpenAI 兼容 embedding 接口生成向量。
- 将 dense 向量写入 Milvus，将 lexical 文档写入 Elasticsearch。

### Phase 04: `src/04-online_rag_service`

- 使用 FastAPI 提供 `/health`、`/reload`、`/retrieve` 接口。
- 检索时融合：
  - Milvus 向量召回
  - Elasticsearch 对 `embedding_text` 的 BM25
  - Elasticsearch 对 `heading_path_text` 的 BM25
- 最后再通过 rerank 模型排序结果。

## 运行依赖

- Python 3.12
- Docker / Docker Compose
- OpenAI 兼容 API 凭据
- Milvus 和 Elasticsearch

仓库已经提供：

- `compose.yaml`：启动 Milvus、Elasticsearch、Kibana，以及 4 个阶段的运行容器
- `requirements/phase01.txt` ~ `requirements/phase04.txt`：按阶段拆分依赖
- `.env.example`：环境变量模板

## 环境变量

先复制一份环境变量模板：

```powershell
Copy-Item .env.example .env
```

最关键的配置包括：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_RECALL_MODEL`
- `OPENAI_RERANK_MODEL`

如果本地用 Compose 启动 Milvus / Elasticsearch，通常不需要额外改：

- `MILVUS_URI`
- `ELASTICSEARCH_URL`

Phase 01 默认使用 Marker，并且默认启用 LLM 模式；如果没有 GPU，可以把 `PHASE01_TORCH_DEVICE=cpu` 写进 `.env` 或运行环境。

## 最小使用流程

### 1. 构建镜像

```powershell
docker compose build phase01 phase02 phase03 phase04
```

### 2. 启动基础服务

```powershell
docker compose up -d milvus-standalone elasticsearch kibana
```

### 3. 运行离线处理流水线

以仓库里的示例 PDF 为例：

```powershell
docker compose run --rm phase01 `
  python src/01-structure_aware_chunking/pipeline.py `
  --input-pdf data/raw/ManualClinProcDentistry-Sample.pdf `
  --emit-json

docker compose run --rm phase02 `
  python src/02-semantic_chunking/pipeline.py `
  --doc-id manualclinprocdentistry-sample

docker compose run --rm phase03 `
  python src/03-vectorization/pipeline.py `
  --doc-id manualclinprocdentistry-sample
```

### 4. 启动在线检索服务

```powershell
docker compose up -d phase04
```

服务默认映射到本机 `8010` 端口，对外提供：

- `GET /health`
- `POST /reload`
- `POST /retrieve`

## 数据与产物

离线产物默认写入：

- `data/processed/01_structure_aware`
- `data/processed/02_semantic_chunking`
- `data/processed/03_vectorization`
