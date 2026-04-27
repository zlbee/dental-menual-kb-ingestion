# 图片 VLM 描述 + 图片向量入库改造思路

## Summary
- 目标：图片块最终同时包含 `文字描述/图注` 和 `image_embedding`，可被文本 query 检索出来，并能返回原图路径。
- 采用你选择的方案：复用 Marker 生成的图片描述，不单独再调 VLM；额外用 OpenAI 兼容的多模态 embedding 模型生成图片向量。
- 存储采用同一个 Milvus collection 双向量字段：现有文本向量 `embedding` 保留，新增图片向量 `image_embedding`。

## Key Changes
- Phase01：保留现有 Marker JSON 解析，同时基于 Marker 的 `Figure/Picture` block `bbox/polygon` 用 PyMuPDF 从 PDF 裁出图片资产，写入 `data/processed/01_structure_aware/image_assets/<doc_id>/...`；`normalized_blocks` 增加 `media_assets`，记录图片路径、页码、bbox、marker block id、sha256。
- Phase02：继续把 `figure/picture + caption` 合并为 `visual` chunk；visual chunk 的 `display_text` 使用 Marker 图片描述 + 图注，metadata 透传 `media_assets`，并标记 `has_image=true`。
- Phase03：新增图片 embedding 客户端，读取 visual chunk 的图片文件生成 `image_embedding`；Milvus schema 增加 `image_embedding`、`has_image`，文本块用零向量占位并通过 `has_image == true` 过滤图片召回。
- Phase04：文本 query 同时走三路召回：现有文本向量、Elasticsearch BM25、新增图片向量文本查询；用 RRF 融合后继续用现有 rerank，对 visual 结果返回 `display_text`、`retrieval_text`、`media_assets` 和 citation。
- Collection 命名需要加入 text embedding model + image embedding model，避免旧 collection schema 不兼容；启用该能力后需要重跑 Phase01-Phase03 或新建 collection。

## Interfaces
- 新增环境变量：`IMAGE_EMBEDDING_MODEL` 必填；`IMAGE_EMBEDDING_BASE_URL` 默认复用 `OPENAI_BASE_URL`；`IMAGE_EMBEDDING_API_KEY` 默认复用 `OPENAI_API_KEY`；`IMAGE_EMBEDDING_DIM` 可选，未填则用首个图片向量推断。
- 新增 Phase01 开关建议为 `--export-image-assets`，启用时自动要求/触发 JSON artifact；保留当前默认不破坏旧文本流水线。
- Phase04 `/retrieve` 返回结构保持兼容，只在结果 metadata/citation 中增加 `media_assets`、`has_image`、`image_embedding_model` 等字段。

## Test Plan
- 用现有 sample PDF 跑 Phase01，验证 `Figure/Picture` 有图片文件、bbox、sha256，caption-only 块不会伪造图片路径。
- 跑 Phase02，验证 visual chunk 合并图片描述和图注，`media_assets` 没丢。
- 跑 Phase03，验证 Milvus 新 schema 有两个向量字段，visual chunk 有真实 `image_embedding`，文本 chunk 仅占位且 `has_image=false`。
- 跑 Phase04 检索：例如“world map population cartogram”“dental x-ray artefact”能召回 visual chunk，并返回原图路径和页码。

## Assumptions
- 图片 embedding 后端支持同一个 OpenAI 兼容接口对图片和文本 query 生成同一向量空间的 embedding。
- 首版只支持“文本查图”，不做“以图搜图”上传查询。
- 为避免 Marker segmented 模式的图片路径问题，原图导出不依赖 Marker 图片导出，而是用 Marker JSON 坐标从 PDF 裁图。
