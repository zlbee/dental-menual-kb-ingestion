# Phase 04: Online RAG Service

This stage serves hybrid retrieval for the AI assistant on top of phase-03
Milvus vectors and Elasticsearch lexical documents.

## Retrieval Design

Phase 04 reads dense vectors and response metadata from Milvus, and reads
lexical recall candidates from Elasticsearch.

At startup it loads the `indexable` rows from the Milvus collection for
neighbor lookup, result shaping, and reranking.

At query time the service runs three first-stage retrieval lanes:

- dense vector recall against Milvus `embedding` using `OPENAI_RECALL_MODEL`
- Elasticsearch BM25 over `embedding_text`
- Elasticsearch BM25 over `heading_path_text`

The three lanes are fused with weighted reciprocal rank fusion (RRF), then the
candidate set is reranked through an external rerank API using
`OPENAI_RERANK_MODEL`. The final score is:

- `final_score = rerank_relevance_score + rerank_fusion_boost * fused_rrf`

## Data Contract

Phase 04 expects:

- the phase-03 Milvus collection schema defined in `src/03-vectorization/pipeline.py`
- the phase-03 Elasticsearch lexical index published by the same pipeline

It reads these Milvus fields:

- scalar fields: `chunk_id`, `doc_id`, `chunk_type`, `content_modality`,
  `document_title`, `section_title`, `page_start`, `page_end`,
  `prev_chunk_id`, `next_chunk_id`, `display_text`, `embedding_text`
- JSON field: `metadata`
- vector field: `embedding`

It queries these Elasticsearch fields:

- `embedding_text`
- `heading_path_text`
- `doc_id`, `chunk_type`, `content_modality`, `document_title`,
  `section_title`, `page_start`, `page_end`, `indexable`

The response also exposes citation-friendly metadata such as:

- `metadata.heading_path`
- `metadata.source_block_ids`
- `metadata.source_marker_block_ids`

## Endpoints

- `GET /health`
- `POST /reload`
- `POST /retrieve`

Example request:

```json
{
  "query": "How should the dental team position a patient for crown preparation?",
  "top_k": 5,
  "include_neighbors": true,
  "max_neighbors_per_side": 1,
  "filters": {
    "content_modalities": ["text"],
    "page_from": 30,
    "page_to": 120
  }
}
```

## Required Environment Variables

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_RECALL_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `IMAGE_EMBEDDING_MODEL`
- `OPENAI_RERANK_MODEL`

Optional image embedding overrides:

- `IMAGE_EMBEDDING_BASE_URL` (defaults to `OPENAI_BASE_URL`)
- `IMAGE_EMBEDDING_API_KEY` (defaults to `OPENAI_API_KEY`)

Milvus settings follow the same convention as phase 03:

- `MILVUS_URI`
- `MILVUS_TOKEN`
- `MILVUS_DB_NAME`
- `MILVUS_COLLECTION_NAME`
- `MILVUS_COLLECTION_PREFIX`

If `MILVUS_URI` is not set, phase 04 defaults to the local Milvus standalone
endpoint at `http://localhost:19530`.

Elasticsearch settings:

- `ELASTICSEARCH_URL`
- `ELASTICSEARCH_INDEX_NAME`
- `ELASTICSEARCH_USERNAME`
- `ELASTICSEARCH_PASSWORD`
- `ELASTICSEARCH_API_KEY`
- `ELASTICSEARCH_VERIFY_CERTS`
- `ELASTICSEARCH_CA_CERT_PATH`

If `ELASTICSEARCH_URL` is not set, phase 04 defaults to `http://localhost:9200`.

## Local Run

From the repo root:

```powershell
docker compose up -d milvus-standalone elasticsearch kibana

python src/04-online_rag_service/server.py
```

When Kibana is running, open `http://localhost:5601` to inspect the
Elasticsearch lexical index.

Or with Uvicorn:

```powershell
uvicorn server:app --app-dir src/04-online_rag_service --host 0.0.0.0 --port 8000
```

## Docker Run

The repository includes a `phase04` Compose service:

```powershell
docker compose up phase04
```

## Notes

- `POST /reload` refreshes the in-memory Milvus row cache and reports the
  current Elasticsearch lexical document count.
- The service returns both `display_text` and `retrieval_text`
  (`embedding_text`) so the downstream assistant can choose between
  presentation text and retrieval-optimized text.
- Text queries also search the `image_embedding` vector field for chunks with
  `has_image=true`; visual results include `media_assets` so callers can render
  the cropped source image.
- `OPENAI_RERANK_MODEL` must be set explicitly in `.env` or the runtime environment.
- Phase 04 assumes the phase-03 vectors live in Milvus standalone/server
  because phase 03 now builds an `HNSW` index. In Docker Compose, `phase04`
  defaults to `http://milvus-standalone:19530`.
- In Docker Compose, `phase04` defaults to `http://elasticsearch:9200` for
  lexical retrieval.
