# Phase 04: Online RAG Service

This stage serves hybrid retrieval for the AI assistant on top of the
phase-03 Milvus collection.

## Retrieval Design

Phase 04 reads its corpus from Milvus only.

At startup it loads the `indexable` rows from the Milvus collection and builds
two in-memory BM25 indexes:

- `embedding_text` for the primary lexical recall lane
- `metadata.heading_path` for a lighter heading-aware lexical lane

At query time the service runs three first-stage retrieval lanes in parallel:

- dense vector recall against Milvus `embedding` using `OPENAI_RECALL_MODEL`
- BM25 over `embedding_text`
- BM25 over `metadata.heading_path`

The three lanes are fused with weighted reciprocal rank fusion (RRF), then the
candidate set is reranked with exact cosine similarity using
`OPENAI_EMBEDDING_MODEL` against the vectors already stored in Milvus.

## Data Contract

Phase 04 expects the phase-03 collection schema defined in:

- `src/03-vectorization/pipeline.py`

It uses these fields:

- scalar fields: `chunk_id`, `doc_id`, `chunk_type`, `content_modality`,
  `document_title`, `section_title`, `page_start`, `page_end`,
  `prev_chunk_id`, `next_chunk_id`, `display_text`, `embedding_text`
- JSON field: `metadata`
- vector field: `embedding`

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

Milvus settings follow the same convention as phase 03:

- `MILVUS_URI`
- `MILVUS_TOKEN`
- `MILVUS_DB_NAME`
- `MILVUS_COLLECTION_NAME`
- `MILVUS_COLLECTION_PREFIX`

If `MILVUS_URI` is not set, phase 04 defaults to the local Milvus Lite database
under `data/processed/03_vectorization/milvus/knowledge_base.db`.

## Local Run

From the repo root:

```powershell
python src/04-online_rag_service/server.py
```

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

- `POST /reload` refreshes the in-memory BM25 indexes from Milvus so the
  service can pick up newly ingested phase-03 rows without a restart.
- The service returns both `display_text` and `retrieval_text`
  (`embedding_text`) so the downstream assistant can choose between
  presentation text and retrieval-optimized text.
- On Windows hosts, local Milvus Lite database files are still not supported by
  `pymilvus`; use Docker, WSL, or a remote Milvus deployment.
