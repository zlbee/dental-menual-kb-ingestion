# Phase 03: Vectorization

This stage reads phase-02 semantic chunks, embeds each `indexable` chunk using
the OpenAI-compatible embedding model from `.env`, stores the vectors plus
filterable metadata in Milvus, and publishes lexical retrieval documents to
Elasticsearch.

Phase 03 is the last offline knowledge-base build step. Phase 04 can then focus
only on online retrieval.

## Inputs

The pipeline reads a phase-02 manifest and resolves:

- `semantic_chunks/<doc_id>.jsonl`

It preserves phase-02 metadata such as:

- `chunk_id`
- `doc_id`
- `heading_path` / `heading_path_text`
- `chunk_type`, `semantic_hint`, `content_modality`
- `page_start` / `page_end`
- `source_*` lineage fields
- sibling navigation fields

## Outputs

Running `pipeline.py` writes:

- `data/processed/03_vectorization/manifests/<doc_id>.json`
- `data/processed/03_vectorization/enriched_chunks/<doc_id>.jsonl`

It also upserts the `indexable` semantic chunks into:

- a Milvus collection for dense retrieval and reranking
- an Elasticsearch index for BM25 lexical retrieval

## Milvus Storage Model

- One collection can store many documents.
- `chunk_id` is the primary key.
- `doc_id` is stored as a scalar field for later metadata filtering.
- `embedding_text` from phase 02 is embedded directly without re-chunking.
- Rich lineage arrays such as `heading_path` and `source_block_ids` are stored
  in a JSON metadata field.

If `MILVUS_URI` is not set, phase 03 defaults to a local Milvus standalone
endpoint at `http://localhost:19530`.

If you want to target a standalone or managed Milvus deployment, set
`MILVUS_URI` to the server URL and optionally provide:

- `MILVUS_TOKEN`
- `MILVUS_DB_NAME`
- `MILVUS_COLLECTION_NAME`
- `MILVUS_COLLECTION_PREFIX`

Phase 03 uses an `HNSW` vector index with `COSINE` metric and build params
`M=16`, `efConstruction=200`, so it must run against Milvus standalone/server
rather than Milvus Lite.

## Elasticsearch Storage Model

- `chunk_id` is used as the Elasticsearch document `_id`.
- `embedding_text` is indexed for primary BM25 recall.
- `heading_path_text` is indexed for heading-aware BM25 recall.
- metadata needed for filters and citations such as `doc_id`, `chunk_type`,
  `content_modality`, page fields, and source block ids are stored alongside
  the lexical text.

If `ELASTICSEARCH_URL` is not set, phase 03 defaults to
`http://localhost:9200`.

You can override the lexical target index with:

- `ELASTICSEARCH_INDEX_NAME`

## Docker Workflow

The repository includes a `phase03` Compose service:

```powershell
docker compose up -d milvus-standalone elasticsearch kibana

docker compose run --rm phase03 `
  python src/03-vectorization/pipeline.py `
    --doc-id manualclinprocdentistry-sample
```

When Kibana is running, open `http://localhost:5601` to inspect the
Elasticsearch lexical index.

If the phase-02 outputs live in a different mounted location, point the
pipeline at that root explicitly:

```powershell
docker compose run --rm phase03 `
  python src/03-vectorization/pipeline.py `
    --doc-id manualclinprocdentistry-sample `
    --phase02-root /app/data/processed/02_semantic_chunking
```

## Required Environment Variables

Phase 03 expects these embedding settings to be available through `.env` or the
container environment:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`

Milvus settings are optional. Without `MILVUS_URI`, the pipeline uses
`http://localhost:19530`. Inside Docker Compose, `phase03` defaults to
`http://milvus-standalone:19530`.

Elasticsearch settings are also optional for local runs. Without
`ELASTICSEARCH_URL`, the pipeline uses `http://localhost:9200`. Inside Docker
Compose, `phase03` defaults to `http://elasticsearch:9200`.

Optional Elasticsearch settings:

- `ELASTICSEARCH_INDEX_NAME`
- `ELASTICSEARCH_USERNAME`
- `ELASTICSEARCH_PASSWORD`
- `ELASTICSEARCH_API_KEY`
- `ELASTICSEARCH_VERIFY_CERTS`
- `ELASTICSEARCH_CA_CERT_PATH`

## Notes

- Phase 03 does not reinterpret phase-02 chunk boundaries.
- Phase 03 does not introduce extra filtering beyond the inherited
  `indexable` flag.
- Re-running phase 03 for the same `doc_id` replaces that document's rows in
  both Milvus and Elasticsearch before inserting the new version.
- If the target collection still has the old `FLAT` vector index, phase 03 now
  releases the collection, drops the stale vector index, and rebuilds it as
  `HNSW`.
