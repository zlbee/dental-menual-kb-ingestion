# Phase 04 Elasticsearch + Milvus Hybrid Retrieval Design

## Goal

This document describes the high-level design for **scheme 1**:

- BM25 lexical recall comes from **Elasticsearch**
- dense vector recall and vector fetch come from **Milvus**
- phase 04 keeps responsibility for request orchestration, fusion, reranking,
  filtering, and response shaping
- phase 03 becomes the write point for both Milvus and Elasticsearch

The goal is to replace the current in-memory `rank_bm25` lexical layer with a
real inverted-index search engine while keeping the current Milvus-based dense
retrieval flow.

## Current State

Today the system works like this:

- phase 03 embeds chunks and writes them into Milvus
- phase 04 loads all `indexable` rows from Milvus on startup or `/reload`
- phase 04 builds two in-memory BM25 indexes over:
  - `embedding_text`
  - `heading_path_text`
- query-time fusion is done in phase 04:
  - dense recall from Milvus
  - lexical recall from in-memory BM25
  - weighted RRF fusion
  - exact cosine rerank using vectors fetched from Milvus

Current limitations:

- lexical retrieval is not backed by a true inverted index
- BM25 query cost grows with corpus size
- `/reload` must rebuild lexical state in memory
- lexical retrieval and dense retrieval use different data freshness models

## Target Architecture

The target architecture is:

- **Milvus**
  - stores vectors and scalar metadata needed for dense recall and rerank
- **Elasticsearch**
  - stores lexical fields and filterable metadata needed for BM25 retrieval
- **phase 03**
  - writes the same chunk identity and metadata to both stores
- **phase 04**
  - reads lexical candidates from Elasticsearch
  - reads dense candidates and vectors from Milvus
  - fuses and reranks in application code

This keeps the current hybrid structure but swaps the lexical engine.

## Non-Goals

- do not move dense retrieval from Milvus to Elasticsearch
- do not move reranking into Elasticsearch
- do not redesign the phase 04 API contract
- do not require Elasticsearch-native RRF in the first version

## End-to-End Flow

### Ingestion

1. phase 01 and phase 02 stay unchanged
2. phase 03 prepares chunk rows as it does today
3. phase 03 generates embeddings for `embedding_text`
4. phase 03 writes vector rows to Milvus
5. phase 03 writes lexical documents to Elasticsearch
6. phase 03 records counts and failures for both write paths

### Retrieval

1. phase 04 receives `/retrieve`
2. phase 04 embeds the query for dense recall
3. phase 04 calls Milvus for dense top-k
4. phase 04 calls Elasticsearch for:
   - BM25 over `embedding_text`
   - BM25 over `heading_path_text`
5. phase 04 converts all three lanes into a shared `LaneHit` shape
6. phase 04 applies the existing weighted RRF
7. phase 04 fetches candidate vectors from Milvus
8. phase 04 applies the existing cosine rerank
9. phase 04 returns the same result payload as today

## Canonical Identity and Data Ownership

To keep both backends aligned:

- `chunk_id` is the canonical document key in both systems
- `doc_id` is the document grouping key
- phase 03 is the only component allowed to create or update lexical documents
- phase 04 is read-only with respect to Elasticsearch in the steady state

Recommended ownership model:

- Milvus is the source of truth for vectors
- Elasticsearch is the source of truth for lexical retrieval
- phase 03 is the source of truth for chunk publication

## Elasticsearch Index Design

## Index Naming

Recommended naming pattern:

- `dental_kb_v1_<embedding_model_slug>_lexical`

This mirrors the current Milvus collection naming strategy and makes the
lexical index explicit.

## Stored Document Shape

Each Elasticsearch document should use `chunk_id` as `_id` and contain:

```json
{
  "chunk_id": "doc-001#chunk-0001",
  "doc_id": "doc-001",
  "chunk_order": 1,
  "chunk_type": "paragraph",
  "content_modality": "text",
  "document_title": "Manual of Clinical Procedures in Dentistry",
  "section_title": "Positioning the Patient",
  "heading_path_text": "Restorative Dentistry / Crown Preparation / Positioning the Patient",
  "heading_path": [
    "Restorative Dentistry",
    "Crown Preparation",
    "Positioning the Patient"
  ],
  "page_start": 31,
  "page_end": 32,
  "prev_chunk_id": "doc-001#chunk-0000",
  "next_chunk_id": "doc-001#chunk-0002",
  "display_text": "...",
  "embedding_text": "...",
  "indexable": true,
  "source_block_ids": ["..."],
  "source_marker_block_ids": ["..."]
}
```

## Mapping Recommendation

Suggested first-pass mapping:

```json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "kb_text_analyzer": {
          "type": "english"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "chunk_id": { "type": "keyword" },
      "doc_id": { "type": "keyword" },
      "chunk_order": { "type": "integer" },
      "chunk_type": { "type": "keyword" },
      "content_modality": { "type": "keyword" },
      "document_title": {
        "type": "keyword",
        "fields": {
          "text": { "type": "text", "analyzer": "kb_text_analyzer" }
        }
      },
      "section_title": {
        "type": "keyword",
        "fields": {
          "text": { "type": "text", "analyzer": "kb_text_analyzer" }
        }
      },
      "heading_path_text": {
        "type": "text",
        "analyzer": "kb_text_analyzer"
      },
      "heading_path": { "type": "keyword" },
      "page_start": { "type": "integer" },
      "page_end": { "type": "integer" },
      "prev_chunk_id": { "type": "keyword" },
      "next_chunk_id": { "type": "keyword" },
      "display_text": { "type": "text", "index": false },
      "embedding_text": {
        "type": "text",
        "analyzer": "kb_text_analyzer"
      },
      "indexable": { "type": "boolean" },
      "source_block_ids": { "type": "keyword" },
      "source_marker_block_ids": { "type": "keyword" }
    }
  }
}
```

Notes:

- `embedding_text` and `heading_path_text` are the primary BM25 fields
- equality filters should use `keyword` fields
- `display_text` can be stored but does not need full-text indexing
- the first version should keep the mapping simple and avoid nested objects

## Analyzer Strategy

Recommended first version:

- use `english` analyzer for `embedding_text`
- use `english` analyzer for `heading_path_text`
- keep filter fields as `keyword`

Why:

- the manual content appears to be predominantly English
- stemming should help lexical recall for morphology variants
- this is a practical first step without custom token filters

Known tradeoff:

- Elasticsearch tokenization will not exactly match the current
  `tokenize()` behavior in phase 04
- lexical scores and ranking will change after migration

## Phase 03 Changes

Phase 03 is the main place that needs architectural change.

## Responsibilities

Phase 03 should:

- create or validate the Elasticsearch index
- transform each prepared chunk into an Elasticsearch document
- bulk index lexical documents after vectors are ready
- fail clearly if either backend cannot accept the write

## Write Sequence

Recommended first version:

1. prepare chunk rows
2. generate embeddings
3. ensure Milvus collection
4. ensure Elasticsearch index
5. upsert Milvus rows
6. bulk upsert Elasticsearch docs
7. emit a manifest with both backend write counts

Rationale:

- embeddings are needed only for Milvus
- Elasticsearch documents are derived from the same prepared chunk payload
- write order stays deterministic and easier to reason about

## Failure Policy

Recommended first version:

- treat phase 03 as failed if either Milvus or Elasticsearch write fails
- do not silently continue on partial success
- surface counts of attempted, succeeded, and failed rows

Operational note:

- this can still leave temporary inconsistency if Milvus succeeds and
  Elasticsearch fails after partial bulk indexing
- phase 03 should therefore support idempotent re-run for the same `doc_id`

## Deletion and Reindex

The existing Milvus path already supports replacing rows for a document. The
Elasticsearch side should mirror that behavior:

- delete old lexical documents for the target `doc_id`
- bulk insert the new version

This keeps lexical and vector stores aligned at document granularity.

## Phase 04 Changes

Phase 04 remains the online orchestrator.

## New Responsibilities

- connect to Elasticsearch
- execute two lexical recalls through Elasticsearch
- keep the existing Milvus dense recall
- keep the existing RRF fusion and rerank logic

## Query Plan

For each request:

1. build the existing filter model
2. embed the query for Milvus dense recall
3. search Milvus and collect `dense_hits`
4. search Elasticsearch `embedding_text` and collect `body_hits`
5. search Elasticsearch `heading_path_text` and collect `heading_hits`
6. fuse all lanes with the existing weighted RRF
7. fetch vectors for candidate `chunk_id`s from Milvus
8. rerank with exact cosine similarity
9. serialize the same response payload as today

## Elasticsearch Query Shape

Recommended first version:

- use one ES request for the body lane
- use one ES request for the heading lane
- use `bool.must` with `match`
- use `bool.filter` for exact filters and page ranges
- request only the fields phase 04 actually needs

Example body lane query:

```json
{
  "size": 60,
  "_source": [
    "chunk_id"
  ],
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "embedding_text": {
              "query": "How should the dental team position a patient for crown preparation?"
            }
          }
        }
      ],
      "filter": [
        { "term": { "indexable": true } }
      ]
    }
  }
}
```

The heading lane is identical except the field becomes `heading_path_text`.

## Filtering Strategy

The current phase 04 filters should map naturally to Elasticsearch:

- `doc_ids` -> `terms`
- `chunk_types` -> `terms`
- `content_modalities` -> `terms`
- `document_titles` -> `terms`
- `section_titles` -> `terms`
- `page_from` -> `range` on `page_end >= value`
- `page_to` -> `range` on `page_start <= value`

This is a better fit than the current local Python-side lexical filtering.

## RRF and Rerank

The first version should keep all fusion logic in phase 04:

- no Elasticsearch-native RRF
- no Elasticsearch vector search
- no score normalization changes

Why:

- this minimizes semantic drift
- existing scoring fields can remain mostly unchanged
- Milvus stays the only vector backend

## Reload Semantics

`POST /reload` should change meaning.

Current meaning:

- reload in-memory BM25 state from Milvus

Proposed meaning:

- refresh internal clients and health state only
- do not rebuild lexical data in memory

Optionally:

- `/reload` can also validate that the target Elasticsearch index exists
- `/reload` can report document counts from Milvus and Elasticsearch for sanity checking

## Deployment Changes

## Docker Compose

`compose.yaml` will need:

- a new `elasticsearch` service
- persistent volume for Elasticsearch data
- environment for memory settings and single-node mode
- `phase04` dependency on Elasticsearch health
- `phase03` dependency on Elasticsearch health if dual-write is required during ingestion

## Environment Variables

Suggested new variables:

- `ELASTICSEARCH_URL`
- `ELASTICSEARCH_USERNAME`
- `ELASTICSEARCH_PASSWORD`
- `ELASTICSEARCH_API_KEY`
- `ELASTICSEARCH_INDEX_NAME`
- `ELASTICSEARCH_VERIFY_CERTS`
- `ELASTICSEARCH_CA_CERT_PATH`

The first version should support one auth mode cleanly rather than many.

## Python Dependencies

Expected additions:

- `elasticsearch`

Optional later:

- helper utilities for bulk indexing if not using the low-level client directly

## Consistency Model

Scheme 1 introduces cross-system consistency risk. The design should make this
explicit.

Expected consistency model:

- query-time reads are eventually consistent across Milvus and Elasticsearch
- ingestion-time writes aim for document-level convergence
- a single document re-run should reconcile both backends

What we should not promise:

- distributed transaction semantics between Milvus and Elasticsearch
- strict atomic commit across both systems

## Observability

The first implementation should add clear metrics and logs for:

- Milvus write count
- Elasticsearch bulk write count
- Elasticsearch bulk failures
- dense recall latency
- ES body lane latency
- ES heading lane latency
- candidate pool size after fusion
- rerank latency
- backend count mismatch warnings during reload or health checks

## Rollout Plan

Recommended rollout:

1. add Elasticsearch service and local developer wiring
2. add phase 03 Elasticsearch index creation and bulk indexing
3. verify lexical documents are published correctly
4. add phase 04 Elasticsearch read client
5. switch BM25 lanes from local BM25 to Elasticsearch
6. keep current RRF and rerank unchanged
7. add comparison tooling for old vs new lexical results
8. remove `rank-bm25` after validation

## Complexity Assessment

Rough implementation complexity for this repository:

- deployment and local compose wiring: medium
- phase 03 dual-write: medium to high
- phase 04 lexical query migration: medium
- consistency, retries, and operational hardening: medium to high

Overall complexity:

- **medium to high**

Main reason:

- Elasticsearch itself is straightforward
- the real complexity comes from dual-write, failure handling, and keeping
  Milvus and Elasticsearch aligned by `chunk_id` and `doc_id`

## Risks

- lexical ranking changes because Elasticsearch analysis differs from the
  current simple regex tokenization
- partial write failures can desynchronize Milvus and Elasticsearch
- developer setup becomes heavier because another backend is required
- index mapping mistakes can force reindexing

## Open Questions

- should `document_title` and `section_title` stay filter-only, or should they
  later participate in lexical recall with boosting?
- should Elasticsearch use `english` or `standard` analyzer for the first rollout?
- should phase 03 fail fast on the first Elasticsearch bulk error or collect all
  failures and report them together?
- should `/health` report backend-specific readiness for both Milvus and Elasticsearch?

## Recommendation

Proceed with scheme 1 in two implementation steps:

1. introduce Elasticsearch dual-write in phase 03 and keep phase 04 behavior unchanged
2. switch phase 04 lexical lanes from in-memory BM25 to Elasticsearch once data publication is stable

This keeps the migration incremental and reduces the chance of debugging both
write-path and query-path changes at the same time.
