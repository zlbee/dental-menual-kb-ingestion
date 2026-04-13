# Phase 04 Elasticsearch + Milvus Hybrid Retrieval Task Breakdown

This document breaks the approved scheme into concrete implementation tasks for
this repository.

Reference design:

- `docs/phase04_es_milvus_hybrid_retrieval_design.md`

Scope of this task list:

- BM25 lexical recall from Elasticsearch
- dense recall and vector rerank from Milvus
- phase 03 dual-write into Milvus and Elasticsearch
- phase 04 query-time orchestration across both backends

## Milestone 0: Freeze Decisions

These decisions should be confirmed before code changes begin.

### Task 0.1: Freeze Elasticsearch version and deployment mode

Files likely touched:

- `compose.yaml`
- `docs/phase04_es_milvus_hybrid_retrieval_design.md`

Decision points:

- Elasticsearch version
- single-node local deployment settings
- storage path and memory limits

Acceptance criteria:

- one Elasticsearch image tag is selected
- one local deployment mode is documented

### Task 0.2: Freeze analyzer choice for first rollout

Files likely touched:

- `docs/phase04_es_milvus_hybrid_retrieval_design.md`
- `src/03-vectorization/pipeline.py`

Decision points:

- `english` analyzer vs `standard`
- whether `document_title` and `section_title` need text subfields in v1

Acceptance criteria:

- one analyzer strategy is chosen for `embedding_text`
- one analyzer strategy is chosen for `heading_path_text`

### Task 0.3: Freeze authentication mode for local and non-local runs

Files likely touched:

- `compose.yaml`
- `src/03-vectorization/pipeline.py`
- `src/04-online_rag_service/server.py`

Decision points:

- unauthenticated local node vs username/password vs API key
- TLS verification requirements

Acceptance criteria:

- one primary auth mode is implemented end to end
- environment variables are named and documented

## Milestone 1: Infrastructure and Dependencies

### Task 1.1: Add Elasticsearch service to Docker Compose

Files to update:

- `compose.yaml`

Implementation notes:

- add an `elasticsearch` service
- configure single-node mode
- configure persistent storage volume
- add health check
- wire `phase03` and `phase04` dependencies to Elasticsearch health

Acceptance criteria:

- `docker compose up elasticsearch` starts successfully
- `phase03` and `phase04` can depend on Elasticsearch readiness

### Task 1.2: Add Elasticsearch Python dependency

Files to update:

- `requirements/phase03.txt`
- `requirements/phase04.txt`

Implementation notes:

- add the official `elasticsearch` Python client
- keep versioning consistent between phase 03 and phase 04

Acceptance criteria:

- both phase environments can import the Elasticsearch client

### Task 1.3: Define Elasticsearch environment variables

Files to update:

- `src/03-vectorization/README.md`
- `src/04-online_rag_service/README.md`
- optionally `.env` documentation if the repo later adds an example file

Suggested variables:

- `ELASTICSEARCH_URL`
- `ELASTICSEARCH_INDEX_NAME`
- `ELASTICSEARCH_USERNAME`
- `ELASTICSEARCH_PASSWORD`
- `ELASTICSEARCH_API_KEY`
- `ELASTICSEARCH_VERIFY_CERTS`
- `ELASTICSEARCH_CA_CERT_PATH`

Acceptance criteria:

- required vs optional Elasticsearch env vars are documented
- phase 03 and phase 04 document the same naming convention

## Milestone 2: Shared Elasticsearch Model

### Task 2.1: Define lexical index naming strategy

Files to update:

- `src/03-vectorization/pipeline.py`
- `src/04-online_rag_service/server.py`

Implementation notes:

- mirror the current Milvus naming approach
- support explicit override via env var or CLI option
- support deterministic fallback naming

Acceptance criteria:

- both phases resolve the exact same Elasticsearch index name

### Task 2.2: Define Elasticsearch mapping and settings

Files to update:

- `src/03-vectorization/pipeline.py`
- optionally a new helper module if the mapping is too large

Recommended first-pass fields:

- `chunk_id`
- `doc_id`
- `chunk_order`
- `chunk_type`
- `content_modality`
- `document_title`
- `section_title`
- `heading_path_text`
- `heading_path`
- `page_start`
- `page_end`
- `prev_chunk_id`
- `next_chunk_id`
- `display_text`
- `embedding_text`
- `indexable`
- `source_block_ids`
- `source_marker_block_ids`

Acceptance criteria:

- mapping can be created from code
- lexical and filter fields are explicitly typed
- `chunk_id` is used as the Elasticsearch document `_id`

### Task 2.3: Add Elasticsearch document serializer

Files to update:

- `src/03-vectorization/pipeline.py`

Implementation notes:

- convert prepared chunk rows into one Elasticsearch document per `chunk_id`
- keep field names aligned with phase 04 response needs
- flatten only the metadata needed for lexical retrieval and filtering

Acceptance criteria:

- one helper returns a valid Elasticsearch document for every indexable chunk
- serialized fields are stable and deterministic

## Milestone 3: Phase 03 Dual-Write

### Task 3.1: Add Elasticsearch config to phase 03 CLI/runtime

Files to update:

- `src/03-vectorization/pipeline.py`

Implementation notes:

- add CLI args if needed for explicit index override or URL override
- load Elasticsearch env vars alongside existing Milvus env vars

Acceptance criteria:

- phase 03 can resolve Elasticsearch connection config without affecting the current Milvus path

### Task 3.2: Add Elasticsearch client construction

Files to update:

- `src/03-vectorization/pipeline.py`

Implementation notes:

- build the client from env values
- support the chosen auth mode
- centralize connection validation errors

Acceptance criteria:

- phase 03 can open an Elasticsearch client and fail with clear configuration errors

### Task 3.3: Ensure Elasticsearch index exists before write

Files to update:

- `src/03-vectorization/pipeline.py`

Implementation notes:

- create the index if missing
- validate existence before bulk indexing
- avoid silent mapping drift

Acceptance criteria:

- first run can create the index automatically
- repeated runs do not break if the index already exists

### Task 3.4: Add document-level delete path for Elasticsearch

Files to update:

- `src/03-vectorization/pipeline.py`

Implementation notes:

- mirror current Milvus replacement behavior
- delete prior lexical docs for the same `doc_id` before re-insert
- use `delete_by_query` or a controlled equivalent

Acceptance criteria:

- rerunning phase 03 for the same `doc_id` replaces old lexical rows cleanly

### Task 3.5: Add Elasticsearch bulk indexing

Files to update:

- `src/03-vectorization/pipeline.py`

Implementation notes:

- bulk index only `indexable` chunks
- make `_id == chunk_id`
- collect success and failure counts

Acceptance criteria:

- all indexable chunks are written to Elasticsearch
- bulk failures are surfaced clearly

### Task 3.6: Define partial failure policy in phase 03

Files to update:

- `src/03-vectorization/pipeline.py`
- `src/03-vectorization/README.md`

Implementation notes:

- fail the pipeline if either Milvus or Elasticsearch write fails
- keep the rerun path idempotent
- log enough detail to reconcile partial writes

Acceptance criteria:

- phase 03 exits non-zero on Elasticsearch write failure
- operator can rerun the same `doc_id` after a failed attempt

### Task 3.7: Extend phase 03 manifest with Elasticsearch write metadata

Files to update:

- `src/03-vectorization/pipeline.py`

Suggested manifest additions:

- resolved Elasticsearch URL or sanitized host
- Elasticsearch index name
- lexical write counts
- lexical delete counts
- bulk failure count

Acceptance criteria:

- manifest records both Milvus and Elasticsearch publication metadata

## Milestone 4: Phase 04 Elasticsearch Read Path

### Task 4.1: Add Elasticsearch config to phase 04 runtime

Files to update:

- `src/04-online_rag_service/server.py`

Implementation notes:

- extend `RuntimeConfig`
- load Elasticsearch env vars
- expose resolved lexical backend config in `config_snapshot()`

Acceptance criteria:

- phase 04 can resolve Elasticsearch runtime config at startup

### Task 4.2: Add Elasticsearch client construction to phase 04

Files to update:

- `src/04-online_rag_service/server.py`

Implementation notes:

- create and hold a reusable Elasticsearch client in `OnlineRAGService`
- close it during shutdown if needed

Acceptance criteria:

- phase 04 can connect to Elasticsearch and fail clearly if unavailable

### Task 4.3: Replace in-memory BM25 build during reload

Files to update:

- `src/04-online_rag_service/server.py`
- `src/04-online_rag_service/README.md`

Implementation notes:

- remove or bypass `rank_bm25` index construction
- keep Milvus-backed row loading only if still needed for:
  - neighbor lookup
  - response shaping
  - rerank candidate metadata
- redefine `/reload` so it refreshes in-memory row metadata only

Acceptance criteria:

- phase 04 no longer depends on local BM25 index state for lexical search
- `/reload` remains meaningful and safe

### Task 4.4: Add Elasticsearch filter query builder

Files to update:

- `src/04-online_rag_service/server.py`

Implementation notes:

- translate existing `RetrievalFilters` into Elasticsearch `bool.filter`
- support:
  - `doc_ids`
  - `chunk_types`
  - `content_modalities`
  - `document_titles`
  - `section_titles`
  - `page_from`
  - `page_to`
- keep semantics aligned with current Milvus/local filter behavior

Acceptance criteria:

- Elasticsearch lexical queries respect the same filter contract as today

### Task 4.5: Add Elasticsearch body-lane query

Files to update:

- `src/04-online_rag_service/server.py`

Implementation notes:

- issue a `match` query against `embedding_text`
- request only fields required to produce `LaneHit`
- convert results into the existing `LaneHit` structure

Acceptance criteria:

- body lexical lane returns ranked `chunk_id` hits from Elasticsearch

### Task 4.6: Add Elasticsearch heading-lane query

Files to update:

- `src/04-online_rag_service/server.py`

Implementation notes:

- issue a `match` query against `heading_path_text`
- convert results into the existing `LaneHit` structure

Acceptance criteria:

- heading lexical lane returns ranked `chunk_id` hits from Elasticsearch

### Task 4.7: Keep dense recall and rerank unchanged

Files to review:

- `src/04-online_rag_service/server.py`

Implementation notes:

- preserve Milvus dense search
- preserve vector fetch by `chunk_id`
- preserve cosine rerank formula

Acceptance criteria:

- dense recall still comes only from Milvus
- rerank still uses candidate vectors fetched from Milvus

### Task 4.8: Remove `rank-bm25` dependency from phase 04

Files to update:

- `requirements/phase04.txt`
- `src/04-online_rag_service/server.py`
- `src/04-online_rag_service/README.md`

Acceptance criteria:

- `rank-bm25` is no longer imported or required

## Milestone 5: API, Health, and Observability

### Task 5.1: Extend `/health` with Elasticsearch readiness

Files to update:

- `src/04-online_rag_service/server.py`

Implementation notes:

- report Milvus and Elasticsearch connectivity separately
- report resolved collection/index names

Acceptance criteria:

- `/health` can reveal which backend is unavailable

### Task 5.2: Extend config snapshot and debug information

Files to update:

- `src/04-online_rag_service/server.py`

Implementation notes:

- add Elasticsearch index name and connectivity-related config
- keep sensitive values sanitized

Acceptance criteria:

- debug responses expose enough config for troubleshooting without leaking credentials

### Task 5.3: Add retrieval latency logging by lane

Files to update:

- `src/04-online_rag_service/server.py`

Suggested metrics:

- Milvus dense latency
- Elasticsearch body-lane latency
- Elasticsearch heading-lane latency
- candidate pool size
- rerank latency

Acceptance criteria:

- lane-level latency is visible in logs or debug output

## Milestone 6: Documentation Updates

### Task 6.1: Update phase 03 README

Files to update:

- `src/03-vectorization/README.md`

Topics to add:

- Elasticsearch requirements
- dual-write behavior
- re-run and replacement semantics

Acceptance criteria:

- phase 03 README reflects Milvus + Elasticsearch publication

### Task 6.2: Update phase 04 README

Files to update:

- `src/04-online_rag_service/README.md`

Topics to change:

- lexical retrieval now comes from Elasticsearch
- `/reload` no longer rebuilds in-memory BM25
- required environment variables include Elasticsearch config

Acceptance criteria:

- phase 04 README matches the new runtime behavior

### Task 6.3: Keep architecture docs aligned

Files to update:

- `docs/phase04_es_milvus_hybrid_retrieval_design.md`
- this task file

Acceptance criteria:

- docs remain consistent with implementation decisions made during coding

## Milestone 7: Verification and Rollout

### Task 7.1: Add local smoke test procedure

Files to update:

- `docs/phase04_es_milvus_hybrid_retrieval_tasks.md`
- optionally one README if a command sequence should live there

Suggested smoke test flow:

1. start Milvus and Elasticsearch
2. run phase 03 for one document
3. verify docs appear in Milvus and Elasticsearch
4. start phase 04
5. issue one query that should hit lexical terms clearly
6. verify response includes dense and lexical score fields

Acceptance criteria:

- a developer can follow one documented flow to validate the feature locally

### Task 7.2: Add cross-backend consistency checks

Files to update:

- `src/04-online_rag_service/server.py`
- or a small verification helper if preferred later

Implementation notes:

- compare basic counts between Milvus and Elasticsearch
- expose count mismatch warnings in `/reload` or `/health`

Acceptance criteria:

- obvious publication drift can be detected without manual debugging

### Task 7.3: Run lexical parity comparison before removing old path

Files to update:

- optional temporary tooling or debug code

Implementation notes:

- compare a sample set of queries between:
  - old in-memory BM25
  - new Elasticsearch lexical recall
- record recall differences and obvious regressions

Acceptance criteria:

- migration is evaluated with sample queries before old lexical code is deleted

### Task 7.4: Remove temporary migration scaffolding

Files to update:

- any temporary comparison code added during rollout

Acceptance criteria:

- only the final Elasticsearch lexical path remains

## Recommended Implementation Order

Recommended sequence for actual work:

1. Milestone 1
2. Milestone 2
3. Milestone 3
4. verify phase 03 publication works
5. Milestone 4
6. Milestone 5
7. Milestone 6
8. Milestone 7

This order reduces the chance of debugging write-path and read-path problems at
the same time.

## Suggested PR Breakdown

If this is split into multiple PRs, a practical breakdown is:

### PR 1: Infrastructure and dependency wiring

- add Elasticsearch to compose
- add Python dependencies
- add env handling skeleton

### PR 2: Phase 03 dual-write

- add index creation
- add delete and bulk index
- extend manifest output

### PR 3: Phase 04 Elasticsearch lexical recall

- add Elasticsearch client
- replace local BM25 lanes
- keep RRF and rerank unchanged

### PR 4: Observability, docs, and cleanup

- add health/reporting updates
- update READMEs
- remove `rank-bm25`

## Definition of Done

The migration is complete when all of the following are true:

- phase 03 writes indexable lexical docs to Elasticsearch
- phase 04 lexical recall no longer depends on in-memory BM25
- dense recall still comes from Milvus
- rerank still uses vectors from Milvus
- `/health` reports both backend states
- local compose can bring up the full stack
- documentation matches the shipped behavior
