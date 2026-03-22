# Phase 02: Semantic Chunking

This stage performs semantic chunking inside the structural boundaries created by
phase 01. It uses LangChain `SemanticChunker` with OpenRouter-hosted OpenAI
embeddings and does not attempt to repair or reinterpret hierarchy metadata.

## Inputs

The pipeline reads a phase-01 manifest and resolves:

- `normalized_blocks/<doc_id>.jsonl`
- `structural_chunks/<doc_id>.jsonl`

Phase 02 treats `heading_path`, `indexable`, and the broader content-role
decisions from phase 01 as read-only inputs.

## Outputs

Running `pipeline.py` writes:

- `data/processed/02_semantic_chunking/manifests/<doc_id>.json`
- `data/processed/02_semantic_chunking/semantic_chunks/<doc_id>.jsonl`

Each semantic chunk preserves:

- `heading_path` inherited from phase 01
- `source_structural_chunk_ids`
- `source_block_ids`
- `page_start` / `page_end`
- `embedding_text` with structural context for downstream vectorization

## What It Does

1. Reads phase-01 structural chunks as the outer boundaries.
2. Keeps non-indexable phase-01 chunks as-is instead of reclassifying them.
3. Uses `SemanticChunker` only for narrative text-like chunks.
4. Groups adjacent list items into slightly larger retrieval units.
5. Merges figures/pictures with following captions when they are adjacent.
6. Applies a hard-cap fallback for oversized chunks so phase-03 vectorization
   gets bounded inputs.

## Docker Workflow

The repository now includes a `phase02` Compose service:

```powershell
docker compose run --rm phase02 `
  python src/02-semantic_chunking/pipeline.py `
    --doc-id manualclinprocdentistry-sample
```

If the phase-01 outputs live in a different mounted location, point the pipeline
at that root explicitly:

```powershell
docker compose run --rm phase02 `
  python src/02-semantic_chunking/pipeline.py `
    --doc-id manualclinprocdentistry-sample `
    --phase01-root /app/data/processed/01_structure_aware
```

## Required Environment Variables

Phase 02 expects these settings to be available through `.env` or container
environment variables:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`

Example embedding configuration:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url=env_values["OPENAI_BASE_URL"],
    api_key=env_values["OPENAI_API_KEY"],
    model=env_values["OPENAI_EMBEDDING_MODEL"],
)
```

## Notes

- Phase 02 does not fix `heading_path` issues. If hierarchy needs to improve,
  update phase 01 and rerun phase 02.
- The default chunk profile is intentionally slightly larger and more complete:
  target `450` tokens with an `800` token hard cap.
- The pipeline stores non-indexable chunks too, but keeps their inherited
  `indexable` flag so later stages can skip embedding them.
