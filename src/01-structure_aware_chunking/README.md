# Phase 01: Structure-Aware Chunking

This stage uses Marker to recover document structure first, then applies repository-specific normalization rules so later semantic chunking can stay inside structural boundaries.

## Outputs

Running `pipeline.py` writes:

- `data/processed/01_structure_aware/manifests/<doc_id>.json`
- `data/processed/01_structure_aware/marker_raw/<doc_id>/markdown/...`
- `data/processed/01_structure_aware/marker_raw/<doc_id>/json/...`
- `data/processed/01_structure_aware/normalized_blocks/<doc_id>.jsonl`
- `data/processed/01_structure_aware/structural_chunks/<doc_id>.jsonl`

## What It Does

1. Calls `marker_single` twice for the same PDF:
   - once for `markdown`
   - once for `json`
2. Uses Marker JSON as the primary structured source.
3. Rebuilds `heading_path` from `section_hierarchy`.
4. Filters out page headers/footers and keeps meaningful blocks like text, lists, tables, captions, and table-of-contents blocks.
5. Emits phase-01 structural chunks without worrying about overlong sections yet. Long sections are intentionally deferred to phase 02 semantic chunking.
6. Defaults to Marker's Gemini service, while still allowing an explicit switch back to OpenAI if needed.

## Docker Workflow

The repository is now set up to use Docker for environment management.

There is also a `compose.yaml` with a `phase01` service so we can standardize the runtime entrypoint later.

### 1. Build the image

The `Dockerfile` is prepared for Marker, but dependency installation is still intentionally commented out until we are ready to install:

```dockerfile
RUN pip install -r /app/requirements.txt
```

### 2. Typical sample run

The image installs dependencies once, then we bind-mount the host `src/` into the
container so Python code changes are picked up immediately without rebuilding.

Once dependency installation is enabled in the image:

```powershell
docker build -t dental-kb-ingestion .
docker run --rm `
  --env-file .env `
  -v "${PWD}/src:/app/src" `
  -v "${PWD}/data:/app/data" `
  dental-kb-ingestion `
  python src/01-structure_aware_chunking/pipeline.py `
    --input-pdf data/raw/ManualClinProcDentistry-Sample.pdf
```

Or with Compose:

```powershell
docker compose run --rm phase01 `
  python src/01-structure_aware_chunking/pipeline.py `
    --input-pdf data/raw/ManualClinProcDentistry-Sample.pdf
```

### 3. Running a PDF outside the workspace

If the source PDF lives outside this repository, mount that parent folder explicitly and pass the in-container path:

```powershell
docker run --rm `
  --env-file .env `
  -v "${PWD}/src:/app/src" `
  -v "${PWD}/data:/app/data" `
  -v "C:/path/to/external-pdfs:/external-data:ro" `
  dental-kb-ingestion `
  python src/01-structure_aware_chunking/pipeline.py `
    --input-pdf /external-data/ManualClinProcDentistry-page-1.pdf
```

## Notes

- The script now defaults to Marker `--use_llm` with `marker.services.gemini.GoogleGeminiService`.
- `.env` values like `GEMINI_API_KEY` and optional `GEMINI_MODEL_NAME` are forwarded to the Marker CLI automatically.
- `compose.yaml` mounts host `src/` and `data/` into `/app`, so code changes do not require rebuilding the image.
- If you want to force OpenAI-compatible routing again, pass `--llm-service marker.services.openai.OpenAIService`.
- If Marker JSON is unavailable or unexpectedly shaped, the script falls back to a markdown-based parser so we still get phase-01 outputs.
