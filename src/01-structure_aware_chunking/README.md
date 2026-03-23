# Phase 01: Structure-Aware Chunking

This stage uses Marker to recover document structure first, then applies repository-specific normalization rules so later semantic chunking can stay inside structural boundaries.

## Outputs

Running `pipeline.py` writes:

- `data/processed/01_structure_aware/manifests/<doc_id>.json`
- `data/processed/01_structure_aware/marker_raw/<doc_id>/markdown/...`
- `data/processed/01_structure_aware/marker_raw/<doc_id>/json/...` when `--emit-json` is enabled
- `data/processed/01_structure_aware/normalized_blocks/<doc_id>.jsonl`
- `data/processed/01_structure_aware/structural_chunks/<doc_id>.jsonl`

## What It Does

1. Calls `marker_single` once for `markdown` by default.
2. If `--emit-json` is enabled, it also renders `json` and prefers JSON-derived normalization when usable.
3. Rebuilds `heading_path` from `section_hierarchy` when JSON output is available.
4. Filters out page headers/footers and keeps meaningful blocks like text, lists, tables, captions, and table-of-contents blocks.
5. Emits phase-01 structural chunks without worrying about overlong sections yet. Long sections are intentionally deferred to phase 02 semantic chunking.

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

To also persist Marker JSON for the same run:

```powershell
docker compose run --rm phase01 `
  python src/01-structure_aware_chunking/pipeline.py `
    --input-pdf data/raw/ManualClinProcDentistry-Sample.pdf `
    --emit-json
```

To opt into resumable segmented Marker runs, process the PDF in fixed-size page
batches:

```powershell
docker compose run --rm phase01 `
  python src/01-structure_aware_chunking/pipeline.py `
    --input-pdf data/raw/ManualClinProcDentistry-Sample.pdf `
    --segment-pages 10
```

Completed page batches are cached on the host under
`.cache/datalab/phase01/...` through the existing Compose volume mount, so a
later rerun can skip finished batches and continue from the missing ones.

If the PDF exposes a built-in outline/table of contents, you can use those
boundaries as the primary segments instead:

```powershell
docker compose run --rm phase01 `
  python src/01-structure_aware_chunking/pipeline.py `
    --input-pdf data/raw/ManualClinProcDentistry-Sample.pdf `
    --segment-by-outline
```

You can also combine both modes so outline sections stay intact where possible,
while very large sections still get split into smaller page batches:

```powershell
docker compose run --rm phase01 `
  python src/01-structure_aware_chunking/pipeline.py `
    --input-pdf data/raw/ManualClinProcDentistry-Sample.pdf `
    --segment-by-outline `
    --segment-pages 12
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

- The script renders markdown by default. Pass `--emit-json` if you also want Marker JSON artifacts and JSON-based normalization.
- The script now defaults to Marker `--use_llm` with `marker.services.openai.OpenAIService`.
- `.env` values like `OPENAI_API_KEY`, `OPENAI_MODEL`, and optional `OPENAI_BASE_URL` are forwarded to the Marker CLI automatically.
- `compose.yaml` mounts host `src/` and `data/` into `/app`, so code changes do not require rebuilding the image.
- `--segment-pages` is an explicit opt-in recovery mode. It runs Marker in page batches and reassembles the final artifact afterwards. This improves resumability, but the final output can differ slightly from a single full-document Marker run, especially near segment boundaries or for cross-page table/header inference.
- `--segment-by-outline` is another explicit opt-in recovery mode. It first tries to read the PDF's own outline/TOC entries and uses them as segment boundaries. If the outline is missing and `--segment-pages` is also set, the pipeline falls back to fixed page batches.
- Segmented runs currently require `--disable-image-extraction` to stay enabled. That is already the default in this repository.
- If you want to switch to Gemini routing, pass `--llm-service marker.services.gemini.GoogleGeminiService` and provide the matching Gemini credentials in `.env`.
- If Marker JSON is unavailable or unexpectedly shaped, the script keeps the markdown-based normalization so we still get phase-01 outputs.
