from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import html
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "01_structure_aware"
DEFAULT_MARKER_EXECUTABLE = "marker_single"
DEFAULT_LLM_SERVICE = "marker.services.openai.OpenAIService"
SEGMENTED_ARTIFACT_DIRNAME = "__segmented__"

TEXTUAL_BLOCK_TYPES = {
    "Caption",
    "Code",
    "Equation",
    "Figure",
    "Footnote",
    "Form",
    "Handwriting",
    "ListGroup",
    "ListItem",
    "Picture",
    "SectionHeader",
    "Table",
    "TableOfContents",
    "Text",
    "TextInlineMath",
}
GROUP_ONLY_BLOCK_TYPES = {
    "Document",
    "FigureGroup",
    "Page",
    "PictureGroup",
    "TableGroup",
}
IGNORED_BLOCK_TYPES = {
    "Line",
    "PageFooter",
    "PageHeader",
    "Span",
}
FRONT_MATTER_PATTERNS = (
    re.compile(r"^(contents|table of contents)$", re.IGNORECASE),
    re.compile(r"^preface$", re.IGNORECASE),
    re.compile(r"^(list of )?contributors?$", re.IGNORECASE),
    re.compile(r"^copyright$", re.IGNORECASE),
    re.compile(r"^about the editors?$", re.IGNORECASE),
)
PROCEDURE_PATTERNS = (
    re.compile(r"\bprocedure\b", re.IGNORECASE),
    re.compile(r"\btreatment\b", re.IGNORECASE),
    re.compile(r"\blearning points?\b", re.IGNORECASE),
)
PAGE_MARKER_PATTERN = re.compile(r"^\d+$|^[ivxlcdm]+$", re.IGNORECASE)
WORD_HYPHENATION_PATTERN = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])")
PAGE_REF_PATTERN = re.compile(r"/page/(\d+)/")


@dataclasses.dataclass(slots=True)
class PipelineConfig:
    input_pdf: Path
    output_root: Path
    env_file: Path | None
    doc_id: str
    marker_executable: str
    llm_service: str
    use_llm: bool
    torch_device: str | None
    emit_json: bool
    disable_image_extraction: bool
    export_image_assets: bool
    paginate_output: bool
    page_range: str | None
    segment_by_outline: bool
    segment_pages: int | None
    force: bool
    debug: bool


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Run structure-aware chunking with Marker and normalize the output for later semantic chunking."
    )
    parser.add_argument(
        "--input-pdf",
        required=True,
        type=Path,
        help="PDF to process. This can be inside or outside the repository.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for phase-01 outputs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Optional .env file used to hydrate Marker/OpenAI-compatible settings.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Optional stable document id. Defaults to a slugified input filename.",
    )
    parser.add_argument(
        "--marker-executable",
        type=str,
        default=DEFAULT_MARKER_EXECUTABLE,
        help="Marker CLI executable name or path.",
    )
    parser.add_argument(
        "--llm-service",
        type=str,
        default=DEFAULT_LLM_SERVICE,
        help="Marker LLM service class used when --use-llm is enabled.",
    )
    parser.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Marker hybrid mode with --use_llm. Defaults to on.",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default=None,
        help=(
            "Optional torch device forwarded to Marker via TORCH_DEVICE, for example "
            "'cuda', 'cuda:0', or 'cpu'. When omitted, Marker keeps its default "
            "device auto-detection."
        ),
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Also render Marker JSON in addition to markdown. Defaults to markdown only.",
    )
    parser.add_argument(
        "--export-image-assets",
        action="store_true",
        help=(
            "Crop Figure/Picture blocks from the source PDF into stable image assets "
            "and attach their paths to normalized visual blocks. This also enables "
            "Marker JSON output."
        ),
    )
    parser.add_argument(
        "--disable-image-extraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid exporting extracted images unless explicitly needed.",
    )
    parser.add_argument(
        "--paginate-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask Marker to paginate markdown output.",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help="Optional Marker page range, for example 0,5-10,20.",
    )
    parser.add_argument(
        "--segment-by-outline",
        action="store_true",
        help=(
            "Optional opt-in segmented Marker mode that first reads the PDF outline/"
            "table-of-contents entries and uses them as segment boundaries. Combine "
            "with --segment-pages to further split very large outline sections."
        ),
    )
    parser.add_argument(
        "--segment-pages",
        type=int,
        default=None,
        help=(
            "Optional opt-in segmented Marker mode. Process the selected pages in "
            "fixed-size page batches, or further split outline-based batches when "
            "--segment-by-outline is enabled. This may slightly change results near "
            "segment boundaries."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun Marker even if raw artifacts already exist.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Marker debug mode and keep more diagnostics.",
    )
    parsed = parser.parse_args()
    if parsed.segment_pages is not None and parsed.segment_pages < 1:
        parser.error("--segment-pages must be >= 1.")

    input_pdf = parsed.input_pdf.expanduser().resolve()
    output_root = parsed.output_root.expanduser().resolve()
    env_file = parsed.env_file.expanduser().resolve() if parsed.env_file else None
    doc_id = parsed.doc_id or slugify(parsed.input_pdf.stem)

    return PipelineConfig(
        input_pdf=input_pdf,
        output_root=output_root,
        env_file=env_file,
        doc_id=doc_id,
        marker_executable=parsed.marker_executable,
        llm_service=parsed.llm_service,
        use_llm=parsed.use_llm,
        torch_device=(parsed.torch_device.strip() or None) if parsed.torch_device else None,
        emit_json=bool(parsed.emit_json or parsed.export_image_assets),
        disable_image_extraction=parsed.disable_image_extraction,
        export_image_assets=parsed.export_image_assets,
        paginate_output=parsed.paginate_output,
        page_range=parsed.page_range,
        segment_by_outline=parsed.segment_by_outline,
        segment_pages=parsed.segment_pages,
        force=parsed.force,
        debug=parsed.debug,
    )


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("-._")
    return normalized.lower() or "document"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_simple_env(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded[key] = value
    return loaded


def merged_env(extra_env: dict[str, str]) -> dict[str, str]:
    merged = dict(os.environ)
    merged.update(extra_env)
    return merged


def resolve_torch_device(cfg: PipelineConfig, env_values: dict[str, str]) -> str | None:
    if cfg.torch_device:
        return cfg.torch_device

    value = env_values.get("TORCH_DEVICE")
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def build_marker_runtime_env(cfg: PipelineConfig, env_values: dict[str, str]) -> dict[str, str]:
    runtime_env = dict(env_values)
    torch_device = resolve_torch_device(cfg, env_values)
    if torch_device:
        runtime_env["TORCH_DEVICE"] = torch_device
    return runtime_env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_marker_command(
    cfg: PipelineConfig,
    output_dir: Path,
    output_format: str,
    env_values: dict[str, str],
) -> list[str]:
    command = [
        cfg.marker_executable,
        str(cfg.input_pdf),
        "--output_dir",
        str(output_dir),
        "--output_format",
        output_format,
    ]

    if cfg.use_llm:
        command.append("--use_llm")
        command.extend(["--llm_service", cfg.llm_service])
        command.extend(build_llm_service_args(cfg.llm_service, env_values))

    if cfg.disable_image_extraction:
        command.append("--disable_image_extraction")
    if cfg.paginate_output:
        command.append("--paginate_output")
    if cfg.page_range:
        command.extend(["--page_range", cfg.page_range])
    if cfg.debug:
        command.append("--debug")

    return command


def redact_command(command: list[str]) -> str:
    redacted: list[str] = []
    secret_flags = {"--openai_api_key", "--gemini_api_key"}
    skip_next = False
    for index, token in enumerate(command):
        if skip_next:
            skip_next = False
            continue
        if token in secret_flags and index + 1 < len(command):
            redacted.append(token)
            redacted.append("***")
            skip_next = True
            continue
        redacted.append(token)
    return " ".join(shlex.quote(part) for part in redacted)


def build_llm_service_args(llm_service: str, env_values: dict[str, str]) -> list[str]:
    service_name = llm_service.lower()
    args: list[str] = []

    if "gemini" in service_name:
        gemini_api_key = env_values.get("GEMINI_API_KEY") or env_values.get("GOOGLE_API_KEY")
        gemini_model_name = env_values.get("GEMINI_MODEL_NAME") or env_values.get("GEMINI_MODEL")

        if gemini_api_key:
            args.extend(["--gemini_api_key", gemini_api_key])
        if gemini_model_name:
            args.extend(["--gemini_model_name", gemini_model_name])
        return args

    if "openai" in service_name:
        if env_values.get("OPENAI_API_KEY"):
            args.extend(["--openai_api_key", env_values["OPENAI_API_KEY"]])
        if env_values.get("OPENAI_MODEL"):
            args.extend(["--openai_model", env_values["OPENAI_MODEL"]])
        if env_values.get("OPENAI_BASE_URL"):
            args.extend(["--openai_base_url", env_values["OPENAI_BASE_URL"]])
        return args

    return args


def find_primary_artifact(
    output_dir: Path,
    output_format: str,
    preferred_stems: Iterable[str] | None = None,
) -> Path:
    suffixes = {
        "markdown": {".md", ".markdown"},
        "json": {".json"},
    }[output_format]

    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and SEGMENTED_ARTIFACT_DIRNAME not in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a {output_format} artifact under {output_dir}."
        )

    normalized_preferred = {
        token.lower() for token in (preferred_stems or []) if token and token.strip()
    }

    def sort_key(item: Path) -> tuple[int, int, int, str]:
        stem = item.stem.lower()
        preferred_rank = 0 if any(token in stem for token in normalized_preferred) else 1
        return (preferred_rank, len(item.parts), -item.stat().st_size, item.name)

    candidates.sort(key=sort_key)
    return candidates[0]


def default_segment_cache_root() -> Path:
    override = os.environ.get("PHASE01_SEGMENT_CACHE_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    datalab_cache = Path("/root/.cache/datalab")
    if datalab_cache.exists() or Path("/.dockerenv").exists():
        return datalab_cache / "phase01"
    return REPO_ROOT / ".cache" / "phase01"


def load_pdf_reader_cls():
    reader_cls = None
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name, fromlist=["PdfReader"])
            reader_cls = getattr(module, "PdfReader")
            break
        except (ImportError, AttributeError):
            continue

    if reader_cls is None:
        raise RuntimeError(
            "Segmented phase01 runs require pypdf or PyPDF2 so the pipeline can "
            "count PDF pages before building page batches."
        )

    return reader_cls


def get_pdf_page_count(pdf_path: Path) -> int:
    reader_cls = load_pdf_reader_cls()
    return len(reader_cls(str(pdf_path)).pages)


def parse_page_range_spec(spec: str, total_pages: int) -> list[int]:
    pages: list[int] = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start = int(start_text.strip())
                end = int(end_text.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid page range token: {token!r}") from exc
            if start < 0 or end < 0 or end < start:
                raise ValueError(f"Invalid page range token: {token!r}")
            pages.extend(range(start, end + 1))
            continue

        try:
            page = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid page token: {token!r}") from exc
        if page < 0:
            raise ValueError(f"Invalid page token: {token!r}")
        pages.append(page)

    unique_pages = sorted(set(pages))
    if not unique_pages:
        raise ValueError("The requested page range resolved to no pages.")
    if unique_pages[-1] >= total_pages:
        raise ValueError(
            f"Requested page {unique_pages[-1]} is outside the PDF page count ({total_pages})."
        )
    return unique_pages


def format_page_range_spec(pages: Iterable[int]) -> str:
    normalized = sorted(set(int(page) for page in pages))
    if not normalized:
        raise ValueError("Cannot format an empty page range.")

    tokens: list[str] = []
    start = normalized[0]
    end = normalized[0]
    for page in normalized[1:]:
        if page == end + 1:
            end = page
            continue
        tokens.append(f"{start}-{end}" if start != end else str(start))
        start = end = page
    tokens.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(tokens)


def chunk_pages(pages: list[int], chunk_size: int) -> list[list[int]]:
    return [pages[index : index + chunk_size] for index in range(0, len(pages), chunk_size)]


def resolve_requested_pages(cfg: PipelineConfig) -> tuple[list[int], int]:
    total_pages = get_pdf_page_count(cfg.input_pdf)
    if cfg.page_range:
        return parse_page_range_spec(cfg.page_range, total_pages), total_pages
    return list(range(total_pages)), total_pages


def normalize_outline_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_outline_entries_with_pymupdf(pdf_path: Path) -> list[dict[str, Any]]:
    module = None
    for module_name in ("pymupdf", "fitz"):
        try:
            module = __import__(module_name)
            break
        except ImportError:
            continue

    if module is None:
        raise ImportError("PyMuPDF is not available.")

    entries: list[dict[str, Any]] = []
    with module.open(str(pdf_path)) as document:
        try:
            toc_entries = document.get_toc(simple=True)
        except TypeError:
            toc_entries = document.get_toc()

    for item in toc_entries or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            level = int(item[0])
            page_start = int(item[2]) - 1
        except (TypeError, ValueError):
            continue
        title = normalize_outline_title(str(item[1] or ""))
        if not title:
            continue
        entries.append(
            {
                "level": level,
                "title": title,
                "page_start": page_start,
            }
        )

    return entries


def load_outline_entries_with_pypdf(pdf_path: Path) -> list[dict[str, Any]]:
    reader_cls = load_pdf_reader_cls()
    reader = reader_cls(str(pdf_path))
    raw_outline = getattr(reader, "outline", None)
    if raw_outline is None:
        raw_outline = getattr(reader, "outlines", None)
    if raw_outline is None:
        return []

    entries: list[dict[str, Any]] = []

    def walk(items: Any, level: int) -> None:
        if isinstance(items, list):
            for item in items:
                walk(item, level)
            return

        if isinstance(items, tuple):
            for item in items:
                walk(item, level)
            return

        if isinstance(items, dict):
            children = items.get("/Kids") or items.get("children")
            title = normalize_outline_title(str(items.get("/Title") or items.get("title") or ""))
            page_value = items.get("/Page")
            page_start = None
            if page_value is not None:
                try:
                    page_start = int(page_value)
                except (TypeError, ValueError):
                    page_start = None
            if title and page_start is not None:
                entries.append(
                    {
                        "level": level,
                        "title": title,
                        "page_start": page_start,
                    }
                )
            if children:
                walk(children, level + 1)
            return

        title = normalize_outline_title(str(getattr(items, "title", "") or ""))
        if title:
            try:
                page_start = int(reader.get_destination_page_number(items))
            except Exception:
                page_start = None
            if page_start is not None and page_start >= 0:
                entries.append(
                    {
                        "level": level,
                        "title": title,
                        "page_start": page_start,
                    }
                )

        nested = getattr(items, "children", None)
        if nested:
            walk(nested, level + 1)

    walk(raw_outline, 1)
    return entries


def load_pdf_outline_entries(pdf_path: Path, total_pages: int) -> list[dict[str, Any]]:
    loaders = (
        load_outline_entries_with_pymupdf,
        load_outline_entries_with_pypdf,
    )
    entries: list[dict[str, Any]] = []
    for loader in loaders:
        try:
            entries = loader(pdf_path)
            if entries:
                break
        except ImportError:
            continue
        except Exception:
            continue

    filtered = [
        entry
        for entry in entries
        if 0 <= int(entry["page_start"]) < total_pages and str(entry["title"]).strip()
    ]
    filtered.sort(key=lambda entry: (int(entry["page_start"]), int(entry["level"])))
    return filtered


def build_fixed_page_batches(requested_pages: list[int], segment_pages: int) -> list[dict[str, Any]]:
    return [
        {
            "pages": page_group,
            "segment_source": "pages",
        }
        for page_group in chunk_pages(requested_pages, segment_pages)
    ]


def build_outline_batches(
    outline_entries: list[dict[str, Any]],
    requested_pages: list[int],
    segment_pages: int | None,
) -> list[dict[str, Any]]:
    if not requested_pages:
        return []

    requested_pages = sorted(set(requested_pages))
    requested_min = requested_pages[0]
    requested_max = requested_pages[-1]

    entries_by_start: dict[int, list[dict[str, Any]]] = {}
    for entry in outline_entries:
        page_start = int(entry["page_start"])
        if page_start < requested_min or page_start > requested_max:
            continue
        entries_by_start.setdefault(page_start, []).append(entry)

    boundary_pages = sorted(entries_by_start)
    if requested_min not in entries_by_start:
        entries_by_start[requested_min] = [
            {
                "level": 0,
                "title": "Front Matter",
                "page_start": requested_min,
            }
        ]
        boundary_pages = sorted(entries_by_start)

    batches: list[dict[str, Any]] = []
    requested_iter = iter(requested_pages)
    current_page = next(requested_iter, None)

    for index, start_page in enumerate(boundary_pages):
        next_start = boundary_pages[index + 1] if index + 1 < len(boundary_pages) else requested_max + 1
        span_pages: list[int] = []
        while current_page is not None and current_page < next_start:
            if current_page >= start_page:
                span_pages.append(current_page)
            current_page = next(requested_iter, None)

        if not span_pages:
            continue

        source_entries = entries_by_start.get(start_page, [])
        titles = [str(entry["title"]) for entry in source_entries]
        levels = [int(entry["level"]) for entry in source_entries]
        if segment_pages is not None and len(span_pages) > segment_pages:
            for sub_pages in chunk_pages(span_pages, segment_pages):
                batches.append(
                    {
                        "pages": sub_pages,
                        "segment_source": "outline+pages",
                        "outline_titles": titles,
                        "outline_levels": levels,
                        "outline_page_start": start_page,
                    }
                )
            continue

        batches.append(
            {
                "pages": span_pages,
                "segment_source": "outline",
                "outline_titles": titles,
                "outline_levels": levels,
                "outline_page_start": start_page,
            }
        )

    return batches


def resolve_segment_batches(cfg: PipelineConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_pages, total_pages = resolve_requested_pages(cfg)
    if cfg.segment_by_outline:
        outline_entries = load_pdf_outline_entries(cfg.input_pdf, total_pages)
        outline_batches = build_outline_batches(
            outline_entries=outline_entries,
            requested_pages=requested_pages,
            segment_pages=cfg.segment_pages,
        )
        if outline_batches:
            return (
                outline_batches,
                {
                    "segment_strategy": "outline",
                    "outline_entry_count": len(outline_entries),
                    "selected_page_count": len(requested_pages),
                    "total_pdf_pages": total_pages,
                },
            )

        if cfg.segment_pages is not None:
            print(
                (
                    "[phase01] PDF outline was unavailable or unusable for the selected pages; "
                    "falling back to fixed page batches."
                ),
                file=sys.stderr,
                flush=True,
            )
            return (
                build_fixed_page_batches(requested_pages, cfg.segment_pages),
                {
                    "segment_strategy": "pages_fallback",
                    "outline_entry_count": len(outline_entries),
                    "selected_page_count": len(requested_pages),
                    "total_pdf_pages": total_pages,
                },
            )

        raise ValueError(
            "Segmented outline mode was requested, but the PDF did not expose a usable outline."
        )

    if cfg.segment_pages is None:
        raise ValueError("Fixed page segmentation requires --segment-pages.")

    return (
        build_fixed_page_batches(requested_pages, cfg.segment_pages),
        {
            "segment_strategy": "pages",
            "selected_page_count": len(requested_pages),
            "total_pdf_pages": total_pages,
        },
    )


def build_segment_cache_fingerprint(
    cfg: PipelineConfig,
    env_values: dict[str, str],
    output_format: str,
    source_pdf_sha256: str,
    requested_pages: list[int],
) -> str:
    env_keys = (
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "GEMINI_MODEL",
        "GEMINI_MODEL_NAME",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "CLAUDE_MODEL_NAME",
        "AZURE_ENDPOINT",
        "DEPLOYMENT_NAME",
    )
    payload = {
        "source_pdf_sha256": source_pdf_sha256,
        "output_format": output_format,
        "marker_executable": cfg.marker_executable,
        "use_llm": cfg.use_llm,
        "llm_service": cfg.llm_service if cfg.use_llm else None,
        "torch_device": resolve_torch_device(cfg, env_values),
        "disable_image_extraction": cfg.disable_image_extraction,
        "paginate_output": cfg.paginate_output,
        "page_range": cfg.page_range,
        "segment_by_outline": cfg.segment_by_outline,
        "segment_pages": cfg.segment_pages,
        "debug": cfg.debug,
        "requested_pages": format_page_range_spec(requested_pages),
        "env": {
            key: env_values[key]
            for key in env_keys
            if key in env_values and env_values[key]
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return "035214a07e9897d7"


def segment_name_for_pages(pages: list[int]) -> str:
    page_range = format_page_range_spec(pages)
    digest = hashlib.sha1(page_range.encode("utf-8")).hexdigest()[:8]
    return f"pages_{pages[0] + 1:04d}-{pages[-1] + 1:04d}_{digest}"


def load_completed_segment_record(segment_root: Path) -> dict[str, Any] | None:
    record_path = segment_root / "segment.json"
    if not record_path.exists():
        return None

    record = read_json(record_path)
    artifact_path = Path(str(record.get("artifact_path") or ""))
    if not artifact_path.exists():
        return None
    return record


def assemble_segment_logs(
    records: list[dict[str, Any]],
    stream_name: str,
    destination: Path,
) -> None:
    ensure_dir(destination.parent)
    with destination.open("wb") as handle:
        for record in records:
            label = ""
            outline_titles = record.get("outline_titles") or []
            if outline_titles:
                label = f" outline={' / '.join(str(title) for title in outline_titles)}"
            header = (
                f"===== {record['segment_name']} page_range={record['page_range']} "
                f"source={record.get('segment_source', 'pages')}{label} "
                f"artifact={record['artifact_path']} =====\n"
            ).encode("utf-8")
            handle.write(header)
            log_path = Path(str(record[f"{stream_name}_log"]))
            if log_path.exists():
                payload = log_path.read_bytes()
                handle.write(payload)
                if payload and not payload.endswith(b"\n"):
                    handle.write(b"\n")
            handle.write(b"\n")


def assemble_segmented_markdown(records: list[dict[str, Any]], destination: Path) -> None:
    parts: list[str] = []
    for record in records:
        payload = Path(str(record["artifact_path"])).read_text(encoding="utf-8").strip()
        if payload:
            parts.append(payload)

    content = "\n\n".join(parts).strip()
    if content:
        content += "\n"
    ensure_dir(destination.parent)
    destination.write_text(content, encoding="utf-8")


def shift_page_references(value: Any, delta: int) -> Any:
    if delta == 0:
        return copy.deepcopy(value)

    if isinstance(value, dict):
        shifted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"page_id", "page", "page_idx"} and isinstance(item, int):
                shifted[key] = item + delta
            else:
                shifted[key] = shift_page_references(item, delta)
        return shifted

    if isinstance(value, list):
        return [shift_page_references(item, delta) for item in value]

    if isinstance(value, str):
        return PAGE_REF_PATTERN.sub(
            lambda match: f"/page/{int(match.group(1)) + delta}/",
            value,
        )

    return copy.deepcopy(value)


def extract_zero_based_page_number(page_block: dict[str, Any]) -> int:
    for key in ("page_id", "page", "page_idx"):
        if key not in page_block:
            continue
        try:
            return int(page_block[key])
        except (TypeError, ValueError):
            continue

    match = PAGE_REF_PATTERN.search(str(page_block.get("id") or ""))
    if match:
        return int(match.group(1))
    return 0


def rebase_marker_segment_payload(payload: Any, expected_first_page: int) -> dict[str, Any]:
    pages, metadata = extract_pages_and_metadata(payload)
    if not pages:
        return {
            "block_type": "Document",
            "children": [],
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    observed_first_page = extract_zero_based_page_number(pages[0])
    delta = expected_first_page - observed_first_page
    rebased_pages = shift_page_references(pages, delta)
    rebased_metadata = shift_page_references(metadata, delta)
    return {
        "block_type": "Document",
        "children": rebased_pages,
        "metadata": rebased_metadata if isinstance(rebased_metadata, dict) else {},
    }


def assemble_segmented_json(records: list[dict[str, Any]], destination: Path) -> None:
    merged_pages: list[dict[str, Any]] = []
    merged_metadata_lists: dict[str, list[Any]] = {}
    merged_metadata_scalars: dict[str, Any] = {}

    for record in records:
        payload = read_json(Path(str(record["artifact_path"])))
        rebased_payload = rebase_marker_segment_payload(payload, int(record["pages"][0]))
        pages, metadata = extract_pages_and_metadata(rebased_payload)
        merged_pages.extend(pages)
        if not isinstance(metadata, dict):
            continue

        for key, value in metadata.items():
            if isinstance(value, list):
                merged_metadata_lists.setdefault(key, []).extend(value)
                continue
            if key not in merged_metadata_scalars and value not in (None, "", {}, []):
                merged_metadata_scalars[key] = value

    merged_pages.sort(key=extract_page_number)

    merged_metadata: dict[str, Any] = dict(merged_metadata_scalars)
    for key, rows in merged_metadata_lists.items():
        if all(isinstance(row, dict) and "page_id" in row for row in rows):
            rows = sorted(rows, key=lambda row: int(row["page_id"]))
        merged_metadata[key] = rows

    ensure_dir(destination.parent)
    write_json(
        destination,
        {
            "block_type": "Document",
            "children": merged_pages,
            "metadata": merged_metadata,
        },
    )


def run_segmented_marker_render(
    cfg: PipelineConfig,
    output_format: str,
    env_values: dict[str, str],
    artifacts_dir: Path,
    source_pdf_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    if cfg.segment_pages is None and not cfg.segment_by_outline:
        raise ValueError(
            "Segmented Marker rendering requires --segment-pages or --segment-by-outline."
        )
    if not cfg.disable_image_extraction:
        raise ValueError(
            "Segmented Marker mode currently requires --disable-image-extraction "
            "so the assembled markdown artifact does not end up with broken image paths."
        )

    segment_batches, segmentation_meta = resolve_segment_batches(cfg)
    requested_pages, total_pages = resolve_requested_pages(cfg)
    render_dir = ensure_dir(artifacts_dir / output_format)
    segmented_dir = ensure_dir(render_dir / SEGMENTED_ARTIFACT_DIRNAME)
    extension = ".md" if output_format == "markdown" else ".json"
    assembled_artifact = segmented_dir / f"{cfg.doc_id}{extension}"
    segment_manifest_path = segmented_dir / "segments.json"
    stdout_log_path = render_dir / "marker.stdout.log"
    stderr_log_path = render_dir / "marker.stderr.log"

    cache_root = ensure_dir(
        default_segment_cache_root()
        / cfg.doc_id
        / build_segment_cache_fingerprint(
            cfg=cfg,
            env_values=env_values,
            output_format=output_format,
            source_pdf_sha256=source_pdf_sha256,
            requested_pages=requested_pages,
        )
        / output_format
    )

    print(
        (
            f"[phase01] Segmented Marker mode is enabled for {output_format}: "
            f"{len(segment_batches)} batches via {segmentation_meta['segment_strategy']}. "
            "This improves resumability but may slightly change boundary behavior "
            "compared with a single full-document Marker run."
        ),
        file=sys.stderr,
        flush=True,
    )

    records: list[dict[str, Any]] = []
    for segment_index, batch in enumerate(segment_batches, start=1):
        pages = list(batch["pages"])
        page_range = format_page_range_spec(pages)
        segment_name = segment_name_for_pages(pages)
        segment_root = ensure_dir(cache_root / segment_name)
        existing_record = None if cfg.force else load_completed_segment_record(segment_root)
        if existing_record is not None:
            print(
                (
                    f"[phase01] Reusing completed segment {segment_index}/{len(segment_batches)} "
                    f"for {output_format}: {segment_name} ({page_range})"
                ),
                file=sys.stderr,
                flush=True,
            )
            records.append(existing_record)
            continue

        print(
            (
                f"[phase01] Rendering segment {segment_index}/{len(segment_batches)} "
                f"for {output_format}: {segment_name} ({page_range})"
            ),
            file=sys.stderr,
            flush=True,
        )
        segment_cfg = dataclasses.replace(
            cfg,
            page_range=page_range,
            force=True,
        )
        artifact, command = run_marker_render(
            cfg=segment_cfg,
            output_format=output_format,
            env_values=env_values,
            artifacts_dir=segment_root,
        )
        segment_render_dir = segment_root / output_format
        record = {
            "segment_name": segment_name,
            "segment_index": segment_index,
            "page_range": page_range,
            "pages": pages,
            "segment_source": str(batch["segment_source"]),
            "outline_titles": list(batch.get("outline_titles") or []),
            "outline_levels": list(batch.get("outline_levels") or []),
            "outline_page_start": batch.get("outline_page_start"),
            "artifact_path": str(artifact),
            "stdout_log": str(segment_render_dir / "marker.stdout.log"),
            "stderr_log": str(segment_render_dir / "marker.stderr.log"),
            "command": command,
            "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        write_json(segment_root / "segment.json", record)
        records.append(record)

    records.sort(key=lambda record: int(record["pages"][0]))
    assemble_segment_logs(records, "stdout", stdout_log_path)
    assemble_segment_logs(records, "stderr", stderr_log_path)
    if output_format == "markdown":
        assemble_segmented_markdown(records, assembled_artifact)
    else:
        assemble_segmented_json(records, assembled_artifact)

    segment_manifest = {
        "doc_id": cfg.doc_id,
        "output_format": output_format,
        "segment_strategy": segmentation_meta["segment_strategy"],
        "segment_by_outline": cfg.segment_by_outline,
        "segment_pages": cfg.segment_pages,
        "requested_page_range": cfg.page_range,
        "selected_page_count": segmentation_meta["selected_page_count"],
        "total_pdf_pages": segmentation_meta["total_pdf_pages"],
        "outline_entry_count": segmentation_meta.get("outline_entry_count"),
        "cache_root": str(cache_root),
        "assembled_artifact": str(assembled_artifact),
        "segments": records,
    }
    write_json(segment_manifest_path, segment_manifest)
    return (
        assembled_artifact,
        (
            f"segmented_marker[{output_format}] x{len(records)} via "
            f"{segmentation_meta['segment_strategy']} segmentation; see {segment_manifest_path}"
        ),
        {
            "segmented": True,
            "segment_strategy": segmentation_meta["segment_strategy"],
            "segment_by_outline": cfg.segment_by_outline,
            "segment_pages": cfg.segment_pages,
            "segment_count": len(records),
            "selected_page_count": segmentation_meta["selected_page_count"],
            "total_pdf_pages": segmentation_meta["total_pdf_pages"],
            "outline_entry_count": segmentation_meta.get("outline_entry_count"),
            "checkpoint_root": str(cache_root),
            "segment_manifest": str(segment_manifest_path),
        },
    )


def render_marker_output(
    cfg: PipelineConfig,
    output_format: str,
    env_values: dict[str, str],
    artifacts_dir: Path,
    source_pdf_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    if cfg.segment_pages is not None or cfg.segment_by_outline:
        return run_segmented_marker_render(
            cfg=cfg,
            output_format=output_format,
            env_values=env_values,
            artifacts_dir=artifacts_dir,
            source_pdf_sha256=source_pdf_sha256,
        )

    artifact, command = run_marker_render(
        cfg=cfg,
        output_format=output_format,
        env_values=env_values,
        artifacts_dir=artifacts_dir,
    )
    return artifact, command, {"segmented": False}


def run_marker_render(
    cfg: PipelineConfig,
    output_format: str,
    env_values: dict[str, str],
    artifacts_dir: Path,
) -> tuple[Path, str]:
    render_dir = ensure_dir(artifacts_dir / output_format)
    preferred_stems = [cfg.input_pdf.stem, cfg.doc_id]
    stdout_log_path = render_dir / "marker.stdout.log"
    stderr_log_path = render_dir / "marker.stderr.log"

    if not cfg.force:
        try:
            existing = find_primary_artifact(render_dir, output_format, preferred_stems)
            print(
                f"[phase01] Reusing existing Marker {output_format} artifact: {existing}",
                file=sys.stderr,
                flush=True,
            )
            return existing, "skipped_existing"
        except FileNotFoundError:
            pass

    command = build_marker_command(cfg, render_dir, output_format, env_values)
    runtime_env = build_marker_runtime_env(cfg, env_values)
    print(
        f"[phase01] Running Marker for {output_format} output...",
        file=sys.stderr,
        flush=True,
    )
    process = subprocess.Popen(
        command,
        env=merged_env(runtime_env),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    def tee_pipe(pipe: Any, log_path: Path) -> None:
        with log_path.open("wb") as log_handle:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                log_handle.write(chunk)
                log_handle.flush()
                sink = getattr(sys.stderr, "buffer", None)
                if sink is not None:
                    sink.write(chunk)
                    sink.flush()
                else:
                    sys.stderr.write(chunk.decode("utf-8", errors="replace"))
                    sys.stderr.flush()
        pipe.close()

    stdout_thread = threading.Thread(
        target=tee_pipe,
        args=(process.stdout, stdout_log_path),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=tee_pipe,
        args=(process.stderr, stderr_log_path),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, redact_command(command))

    artifact = find_primary_artifact(render_dir, output_format, preferred_stems)
    print(
        f"[phase01] Marker {output_format} output is ready: {artifact}",
        file=sys.stderr,
        flush=True,
    )
    return artifact, redact_command(command)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def html_to_text(fragment: str) -> str:
    if not fragment:
        return ""

    text = fragment
    text = re.sub(r"<content-ref[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(tbody|thead|table|ul|ol)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*(p|div|section|article|h[1-6]|caption)\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*li[^>]*>", "- ", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*li\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*tr[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*tr\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*t[hd]\s*>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_display_text(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\ufeff": "",
        "\u200b": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WORD_HYPHENATION_PATTERN.sub("", text)
    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    for line in lines:
        candidate = re.sub(r"\s+", " ", line).strip()
        if not candidate:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if PAGE_MARKER_PATTERN.fullmatch(candidate):
            continue
        cleaned_lines.append(candidate)

    normalized = "\n".join(cleaned_lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def iter_visual_marker_blocks(payload: Any) -> Iterable[dict[str, Any]]:
    pages, _metadata = extract_pages_and_metadata(payload)

    for page in pages:
        page_num = extract_page_number(page)

        def walk(node: dict[str, Any]) -> Iterable[dict[str, Any]]:
            block_type = str(node.get("block_type") or "")
            if block_type in {"Figure", "Picture"}:
                yield {
                    "marker_block_id": str(node.get("id") or ""),
                    "source_block_type": block_type,
                    "page_num": page_num,
                    "bbox": copy.deepcopy(node.get("bbox")),
                    "polygon": copy.deepcopy(node.get("polygon")),
                }

            children = node.get("children") or []
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        yield from walk(child)

        yield from walk(page)


def clean_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def export_marker_image_assets(
    *,
    cfg: PipelineConfig,
    marker_payload: Any,
    assets_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency is provided by phase01 requirements
        raise RuntimeError("Phase-01 image asset export requires PyMuPDF.") from exc

    ensure_dir(assets_root)
    assets_by_marker_id: dict[str, list[dict[str, Any]]] = {}
    exported_count = 0
    skipped_count = 0

    with fitz.open(cfg.input_pdf) as document:
        for block in iter_visual_marker_blocks(marker_payload):
            marker_block_id = block["marker_block_id"]
            if not marker_block_id:
                skipped_count += 1
                continue

            bbox = clean_bbox(block.get("bbox"))
            if bbox is None:
                skipped_count += 1
                continue

            page_num = int(block["page_num"])
            page_index = page_num - 1
            if page_index < 0 or page_index >= len(document):
                skipped_count += 1
                continue

            page = document[page_index]
            clip = fitz.Rect(*bbox) & page.rect
            if clip.is_empty or clip.width <= 1 or clip.height <= 1:
                skipped_count += 1
                continue

            safe_stem = slugify(marker_block_id)
            asset_path = assets_root / f"page_{page_num:04d}_{safe_stem}.png"
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
            pixmap.save(asset_path)
            exported_count += 1

            asset = {
                "asset_id": f"{cfg.doc_id}_image_{exported_count:05d}",
                "marker_block_id": marker_block_id,
                "source_block_type": block["source_block_type"],
                "path": str(asset_path),
                "mime_type": "image/png",
                "page_start": page_num,
                "page_end": page_num,
                "bbox": bbox,
                "polygon": block.get("polygon"),
                "width_px": int(pixmap.width),
                "height_px": int(pixmap.height),
                "sha256": sha256_file(asset_path),
            }
            assets_by_marker_id.setdefault(marker_block_id, []).append(asset)

    manifest = {
        "enabled": True,
        "root": str(assets_root),
        "exported_count": exported_count,
        "skipped_count": skipped_count,
    }
    return assets_by_marker_id, manifest


def extract_pages_and_metadata(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], {}

    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        if isinstance(payload.get("children"), list):
            return [item for item in payload["children"] if isinstance(item, dict)], metadata
        if isinstance(payload.get("pages"), list):
            return [item for item in payload["pages"] if isinstance(item, dict)], metadata
    raise ValueError("Marker JSON payload is not in a recognized shape.")


def extract_page_number(page_block: dict[str, Any]) -> int:
    for key in ("page_id", "page", "page_idx"):
        if key in page_block:
            try:
                return int(page_block[key]) + 1
            except (TypeError, ValueError):
                pass

    block_id = str(page_block.get("id") or "")
    match = re.search(r"/page/(\d+)/", block_id)
    if match:
        return int(match.group(1)) + 1
    return 1


def flatten_blocks(page_block: dict[str, Any]) -> list[dict[str, Any]]:
    page_id = extract_page_number(page_block)
    flattened: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        block_type = str(node.get("block_type") or "")
        children = node.get("children") or []

        if block_type not in GROUP_ONLY_BLOCK_TYPES and block_type not in IGNORED_BLOCK_TYPES:
            flattened.append(
                {
                    "page_num": page_id,
                    "id": node.get("id"),
                    "block_type": block_type,
                    "html": node.get("html") or "",
                    "bbox": copy.deepcopy(node.get("bbox")),
                    "polygon": copy.deepcopy(node.get("polygon")),
                    "section_hierarchy": copy.deepcopy(node.get("section_hierarchy") or {}),
                    "children_count": len(children) if isinstance(children, list) else 0,
                }
            )

        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk(child)

    walk(page_block)
    return flattened


def build_section_header_lookup(flat_blocks: Iterable[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for block in flat_blocks:
        if block["block_type"] != "SectionHeader":
            continue
        text = normalize_display_text(html_to_text(block["html"]))
        if text:
            lookup[str(block["id"])] = text
    return lookup


def heading_path_for_block(
    block: dict[str, Any],
    section_lookup: dict[str, str],
) -> list[str]:
    hierarchy = block.get("section_hierarchy") or {}
    items: list[tuple[int, str]] = []
    if isinstance(hierarchy, dict):
        for level, block_id in hierarchy.items():
            try:
                numeric_level = int(level)
            except (TypeError, ValueError):
                continue
            text = section_lookup.get(str(block_id), "")
            if text:
                items.append((numeric_level, text))

    items.sort(key=lambda item: item[0])
    heading_path = [text for _, text in items]

    if block["block_type"] == "SectionHeader":
        current_text = normalize_display_text(html_to_text(block["html"]))
        if current_text and (not heading_path or heading_path[-1] != current_text):
            heading_path.append(current_text)

    return heading_path


def classify_semantic_hint(block_type: str, heading_path: list[str], text: str) -> str:
    joined_heading = " / ".join(heading_path)
    if block_type == "TableOfContents":
        return "front_matter"
    if any(pattern.search(joined_heading) for pattern in FRONT_MATTER_PATTERNS):
        return "front_matter"
    if any(pattern.search(text) for pattern in FRONT_MATTER_PATTERNS):
        return "front_matter"
    if block_type == "Table":
        return "table"
    if block_type in {"ListGroup", "ListItem"}:
        if any(pattern.search(joined_heading) for pattern in PROCEDURE_PATTERNS):
            return "procedure"
        return "list"
    if block_type in {"Code", "Figure", "Picture", "Caption"}:
        return block_type.lower()
    if any(pattern.search(joined_heading) for pattern in PROCEDURE_PATTERNS):
        return "procedure"
    return "section"


def flatten_marker_json_to_normalized_blocks(
    payload: Any,
    media_assets_by_marker_id: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pages, metadata = extract_pages_and_metadata(payload)
    flat_blocks: list[dict[str, Any]] = []
    for page in pages:
        flat_blocks.extend(flatten_blocks(page))

    section_lookup = build_section_header_lookup(flat_blocks)
    normalized_blocks: list[dict[str, Any]] = []
    counter = 0
    for block in flat_blocks:
        block_type = block["block_type"]
        if block_type in IGNORED_BLOCK_TYPES or block_type in GROUP_ONLY_BLOCK_TYPES:
            continue
        if block_type not in TEXTUAL_BLOCK_TYPES:
            continue

        display_text = normalize_display_text(html_to_text(block["html"]))
        if not display_text:
            continue

        if block_type == "SectionHeader":
            # Section headers are carried through via heading_path metadata instead of
            # becoming standalone searchable chunks.
            continue

        heading_path = heading_path_for_block(block, section_lookup)
        semantic_hint = classify_semantic_hint(block_type, heading_path, display_text)
        indexable = semantic_hint != "front_matter"
        marker_block_id = str(block["id"] or "")
        media_assets = list((media_assets_by_marker_id or {}).get(marker_block_id) or [])

        counter += 1
        normalized_block = {
            "block_order": counter,
            "block_id": f"block_{counter:05d}",
            "marker_block_id": block["id"],
            "source_block_type": block_type,
            "semantic_hint": semantic_hint,
            "heading_path": heading_path,
            "display_text": display_text,
            "search_text": normalize_search_text(display_text),
            "page_start": block["page_num"],
            "page_end": block["page_num"],
            "indexable": indexable,
        }
        if media_assets:
            normalized_block["media_assets"] = media_assets
            normalized_block["has_image"] = True
        normalized_blocks.append(normalized_block)

    return normalized_blocks, metadata


def normalize_markdown_headings(markdown: str) -> list[dict[str, Any]]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_type = "paragraph"
    counter = 0

    def flush() -> None:
        nonlocal buffer, buffer_type, counter
        if not buffer:
            return
        raw_text = "\n".join(buffer).strip()
        display_text = normalize_display_text(raw_text)
        if not display_text:
            buffer = []
            buffer_type = "paragraph"
            return

        semantic_hint = classify_semantic_hint(
            buffer_type.title(),
            [text for _, text in heading_stack],
            display_text,
        )
        counter += 1
        blocks.append(
            {
                "block_order": counter,
                "block_id": f"fallback_{counter:05d}",
                "marker_block_id": None,
                "source_block_type": buffer_type.title(),
                "semantic_hint": semantic_hint,
                "heading_path": [text for _, text in heading_stack],
                "display_text": display_text,
                "search_text": normalize_search_text(display_text),
                "page_start": None,
                "page_end": None,
                "indexable": semantic_hint != "front_matter",
            }
        )
        buffer = []
        buffer_type = "paragraph"

    index = 0
    while index < len(lines):
        line = lines[index]
        heading_match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            text = normalize_display_text(heading_match.group(2))
            heading_stack[:] = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, text))
            index += 1
            continue

        stripped = line.strip()
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if stripped and next_line and set(next_line) <= {"=", "-"} and len(next_line) >= len(stripped):
            flush()
            level = 1 if next_line.startswith("=") else 2
            text = normalize_display_text(stripped)
            heading_stack[:] = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, text))
            index += 2
            continue

        if not stripped:
            flush()
            index += 1
            continue

        is_table = "|" in line and index + 1 < len(lines) and re.search(r"^\s*\|?[-: ]+\|[-|: ]+$", lines[index + 1])
        is_list = bool(re.match(r"^\s*([-*+]|\d+\.)\s+", line))

        new_type = "table" if is_table else "list" if is_list else "paragraph"
        if buffer and new_type != buffer_type:
            flush()
        buffer_type = new_type
        buffer.append(line)

        if is_table:
            index += 1
            while index < len(lines):
                buffer.append(lines[index])
                if index + 1 >= len(lines) or "|" not in lines[index + 1]:
                    break
                index += 1
        index += 1

    flush()
    return blocks


def first_not_none(left: int | None, right: int | None) -> int | None:
    return left if left is not None else right


def max_not_none(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def build_structural_chunks(doc_id: str, normalized_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    accumulator: dict[str, Any] | None = None
    counter = 0

    def flush_accumulator() -> None:
        nonlocal accumulator, counter
        if not accumulator:
            return
        counter += 1
        accumulator["chunk_order"] = counter
        accumulator["chunk_id"] = f"{doc_id}_chunk_{counter:05d}"
        accumulator["search_text"] = normalize_search_text(accumulator["display_text"])
        chunks.append(accumulator)
        accumulator = None

    for block in normalized_blocks:
        semantic_hint = block["semantic_hint"]
        standalone = semantic_hint in {"table", "figure", "picture", "caption", "code", "list"}
        block_media_assets = list(block.get("media_assets") or [])
        if standalone:
            flush_accumulator()
            counter += 1
            chunk = {
                "chunk_order": counter,
                "chunk_id": f"{doc_id}_chunk_{counter:05d}",
                "chunk_type": semantic_hint if semantic_hint in {"table", "list"} else "section",
                "semantic_hint": semantic_hint,
                "heading_path": block["heading_path"],
                "display_text": block["display_text"],
                "search_text": block["search_text"],
                "page_start": block["page_start"],
                "page_end": block["page_end"],
                "indexable": block["indexable"],
                "source_block_ids": [block["block_id"]],
                "source_marker_block_ids": [block["marker_block_id"]] if block["marker_block_id"] else [],
                "has_image": bool(block_media_assets),
            }
            if block_media_assets:
                chunk["media_assets"] = block_media_assets
            chunks.append(chunk)
            continue

        chunk_type = (
            "front_matter"
            if semantic_hint == "front_matter"
            else "procedure"
            if semantic_hint == "procedure"
            else "section"
        )
        key = (
            tuple(block["heading_path"]),
            chunk_type,
            block["indexable"],
        )

        if accumulator is None or accumulator["_group_key"] != key:
            flush_accumulator()
            accumulator = {
                "_group_key": key,
                "chunk_type": chunk_type,
                "semantic_hint": semantic_hint,
                "heading_path": block["heading_path"],
                "display_text": block["display_text"],
                "page_start": block["page_start"],
                "page_end": block["page_end"],
                "indexable": block["indexable"],
                "source_block_ids": [block["block_id"]],
                "source_marker_block_ids": [block["marker_block_id"]] if block["marker_block_id"] else [],
                "media_assets": block_media_assets,
                "has_image": bool(block_media_assets),
            }
            continue

        accumulator["display_text"] = (
            accumulator["display_text"].rstrip() + "\n\n" + block["display_text"].lstrip()
        )
        accumulator["page_start"] = first_not_none(accumulator["page_start"], block["page_start"])
        accumulator["page_end"] = max_not_none(accumulator["page_end"], block["page_end"])
        accumulator["source_block_ids"].append(block["block_id"])
        if block["marker_block_id"]:
            accumulator["source_marker_block_ids"].append(block["marker_block_id"])
        if block_media_assets:
            accumulator.setdefault("media_assets", []).extend(block_media_assets)
            accumulator["has_image"] = True

    flush_accumulator()
    for chunk in chunks:
        chunk.pop("_group_key", None)
        if not chunk.get("media_assets"):
            chunk.pop("media_assets", None)
            chunk["has_image"] = False
    return chunks


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    if not cfg.input_pdf.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {cfg.input_pdf}")

    print(f"[phase01] Starting pipeline for {cfg.input_pdf}", file=sys.stderr, flush=True)
    # Prefer explicitly loaded .env values, but also honor environment variables
    # already injected by Docker Compose or the parent shell.
    env_values = merged_env(load_simple_env(cfg.env_file))
    effective_torch_device = resolve_torch_device(cfg, env_values)
    if effective_torch_device:
        print(
            f"[phase01] Marker torch device: {effective_torch_device}",
            file=sys.stderr,
            flush=True,
        )
    source_pdf_sha256 = sha256_file(cfg.input_pdf)

    base_output = ensure_dir(cfg.output_root)
    manifests_dir = ensure_dir(base_output / "manifests")
    marker_raw_dir = ensure_dir(base_output / "marker_raw" / cfg.doc_id)
    image_assets_dir = base_output / "image_assets" / cfg.doc_id
    normalized_dir = ensure_dir(base_output / "normalized_blocks")
    chunks_dir = ensure_dir(base_output / "structural_chunks")

    markdown_artifact, markdown_command, markdown_render_meta = render_marker_output(
        cfg=cfg,
        output_format="markdown",
        env_values=env_values,
        artifacts_dir=marker_raw_dir,
        source_pdf_sha256=source_pdf_sha256,
    )
    normalized_source = "markdown"
    json_artifact: Path | None = None
    json_command: str | None = None
    json_render_meta: dict[str, Any] = {"segmented": False}
    marker_metadata: dict[str, Any] = {}
    image_asset_manifest: dict[str, Any] = {
        "enabled": cfg.export_image_assets,
        "root": str(image_assets_dir),
        "exported_count": 0,
        "skipped_count": 0,
    }
    media_assets_by_marker_id: dict[str, list[dict[str, Any]]] = {}

    print("[phase01] Normalizing markdown output...", file=sys.stderr, flush=True)
    fallback_markdown = markdown_artifact.read_text(encoding="utf-8")
    normalized_blocks = normalize_markdown_headings(fallback_markdown)

    if cfg.emit_json:
        json_artifact, json_command, json_render_meta = render_marker_output(
            cfg=cfg,
            output_format="json",
            env_values=env_values,
            artifacts_dir=marker_raw_dir,
            source_pdf_sha256=source_pdf_sha256,
        )
        print("[phase01] Normalizing Marker JSON output...", file=sys.stderr, flush=True)
        marker_payload = read_json(json_artifact)
        if cfg.export_image_assets:
            print("[phase01] Exporting image assets from Marker block geometry...", file=sys.stderr, flush=True)
            media_assets_by_marker_id, image_asset_manifest = export_marker_image_assets(
                cfg=cfg,
                marker_payload=marker_payload,
                assets_root=image_assets_dir,
            )
        json_normalized_blocks, marker_metadata = flatten_marker_json_to_normalized_blocks(
            marker_payload,
            media_assets_by_marker_id=media_assets_by_marker_id,
        )
        if json_normalized_blocks:
            normalized_blocks = json_normalized_blocks
            normalized_source = "json"
        else:
            print(
                "[phase01] Marker JSON had no usable blocks, keeping markdown normalization...",
                file=sys.stderr,
                flush=True,
            )

    print("[phase01] Building structural chunks...", file=sys.stderr, flush=True)
    structural_chunks = build_structural_chunks(cfg.doc_id, normalized_blocks)

    normalized_path = normalized_dir / f"{cfg.doc_id}.jsonl"
    structural_chunks_path = chunks_dir / f"{cfg.doc_id}.jsonl"
    manifest_path = manifests_dir / f"{cfg.doc_id}.json"

    write_jsonl(normalized_path, normalized_blocks)
    write_jsonl(structural_chunks_path, structural_chunks)

    manifest = {
        "doc_id": cfg.doc_id,
        "source_pdf": str(cfg.input_pdf),
        "source_pdf_sha256": source_pdf_sha256,
        "source_pdf_size_bytes": cfg.input_pdf.stat().st_size,
        "processed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "marker": {
            "markdown_artifact": str(markdown_artifact),
            "json_artifact": str(json_artifact) if json_artifact else None,
            "markdown_command": markdown_command,
            "json_command": json_command,
            "use_llm": cfg.use_llm,
            "llm_service": cfg.llm_service if cfg.use_llm else None,
            "torch_device": effective_torch_device,
            "emit_json": cfg.emit_json,
            "normalization_source": normalized_source,
            "paginate_output": cfg.paginate_output,
            "disable_image_extraction": cfg.disable_image_extraction,
            "page_range": cfg.page_range,
            "segmented_run": bool(markdown_render_meta.get("segmented") or json_render_meta.get("segmented")),
            "segment_by_outline": bool(
                markdown_render_meta.get("segment_by_outline")
                or json_render_meta.get("segment_by_outline")
                or cfg.segment_by_outline
            ),
            "segment_strategy": (
                markdown_render_meta.get("segment_strategy")
                or json_render_meta.get("segment_strategy")
            ),
            "segment_pages": cfg.segment_pages,
            "segment_count": markdown_render_meta.get("segment_count") or json_render_meta.get("segment_count"),
            "selected_page_count": (
                markdown_render_meta.get("selected_page_count")
                or json_render_meta.get("selected_page_count")
            ),
            "total_pdf_pages": (
                markdown_render_meta.get("total_pdf_pages")
                or json_render_meta.get("total_pdf_pages")
            ),
            "outline_entry_count": (
                markdown_render_meta.get("outline_entry_count")
                or json_render_meta.get("outline_entry_count")
            ),
            "checkpoint_root": (
                markdown_render_meta.get("checkpoint_root")
                or json_render_meta.get("checkpoint_root")
            ),
            "markdown_segment_manifest": markdown_render_meta.get("segment_manifest"),
            "json_segment_manifest": json_render_meta.get("segment_manifest"),
            "debug": cfg.debug,
            "export_image_assets": cfg.export_image_assets,
        },
        "artifacts": {
            "normalized_blocks": str(normalized_path),
            "structural_chunks": str(structural_chunks_path),
            "image_assets": str(image_assets_dir) if cfg.export_image_assets else None,
        },
        "stats": {
            "normalized_block_count": len(normalized_blocks),
            "structural_chunk_count": len(structural_chunks),
            "indexable_chunk_count": sum(1 for chunk in structural_chunks if chunk["indexable"]),
            "image_asset_count": int(image_asset_manifest.get("exported_count") or 0),
            "image_asset_skipped_count": int(image_asset_manifest.get("skipped_count") or 0),
        },
        "image_assets": image_asset_manifest,
        "marker_metadata": marker_metadata,
    }
    print("[phase01] Writing manifest and chunk artifacts...", file=sys.stderr, flush=True)
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    cfg = parse_args()
    manifest = run_pipeline(cfg)
    print(
        json.dumps(
            {
                "doc_id": manifest["doc_id"],
                "normalized_block_count": manifest["stats"]["normalized_block_count"],
                "structural_chunk_count": manifest["stats"]["structural_chunk_count"],
                "normalized_blocks": manifest["artifacts"]["normalized_blocks"],
                "structural_chunks": manifest["artifacts"]["structural_chunks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
