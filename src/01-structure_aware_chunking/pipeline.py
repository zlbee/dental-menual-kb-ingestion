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


@dataclasses.dataclass(slots=True)
class PipelineConfig:
    input_pdf: Path
    output_root: Path
    env_file: Path | None
    doc_id: str
    marker_executable: str
    llm_service: str
    use_llm: bool
    emit_json: bool
    disable_image_extraction: bool
    paginate_output: bool
    page_range: str | None
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
        "--emit-json",
        action="store_true",
        help="Also render Marker JSON in addition to markdown. Defaults to markdown only.",
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
        emit_json=parsed.emit_json,
        disable_image_extraction=parsed.disable_image_extraction,
        paginate_output=parsed.paginate_output,
        page_range=parsed.page_range,
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
        if path.is_file() and path.suffix.lower() in suffixes
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
    print(
        f"[phase01] Running Marker for {output_format} output...",
        file=sys.stderr,
        flush=True,
    )
    process = subprocess.Popen(
        command,
        env=merged_env(env_values),
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


def flatten_marker_json_to_normalized_blocks(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

        counter += 1
        normalized_blocks.append(
            {
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
        )

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
        if standalone:
            flush_accumulator()
            counter += 1
            chunks.append(
                {
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
                }
            )
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

    flush_accumulator()
    for chunk in chunks:
        chunk.pop("_group_key", None)
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

    base_output = ensure_dir(cfg.output_root)
    manifests_dir = ensure_dir(base_output / "manifests")
    marker_raw_dir = ensure_dir(base_output / "marker_raw" / cfg.doc_id)
    normalized_dir = ensure_dir(base_output / "normalized_blocks")
    chunks_dir = ensure_dir(base_output / "structural_chunks")

    markdown_artifact, markdown_command = run_marker_render(
        cfg=cfg,
        output_format="markdown",
        env_values=env_values,
        artifacts_dir=marker_raw_dir,
    )
    normalized_source = "markdown"
    json_artifact: Path | None = None
    json_command: str | None = None
    marker_metadata: dict[str, Any] = {}

    print("[phase01] Normalizing markdown output...", file=sys.stderr, flush=True)
    fallback_markdown = markdown_artifact.read_text(encoding="utf-8")
    normalized_blocks = normalize_markdown_headings(fallback_markdown)

    if cfg.emit_json:
        json_artifact, json_command = run_marker_render(
            cfg=cfg,
            output_format="json",
            env_values=env_values,
            artifacts_dir=marker_raw_dir,
        )
        print("[phase01] Normalizing Marker JSON output...", file=sys.stderr, flush=True)
        marker_payload = read_json(json_artifact)
        json_normalized_blocks, marker_metadata = flatten_marker_json_to_normalized_blocks(marker_payload)
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
        "source_pdf_sha256": sha256_file(cfg.input_pdf),
        "source_pdf_size_bytes": cfg.input_pdf.stat().st_size,
        "processed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "marker": {
            "markdown_artifact": str(markdown_artifact),
            "json_artifact": str(json_artifact) if json_artifact else None,
            "markdown_command": markdown_command,
            "json_command": json_command,
            "use_llm": cfg.use_llm,
            "llm_service": cfg.llm_service if cfg.use_llm else None,
            "emit_json": cfg.emit_json,
            "normalization_source": normalized_source,
            "paginate_output": cfg.paginate_output,
            "disable_image_extraction": cfg.disable_image_extraction,
            "page_range": cfg.page_range,
            "debug": cfg.debug,
        },
        "artifacts": {
            "normalized_blocks": str(normalized_path),
            "structural_chunks": str(structural_chunks_path),
        },
        "stats": {
            "normalized_block_count": len(normalized_blocks),
            "structural_chunk_count": len(structural_chunks),
            "indexable_chunk_count": sum(1 for chunk in structural_chunks if chunk["indexable"]),
        },
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
