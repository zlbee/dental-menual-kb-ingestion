from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional fallback for local linting
    tiktoken = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE01_ROOT = REPO_ROOT / "data" / "processed" / "01_structure_aware"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "02_semantic_chunking"
DEFAULT_TIKTOKEN_MODEL = "text-embedding-3-small"
SENTENCE_SPLIT_REGEX = r"(?<=[.!?。！？])\s+|\n{2,}"
SEMANTIC_HINT_VISUALS = {"caption", "figure", "picture"}
SEMANTIC_HINT_TEXTUAL = {"procedure", "section"}


@dataclasses.dataclass(slots=True)
class PipelineConfig:
    input_manifest: Path | None
    doc_id: str | None
    phase01_root: Path
    output_root: Path
    env_file: Path | None
    breakpoint_threshold_type: str
    breakpoint_threshold_amount: float | None
    buffer_size: int
    min_chunk_tokens: int
    target_chunk_tokens: int
    max_chunk_tokens: int
    list_max_items: int
    list_target_tokens: int
    tiktoken_model_name: str
    force: bool
    debug: bool


@dataclasses.dataclass(slots=True)
class Phase2ParentUnit:
    unit_order: int
    unit_kind: str
    chunk_type: str
    semantic_hint: str
    source_semantic_hints: list[str]
    heading_path: list[str]
    display_text: str
    indexable: bool
    page_start: int | None
    page_end: int | None
    source_structural_chunk_ids: list[str]
    source_structural_chunk_orders: list[int]
    source_block_ids: list[str]
    source_marker_block_ids: list[str]


@dataclasses.dataclass(slots=True)
class BlockSpan:
    block_id: str
    marker_block_id: str | None
    page_start: int | None
    page_end: int | None
    start: int
    end: int


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run phase-02 semantic chunking inside phase-01 structural boundaries "
            "using LangChain SemanticChunker and OpenRouter-hosted OpenAI embeddings."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=None,
        help="Phase-01 manifest path. If omitted, resolve from --doc-id and --phase01-root.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Document id to load from phase-01 manifests when --input-manifest is omitted.",
    )
    parser.add_argument(
        "--phase01-root",
        type=Path,
        default=DEFAULT_PHASE01_ROOT,
        help="Root directory for phase-01 outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for phase-02 outputs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Optional .env file used to hydrate OpenRouter/OpenAI-compatible settings.",
    )
    parser.add_argument(
        "--breakpoint-threshold-type",
        choices=("percentile", "standard_deviation", "interquartile", "gradient"),
        default="percentile",
        help="SemanticChunker breakpoint strategy.",
    )
    parser.add_argument(
        "--breakpoint-threshold-amount",
        type=float,
        default=95.0,
        help="SemanticChunker breakpoint amount. Defaults to 95 for percentile mode.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1,
        help="SemanticChunker sentence buffer size.",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=120,
        help="Merge semantic chunks smaller than this threshold into neighbors when possible.",
    )
    parser.add_argument(
        "--target-chunk-tokens",
        type=int,
        default=450,
        help="Preferred chunk size for narrative text. Chunks lean slightly larger for fuller context.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=800,
        help="Hard cap for oversized chunks after semantic splitting.",
    )
    parser.add_argument(
        "--list-max-items",
        type=int,
        default=5,
        help="Maximum number of adjacent list items to group into one phase-02 parent unit.",
    )
    parser.add_argument(
        "--list-target-tokens",
        type=int,
        default=450,
        help="Preferred token budget when grouping adjacent list items.",
    )
    parser.add_argument(
        "--tiktoken-model-name",
        type=str,
        default=DEFAULT_TIKTOKEN_MODEL,
        help="Tokenizer model used for local token estimation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing phase-02 outputs for the target doc_id.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit extra phase-02 diagnostics to stderr.",
    )
    parsed = parser.parse_args()

    if parsed.input_manifest is None and parsed.doc_id is None:
        parser.error("Provide either --input-manifest or --doc-id.")

    return PipelineConfig(
        input_manifest=parsed.input_manifest.expanduser().resolve() if parsed.input_manifest else None,
        doc_id=parsed.doc_id,
        phase01_root=parsed.phase01_root.expanduser().resolve(),
        output_root=parsed.output_root.expanduser().resolve(),
        env_file=parsed.env_file.expanduser().resolve() if parsed.env_file else None,
        breakpoint_threshold_type=parsed.breakpoint_threshold_type,
        breakpoint_threshold_amount=parsed.breakpoint_threshold_amount,
        buffer_size=parsed.buffer_size,
        min_chunk_tokens=parsed.min_chunk_tokens,
        target_chunk_tokens=parsed.target_chunk_tokens,
        max_chunk_tokens=parsed.max_chunk_tokens,
        list_max_items=parsed.list_max_items,
        list_target_tokens=parsed.list_target_tokens,
        tiktoken_model_name=parsed.tiktoken_model_name,
        force=parsed.force,
        debug=parsed.debug,
    )


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def infer_data_root(phase01_root: Path) -> Path:
    if phase01_root.name == "01_structure_aware" and phase01_root.parent.name == "processed":
        return phase01_root.parent.parent
    return REPO_ROOT / "data"


def resolve_phase01_manifest_path(cfg: PipelineConfig) -> Path:
    if cfg.input_manifest:
        return cfg.input_manifest
    assert cfg.doc_id is not None
    return cfg.phase01_root / "manifests" / f"{cfg.doc_id}.json"


def resolve_artifact_path(raw_path: str, phase01_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    data_root = infer_data_root(phase01_root)
    attempts: list[Path] = []

    if raw_path.startswith("/app/data/"):
        suffix = raw_path.removeprefix("/app/data/")
        attempts.append(REPO_ROOT / "data" / suffix)
        attempts.append(data_root / suffix)
    elif raw_path.startswith("/app/"):
        suffix = raw_path.removeprefix("/app/")
        attempts.append(REPO_ROOT / suffix)
        attempts.append(data_root.parent / suffix)
    elif raw_path.startswith("data/"):
        attempts.append(REPO_ROOT / raw_path)
        attempts.append(data_root.parent / raw_path)

    filename = Path(raw_path).name
    attempts.append(phase01_root / "normalized_blocks" / filename)
    attempts.append(phase01_root / "structural_chunks" / filename)

    for attempt in attempts:
        if attempt.exists():
            return attempt

    raise FileNotFoundError(f"Could not resolve phase-01 artifact path: {raw_path}")


def build_token_estimator(model_name: str):
    encoder = None
    if tiktoken is not None:
        try:
            encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoder = None

    def estimate(text: str) -> int:
        compact = normalize_search_text(text)
        if not compact:
            return 0
        if encoder is not None:
            return len(encoder.encode(compact))
        return max(1, math.ceil(len(compact) / 4))

    return estimate


def build_embeddings(env_values: dict[str, str], tiktoken_model_name: str) -> OpenAIEmbeddings:
    required = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL")
    missing = [key for key in required if not env_values.get(key)]
    if missing:
        raise KeyError(
            "Phase-02 semantic chunking requires these env vars: " + ", ".join(sorted(missing))
        )

    return OpenAIEmbeddings(
        base_url=env_values["OPENAI_BASE_URL"],
        api_key=env_values["OPENAI_API_KEY"],
        model=env_values["OPENAI_EMBEDDING_MODEL"],
        tiktoken_model_name=tiktoken_model_name,
    )


def build_semantic_chunker(cfg: PipelineConfig, embeddings: OpenAIEmbeddings) -> SemanticChunker:
    return SemanticChunker(
        embeddings=embeddings,
        buffer_size=cfg.buffer_size,
        add_start_index=True,
        breakpoint_threshold_type=cfg.breakpoint_threshold_type,
        breakpoint_threshold_amount=cfg.breakpoint_threshold_amount,
        sentence_split_regex=SENTENCE_SPLIT_REGEX,
    )


def same_heading_path(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return list(left.get("heading_path") or []) == list(right.get("heading_path") or [])


def pages_are_adjacent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_end = left.get("page_end")
    right_start = right.get("page_start")
    if left_end is None or right_start is None:
        return True
    return right_start - left_end <= 1


def combine_chunks(
    pieces: list[dict[str, Any]],
    *,
    unit_order: int,
    unit_kind: str,
    chunk_type: str,
    semantic_hint: str,
    joiner: str,
) -> Phase2ParentUnit:
    return Phase2ParentUnit(
        unit_order=unit_order,
        unit_kind=unit_kind,
        chunk_type=chunk_type,
        semantic_hint=semantic_hint,
        source_semantic_hints=unique_preserve_order(piece["semantic_hint"] for piece in pieces),
        heading_path=list(pieces[0].get("heading_path") or []),
        display_text=joiner.join(piece["display_text"] for piece in pieces),
        indexable=all(bool(piece.get("indexable")) for piece in pieces),
        page_start=min(
            (piece["page_start"] for piece in pieces if piece.get("page_start") is not None),
            default=None,
        ),
        page_end=max(
            (piece["page_end"] for piece in pieces if piece.get("page_end") is not None),
            default=None,
        ),
        source_structural_chunk_ids=[piece["chunk_id"] for piece in pieces],
        source_structural_chunk_orders=[piece["chunk_order"] for piece in pieces],
        source_block_ids=unique_preserve_order(
            block_id
            for piece in pieces
            for block_id in piece.get("source_block_ids") or []
        ),
        source_marker_block_ids=unique_preserve_order(
            block_id
            for piece in pieces
            for block_id in piece.get("source_marker_block_ids") or []
            if block_id
        ),
    )


def build_phase2_parent_units(
    structural_chunks: list[dict[str, Any]],
    cfg: PipelineConfig,
    estimate_tokens,
) -> list[Phase2ParentUnit]:
    units: list[Phase2ParentUnit] = []
    index = 0
    unit_order = 0

    while index < len(structural_chunks):
        chunk = structural_chunks[index]
        semantic_hint = str(chunk.get("semantic_hint") or "section")

        if not chunk.get("indexable", True):
            unit_order += 1
            units.append(
                combine_chunks(
                    [chunk],
                    unit_order=unit_order,
                    unit_kind="passthrough",
                    chunk_type=chunk.get("chunk_type") or "section",
                    semantic_hint=semantic_hint,
                    joiner="\n\n",
                )
            )
            index += 1
            continue

        if semantic_hint == "table":
            unit_order += 1
            units.append(
                combine_chunks(
                    [chunk],
                    unit_order=unit_order,
                    unit_kind="table",
                    chunk_type="table",
                    semantic_hint="table",
                    joiner="\n\n",
                )
            )
            index += 1
            continue

        if semantic_hint in SEMANTIC_HINT_VISUALS:
            pieces = [chunk]
            if semantic_hint in {"figure", "picture"}:
                probe = index + 1
                while probe < len(structural_chunks):
                    candidate = structural_chunks[probe]
                    candidate_hint = str(candidate.get("semantic_hint") or "")
                    if candidate_hint != "caption":
                        break
                    if not same_heading_path(chunk, candidate) or not pages_are_adjacent(pieces[-1], candidate):
                        break
                    pieces.append(candidate)
                    probe += 1
                index = probe
            else:
                index += 1

            unit_order += 1
            units.append(
                combine_chunks(
                    pieces,
                    unit_order=unit_order,
                    unit_kind="visual",
                    chunk_type="visual",
                    semantic_hint="visual",
                    joiner="\n\n",
                )
            )
            continue

        if semantic_hint == "list":
            run: list[dict[str, Any]] = [chunk]
            probe = index + 1
            while probe < len(structural_chunks):
                candidate = structural_chunks[probe]
                if str(candidate.get("semantic_hint") or "") != "list":
                    break
                if not same_heading_path(chunk, candidate) or not pages_are_adjacent(run[-1], candidate):
                    break
                run.append(candidate)
                probe += 1

            group: list[dict[str, Any]] = []
            current_tokens = 0
            for item in run:
                item_tokens = estimate_tokens(item["display_text"])
                would_exceed = (
                    group
                    and (
                        len(group) >= cfg.list_max_items
                        or current_tokens + item_tokens > cfg.list_target_tokens
                    )
                )
                if would_exceed:
                    unit_order += 1
                    units.append(
                        combine_chunks(
                            group,
                            unit_order=unit_order,
                            unit_kind="list",
                            chunk_type="list",
                            semantic_hint="list",
                            joiner="\n",
                        )
                    )
                    group = []
                    current_tokens = 0
                group.append(item)
                current_tokens += item_tokens

            if group:
                unit_order += 1
                units.append(
                    combine_chunks(
                        group,
                        unit_order=unit_order,
                        unit_kind="list",
                        chunk_type="list",
                        semantic_hint="list",
                        joiner="\n",
                    )
                )

            index = probe
            continue

        unit_order += 1
        units.append(
            combine_chunks(
                [chunk],
                unit_order=unit_order,
                unit_kind="semantic",
                chunk_type=chunk.get("chunk_type") or "section",
                semantic_hint=semantic_hint if semantic_hint in SEMANTIC_HINT_TEXTUAL else "section",
                joiner="\n\n",
            )
        )
        index += 1

    return units


def merge_small_chunks(chunk_texts: list[str], min_chunk_tokens: int, estimate_tokens) -> list[str]:
    merged: list[str] = []
    index = 0

    while index < len(chunk_texts):
        current = chunk_texts[index].strip()
        if not current:
            index += 1
            continue

        while estimate_tokens(current) < min_chunk_tokens and index + 1 < len(chunk_texts):
            index += 1
            current = current.rstrip() + "\n\n" + chunk_texts[index].strip()

        if merged and estimate_tokens(current) < min_chunk_tokens:
            merged[-1] = merged[-1].rstrip() + "\n\n" + current.lstrip()
        else:
            merged.append(current)
        index += 1

    return merged


def pick_breakpoint(text: str, minimum_end: int, preferred_end: int, maximum_end: int) -> int:
    candidates = (
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "。",
        "！",
        "？",
        "; ",
        "；",
        ", ",
    )

    for token in candidates:
        index = text.rfind(token, minimum_end, preferred_end)
        if index >= 0:
            return index + len(token)
        index = text.find(token, preferred_end, maximum_end)
        if index >= 0:
            return index + len(token)
        index = text.rfind(token, minimum_end, maximum_end)
        if index >= 0:
            return index + len(token)

    return maximum_end


def hard_cap_chunk_text(
    text: str,
    *,
    target_chunk_tokens: int,
    max_chunk_tokens: int,
    estimate_tokens,
) -> list[str]:
    if estimate_tokens(text) <= max_chunk_tokens:
        return [text.strip()]

    min_chars = max(200, target_chunk_tokens * 3)
    target_chars = max(400, target_chunk_tokens * 4)
    max_chars = max(600, max_chunk_tokens * 4)

    chunks: list[str] = []
    cursor = 0
    length = len(text)

    while cursor < length:
        remaining = text[cursor:].lstrip()
        if not remaining:
            break

        trimmed_offset = len(text[cursor:]) - len(remaining)
        cursor += trimmed_offset

        if estimate_tokens(text[cursor:]) <= max_chunk_tokens:
            chunks.append(text[cursor:].strip())
            break

        minimum_end = min(length, cursor + min_chars)
        preferred_end = min(length, cursor + target_chars)
        maximum_end = min(length, cursor + max_chars)
        if maximum_end <= minimum_end:
            chunks.append(text[cursor:maximum_end].strip())
            cursor = maximum_end
            continue

        breakpoint_end = pick_breakpoint(text, minimum_end, preferred_end, maximum_end)
        if breakpoint_end <= cursor:
            breakpoint_end = maximum_end

        chunk = text[cursor:breakpoint_end].strip()
        if not chunk:
            break
        chunks.append(chunk)
        cursor = breakpoint_end

    return [chunk for chunk in chunks if chunk]


def split_parent_unit_text(
    unit: Phase2ParentUnit,
    *,
    chunker: SemanticChunker,
    cfg: PipelineConfig,
    estimate_tokens,
) -> list[str]:
    text = unit.display_text.strip()
    if not text:
        return []

    if unit.unit_kind in {"passthrough", "table", "visual"}:
        return [text]

    if unit.unit_kind == "list":
        return hard_cap_chunk_text(
            text,
            target_chunk_tokens=cfg.target_chunk_tokens,
            max_chunk_tokens=cfg.max_chunk_tokens,
            estimate_tokens=estimate_tokens,
        )

    if estimate_tokens(text) <= cfg.target_chunk_tokens:
        return [text]

    try:
        raw_documents = chunker.create_documents([text], metadatas=[{"unit_order": unit.unit_order}])
        chunk_texts = [document.page_content.strip() for document in raw_documents if document.page_content.strip()]
    except Exception as exc:
        raise RuntimeError(
            f"SemanticChunker failed for parent unit {unit.unit_order} "
            f"({unit.source_structural_chunk_ids})"
        ) from exc

    if not chunk_texts:
        chunk_texts = [text]

    chunk_texts = merge_small_chunks(chunk_texts, cfg.min_chunk_tokens, estimate_tokens)

    final_chunks: list[str] = []
    for chunk_text in chunk_texts:
        final_chunks.extend(
            hard_cap_chunk_text(
                chunk_text,
                target_chunk_tokens=cfg.target_chunk_tokens,
                max_chunk_tokens=cfg.max_chunk_tokens,
                estimate_tokens=estimate_tokens,
            )
        )

    return final_chunks or [text]


def assign_offsets(parent_text: str, child_texts: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for child_text in child_texts:
        query = child_text.strip()
        start = parent_text.find(query, cursor)
        if start < 0:
            start = parent_text.find(query)
        if start < 0:
            start = cursor
        end = min(len(parent_text), start + len(query))
        offsets.append((start, end))
        cursor = end

    return offsets


def build_block_spans(
    parent_text: str,
    source_block_ids: list[str],
    normalized_blocks_by_id: dict[str, dict[str, Any]],
) -> list[BlockSpan]:
    spans: list[BlockSpan] = []
    cursor = 0

    for block_id in source_block_ids:
        block = normalized_blocks_by_id.get(block_id)
        if block is None:
            continue
        block_text = str(block.get("display_text") or "")
        if not block_text:
            continue
        start = parent_text.find(block_text, cursor)
        if start < 0:
            start = parent_text.find(block_text)
        if start < 0:
            start = cursor
        end = min(len(parent_text), start + len(block_text))
        spans.append(
            BlockSpan(
                block_id=block_id,
                marker_block_id=block.get("marker_block_id"),
                page_start=block.get("page_start"),
                page_end=block.get("page_end"),
                start=start,
                end=end,
            )
        )
        cursor = end

    return spans


def block_ids_for_offset(
    spans: list[BlockSpan],
    *,
    start: int,
    end: int,
) -> tuple[list[str], list[str], int | None, int | None]:
    overlapping = [
        span
        for span in spans
        if not (span.end <= start or span.start >= end)
    ]
    if not overlapping and spans:
        midpoint = (start + end) // 2
        closest = min(
            spans,
            key=lambda span: abs(((span.start + span.end) // 2) - midpoint),
        )
        overlapping = [closest]

    source_block_ids = [span.block_id for span in overlapping]
    source_marker_block_ids = [
        span.marker_block_id
        for span in overlapping
        if span.marker_block_id
    ]
    page_start = min(
        (span.page_start for span in overlapping if span.page_start is not None),
        default=None,
    )
    page_end = max(
        (span.page_end for span in overlapping if span.page_end is not None),
        default=None,
    )
    return source_block_ids, source_marker_block_ids, page_start, page_end


def build_embedding_text(
    *,
    doc_id: str,
    heading_path: list[str],
    chunk_type: str,
    page_start: int | None,
    page_end: int | None,
    display_text: str,
) -> str:
    lines = [f"Document ID: {doc_id}"]
    if heading_path:
        lines.append("Heading Path: " + " > ".join(heading_path))
    lines.append(f"Chunk Type: {chunk_type}")
    if page_start is not None:
        if page_end is not None and page_end != page_start:
            lines.append(f"Pages: {page_start}-{page_end}")
        else:
            lines.append(f"Page: {page_start}")
    lines.append("")
    lines.append(display_text)
    return "\n".join(lines)


def finalize_neighbor_metadata(rows: list[dict[str, Any]]) -> None:
    sibling_count = len(rows)
    for index, row in enumerate(rows, start=1):
        row["sibling_index"] = index
        row["sibling_count"] = sibling_count
        row["prev_chunk_id"] = rows[index - 2]["chunk_id"] if index > 1 else None
        row["next_chunk_id"] = rows[index]["chunk_id"] if index < sibling_count else None


def create_semantic_chunks(
    *,
    doc_id: str,
    parent_units: list[Phase2ParentUnit],
    normalized_blocks: list[dict[str, Any]],
    chunker: SemanticChunker,
    cfg: PipelineConfig,
    estimate_tokens,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized_blocks_by_id = {block["block_id"]: block for block in normalized_blocks}
    semantic_chunks: list[dict[str, Any]] = []
    counter = 0

    stats = {
        "parent_unit_count": len(parent_units),
        "semantic_split_parent_count": 0,
        "passthrough_parent_count": 0,
        "grouped_list_parent_count": sum(1 for unit in parent_units if unit.unit_kind == "list"),
        "visual_parent_count": sum(1 for unit in parent_units if unit.unit_kind == "visual"),
        "table_parent_count": sum(1 for unit in parent_units if unit.unit_kind == "table"),
    }

    for unit in parent_units:
        child_texts = split_parent_unit_text(
            unit,
            chunker=chunker,
            cfg=cfg,
            estimate_tokens=estimate_tokens,
        )
        if len(child_texts) > 1:
            stats["semantic_split_parent_count"] += 1
        else:
            stats["passthrough_parent_count"] += 1

        offsets = assign_offsets(unit.display_text, child_texts)
        spans = build_block_spans(unit.display_text, unit.source_block_ids, normalized_blocks_by_id)
        unit_rows: list[dict[str, Any]] = []

        for child_text, (char_start, char_end) in zip(child_texts, offsets, strict=False):
            counter += 1
            source_block_ids, source_marker_block_ids, page_start, page_end = block_ids_for_offset(
                spans,
                start=char_start,
                end=char_end,
            )
            row = {
                "chunk_order": counter,
                "chunk_id": f"{doc_id}_semantic_{counter:05d}",
                "doc_id": doc_id,
                "parent_unit_order": unit.unit_order,
                "parent_unit_kind": unit.unit_kind,
                "chunk_type": unit.chunk_type,
                "semantic_hint": unit.semantic_hint,
                "source_semantic_hints": unit.source_semantic_hints,
                "content_modality": (
                    "visual"
                    if unit.chunk_type == "visual"
                    else "table"
                    if unit.chunk_type == "table"
                    else "list"
                    if unit.chunk_type == "list"
                    else "text"
                ),
                "heading_path": unit.heading_path,
                "heading_path_text": " > ".join(unit.heading_path),
                "document_title": unit.heading_path[0] if unit.heading_path else None,
                "section_title": unit.heading_path[-1] if unit.heading_path else None,
                "display_text": child_text,
                "search_text": normalize_search_text(child_text),
                "page_start": page_start if page_start is not None else unit.page_start,
                "page_end": page_end if page_end is not None else unit.page_end,
                "indexable": unit.indexable,
                "source_structural_chunk_ids": unit.source_structural_chunk_ids,
                "source_structural_chunk_orders": unit.source_structural_chunk_orders,
                "source_structural_chunk_count": len(unit.source_structural_chunk_ids),
                "source_block_ids": source_block_ids or unit.source_block_ids,
                "source_marker_block_ids": source_marker_block_ids or unit.source_marker_block_ids,
                "char_start_in_parent": char_start,
                "char_end_in_parent": char_end,
                "parent_char_count": len(unit.display_text),
                "char_count": len(child_text),
                "estimated_token_count": estimate_tokens(child_text),
            }
            row["embedding_text"] = build_embedding_text(
                doc_id=doc_id,
                heading_path=row["heading_path"],
                chunk_type=row["chunk_type"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                display_text=row["display_text"],
            )
            unit_rows.append(row)

        finalize_neighbor_metadata(unit_rows)
        semantic_chunks.extend(unit_rows)

    return semantic_chunks, stats


def load_phase01_inputs(
    cfg: PipelineConfig,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = resolve_phase01_manifest_path(cfg)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase-01 manifest does not exist: {manifest_path}")

    manifest = read_json(manifest_path)
    normalized_blocks_path = resolve_artifact_path(
        manifest["artifacts"]["normalized_blocks"],
        cfg.phase01_root,
    )
    structural_chunks_path = resolve_artifact_path(
        manifest["artifacts"]["structural_chunks"],
        cfg.phase01_root,
    )

    normalized_blocks = read_jsonl(normalized_blocks_path)
    structural_chunks = read_jsonl(structural_chunks_path)
    return manifest, manifest_path, normalized_blocks, structural_chunks


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    env_values = merged_env(load_simple_env(cfg.env_file))
    phase01_manifest, phase01_manifest_path, normalized_blocks, structural_chunks = load_phase01_inputs(cfg)
    doc_id = str(phase01_manifest["doc_id"])

    print(f"[phase02] Starting semantic chunking for {doc_id}", file=sys.stderr, flush=True)
    estimate_tokens = build_token_estimator(cfg.tiktoken_model_name)
    embeddings = build_embeddings(env_values, cfg.tiktoken_model_name)
    chunker = build_semantic_chunker(cfg, embeddings)

    parent_units = build_phase2_parent_units(structural_chunks, cfg, estimate_tokens)
    if cfg.debug:
        print(
            f"[phase02] Built {len(parent_units)} phase-02 parent units from "
            f"{len(structural_chunks)} structural chunks.",
            file=sys.stderr,
            flush=True,
        )

    semantic_chunks, phase02_stats = create_semantic_chunks(
        doc_id=doc_id,
        parent_units=parent_units,
        normalized_blocks=normalized_blocks,
        chunker=chunker,
        cfg=cfg,
        estimate_tokens=estimate_tokens,
    )

    output_root = ensure_dir(cfg.output_root)
    manifests_dir = ensure_dir(output_root / "manifests")
    chunks_dir = ensure_dir(output_root / "semantic_chunks")
    manifest_path = manifests_dir / f"{doc_id}.json"
    semantic_chunks_path = chunks_dir / f"{doc_id}.jsonl"

    if not cfg.force:
        for candidate in (manifest_path, semantic_chunks_path):
            if candidate.exists():
                raise FileExistsError(
                    f"Phase-02 output already exists: {candidate}. Re-run with --force to overwrite."
                )

    print("[phase02] Writing semantic chunks and manifest...", file=sys.stderr, flush=True)
    write_jsonl(semantic_chunks_path, semantic_chunks)

    manifest = {
        "doc_id": doc_id,
        "processed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_phase01_manifest": str(phase01_manifest_path),
        "source_pdf": phase01_manifest.get("source_pdf"),
        "phase01_artifacts": phase01_manifest.get("artifacts", {}),
        "phase02_policy": {
            "inherits_heading_path_from_phase01": True,
            "reclassifies_front_matter_in_phase02": False,
            "indexes_visual_content": True,
            "larger_more_complete_chunks": True,
        },
        "semantic_chunker": {
            "breakpoint_threshold_type": cfg.breakpoint_threshold_type,
            "breakpoint_threshold_amount": cfg.breakpoint_threshold_amount,
            "buffer_size": cfg.buffer_size,
            "sentence_split_regex": SENTENCE_SPLIT_REGEX,
            "min_chunk_tokens": cfg.min_chunk_tokens,
            "target_chunk_tokens": cfg.target_chunk_tokens,
            "max_chunk_tokens": cfg.max_chunk_tokens,
            "list_max_items": cfg.list_max_items,
            "list_target_tokens": cfg.list_target_tokens,
        },
        "embedding": {
            "base_url": env_values.get("OPENAI_BASE_URL"),
            "model": env_values.get("OPENAI_EMBEDDING_MODEL"),
            "tiktoken_model_name": cfg.tiktoken_model_name,
        },
        "artifacts": {
            "semantic_chunks": str(semantic_chunks_path),
        },
        "stats": {
            "normalized_block_count": len(normalized_blocks),
            "structural_chunk_count": len(structural_chunks),
            "semantic_chunk_count": len(semantic_chunks),
            "indexable_chunk_count": sum(1 for row in semantic_chunks if row["indexable"]),
            **phase02_stats,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    cfg = parse_args()
    manifest = run_pipeline(cfg)
    print(
        json.dumps(
            {
                "doc_id": manifest["doc_id"],
                "semantic_chunk_count": manifest["stats"]["semantic_chunk_count"],
                "semantic_chunks": manifest["artifacts"]["semantic_chunks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
