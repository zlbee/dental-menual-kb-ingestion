from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from langchain_openai import OpenAIEmbeddings
from pymilvus import DataType, MilvusClient


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE02_ROOT = REPO_ROOT / "data" / "processed" / "02_semantic_chunking"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "03_vectorization"
DEFAULT_STANDALONE_MILVUS_URI = "http://localhost:19530"
DEFAULT_COLLECTION_PREFIX = "dental_kb_v1"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
PAGE_NULL_SENTINEL = -1
VECTOR_FIELD_NAME = "embedding"
PRIMARY_KEY_FIELD = "chunk_id"
METADATA_FIELD = "metadata"
MAX_ID_LENGTH = 512
MAX_TITLE_LENGTH = 4096
MAX_TEXT_LENGTH = 65535
PHASE03_SCHEMA_VERSION = "1.0"
VECTOR_INDEX_NAME = "embedding_hnsw"
VECTOR_INDEX_TYPE = "HNSW"
VECTOR_METRIC_TYPE = "COSINE"
VECTOR_INDEX_BUILD_PARAMS = {"M": 16, "efConstruction": 200}
SCALAR_INDEX_FIELDS = (
    "doc_id",
    "chunk_type",
    "content_modality",
    "document_title",
    "section_title",
    "page_start",
    "page_end",
    "indexable",
)


@dataclasses.dataclass(slots=True)
class PipelineConfig:
    input_manifest: Path | None
    doc_id: str | None
    phase02_root: Path
    output_root: Path
    env_file: Path | None
    milvus_uri: str | None
    milvus_token: str | None
    milvus_db_name: str | None
    collection_name: str | None
    collection_prefix: str
    embedding_batch_size: int
    embedding_dimensions: int | None
    force: bool
    debug: bool


@dataclasses.dataclass(slots=True)
class PreparedChunk:
    chunk_id: str
    embedding_input_text: str
    insert_row: dict[str, Any]
    enriched_row: dict[str, Any]


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run phase-03 vectorization by embedding phase-02 semantic chunks "
            "and storing them in Milvus."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=None,
        help="Phase-02 manifest path. If omitted, resolve from --doc-id and --phase02-root.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Document id to load from phase-02 manifests when --input-manifest is omitted.",
    )
    parser.add_argument(
        "--phase02-root",
        type=Path,
        default=DEFAULT_PHASE02_ROOT,
        help="Root directory for phase-02 outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for phase-03 outputs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Optional .env file used to hydrate embedding and Milvus settings.",
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=None,
        help="Optional Milvus URI. Defaults to MILVUS_URI or the local Milvus standalone endpoint.",
    )
    parser.add_argument(
        "--milvus-token",
        type=str,
        default=None,
        help="Optional Milvus token. Defaults to MILVUS_TOKEN from the environment.",
    )
    parser.add_argument(
        "--milvus-db-name",
        type=str,
        default=None,
        help="Optional Milvus database name. Defaults to MILVUS_DB_NAME from the environment.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Optional explicit Milvus collection name.",
    )
    parser.add_argument(
        "--collection-prefix",
        type=str,
        default=DEFAULT_COLLECTION_PREFIX,
        help="Collection prefix used when --collection-name is omitted.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Embedding batch size passed to OpenAIEmbeddings.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help="Optional embedding dimension override for text-embedding-3* models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing phase-03 output artifacts for the target doc_id.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit extra phase-03 diagnostics to stderr.",
    )
    parsed = parser.parse_args()

    if parsed.input_manifest is None and parsed.doc_id is None:
        parser.error("Provide either --input-manifest or --doc-id.")
    if parsed.embedding_batch_size <= 0:
        parser.error("--embedding-batch-size must be greater than 0.")
    if parsed.embedding_dimensions is not None and parsed.embedding_dimensions <= 0:
        parser.error("--embedding-dimensions must be greater than 0.")

    return PipelineConfig(
        input_manifest=parsed.input_manifest.expanduser().resolve() if parsed.input_manifest else None,
        doc_id=parsed.doc_id,
        phase02_root=parsed.phase02_root.expanduser().resolve(),
        output_root=parsed.output_root.expanduser().resolve(),
        env_file=parsed.env_file.expanduser().resolve() if parsed.env_file else None,
        milvus_uri=parsed.milvus_uri,
        milvus_token=parsed.milvus_token,
        milvus_db_name=parsed.milvus_db_name,
        collection_name=parsed.collection_name,
        collection_prefix=parsed.collection_prefix,
        embedding_batch_size=parsed.embedding_batch_size,
        embedding_dimensions=parsed.embedding_dimensions,
        force=parsed.force,
        debug=parsed.debug,
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized.lower() or "default"


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


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


def parse_optional_positive_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than 0.")
    return parsed


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


def infer_data_root(phase02_root: Path) -> Path:
    if phase02_root.name == "02_semantic_chunking" and phase02_root.parent.name == "processed":
        return phase02_root.parent.parent
    return REPO_ROOT / "data"


def resolve_phase02_manifest_path(cfg: PipelineConfig) -> Path:
    if cfg.input_manifest:
        return cfg.input_manifest
    assert cfg.doc_id is not None
    return cfg.phase02_root / "manifests" / f"{cfg.doc_id}.json"


def resolve_artifact_path(raw_path: str, phase02_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    data_root = infer_data_root(phase02_root)
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
    attempts.append(phase02_root / "semantic_chunks" / filename)

    for attempt in attempts:
        if attempt.exists():
            return attempt

    raise FileNotFoundError(f"Could not resolve phase-02 artifact path: {raw_path}")


def load_phase02_inputs(
    cfg: PipelineConfig,
) -> tuple[dict[str, Any], Path, Path, list[dict[str, Any]]]:
    manifest_path = resolve_phase02_manifest_path(cfg)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase-02 manifest does not exist: {manifest_path}")

    manifest = read_json(manifest_path)
    semantic_chunks_path = resolve_artifact_path(
        manifest["artifacts"]["semantic_chunks"],
        cfg.phase02_root,
    )
    semantic_chunks = read_jsonl(semantic_chunks_path)
    return manifest, manifest_path, semantic_chunks_path, semantic_chunks


def resolve_embedding_dimensions(cfg: PipelineConfig, env_values: dict[str, str]) -> int | None:
    if cfg.embedding_dimensions is not None:
        return cfg.embedding_dimensions
    return parse_optional_positive_int(
        env_values.get("OPENAI_EMBEDDING_DIMENSIONS"),
        label="OPENAI_EMBEDDING_DIMENSIONS",
    )


def build_embeddings(
    env_values: dict[str, str],
    batch_size: int,
    *,
    dimensions: int | None,
) -> OpenAIEmbeddings:
    required = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL")
    missing = [key for key in required if not env_values.get(key)]
    if missing:
        raise KeyError(
            "Phase-03 vectorization requires these env vars: " + ", ".join(sorted(missing))
        )

    kwargs: dict[str, Any] = {
        "base_url": env_values["OPENAI_BASE_URL"],
        "api_key": env_values["OPENAI_API_KEY"],
        "model": env_values["OPENAI_EMBEDDING_MODEL"],
        "chunk_size": batch_size,
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    return OpenAIEmbeddings(**kwargs)


def looks_like_remote_milvus_uri(uri: str) -> bool:
    if "://" in uri:
        return True
    if re.fullmatch(r"[A-Za-z0-9.-]+:\d+", uri):
        return True
    return False


def looks_like_windows_path(uri: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", uri))


def is_local_milvus_uri(uri: str) -> bool:
    if looks_like_windows_path(uri):
        return True
    if looks_like_remote_milvus_uri(uri):
        return False
    return uri.endswith(".db") or "/" in uri or "\\" in uri or uri.startswith(".")


def resolve_milvus_uri(cfg: PipelineConfig, env_values: dict[str, str]) -> str:
    if cfg.milvus_uri:
        return cfg.milvus_uri
    if env_values.get("MILVUS_URI"):
        return env_values["MILVUS_URI"]
    return DEFAULT_STANDALONE_MILVUS_URI


def sanitize_milvus_uri(uri: str) -> str:
    if is_local_milvus_uri(uri):
        return uri

    parsed = urlsplit(uri if "://" in uri else f"http://{uri}")
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    sanitized = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    return sanitized if "://" in uri else sanitized.removeprefix("http://")


def resolve_collection_name(
    cfg: PipelineConfig,
    env_values: dict[str, str],
    *,
    embedding_dimensions: int | None,
) -> str:
    explicit_name = cfg.collection_name or env_values.get("MILVUS_COLLECTION_NAME")
    if explicit_name:
        return explicit_name
    prefix = env_values.get("MILVUS_COLLECTION_PREFIX") or cfg.collection_prefix
    embedding_model = env_values["OPENAI_EMBEDDING_MODEL"]
    collection_name = f"{slugify(prefix)}_{slugify(embedding_model)}"
    if embedding_dimensions is not None:
        collection_name = f"{collection_name}_dim{embedding_dimensions}"
    return collection_name


def ensure_server_milvus_uri(uri: str) -> None:
    if not is_local_milvus_uri(uri):
        return
    raise RuntimeError(
        "Local Milvus Lite database files are not supported because phase03 now builds "
        "HNSW indexes. Point MILVUS_URI to a Milvus standalone/server endpoint such as "
        "http://localhost:19530 or http://milvus-standalone:19530 in Docker Compose."
    )


def build_milvus_client(
    cfg: PipelineConfig,
    env_values: dict[str, str],
) -> tuple[MilvusClient, str, bool, str | None]:
    uri = resolve_milvus_uri(cfg, env_values)
    local_mode = is_local_milvus_uri(uri)
    token = cfg.milvus_token or env_values.get("MILVUS_TOKEN")
    db_name = cfg.milvus_db_name or env_values.get("MILVUS_DB_NAME")

    if local_mode:
        ensure_server_milvus_uri(uri)

    client_kwargs: dict[str, Any] = {"uri": uri}
    if token:
        client_kwargs["token"] = token
    if db_name:
        client_kwargs["db_name"] = db_name
    client = MilvusClient(**client_kwargs)
    return client, uri, False, db_name


def describe_collection_dim(client: MilvusClient, collection_name: str) -> int | None:
    description = client.describe_collection(collection_name=collection_name)
    fields = description.get("fields", []) if isinstance(description, dict) else []
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name") or field.get("field_name")
        if name != VECTOR_FIELD_NAME:
            continue
        params = field.get("params") or {}
        dim = params.get("dim")
        if dim is None:
            continue
        try:
            return int(dim)
        except (TypeError, ValueError):
            return None
    return None


def build_vector_index_params():
    vector_index_params = MilvusClient.prepare_index_params()
    vector_index_params.add_index(
        field_name=VECTOR_FIELD_NAME,
        index_name=VECTOR_INDEX_NAME,
        index_type=VECTOR_INDEX_TYPE,
        metric_type=VECTOR_METRIC_TYPE,
        params=dict(VECTOR_INDEX_BUILD_PARAMS),
    )
    return vector_index_params


def build_scalar_index_params():
    scalar_index_params = MilvusClient.prepare_index_params()
    for field_name in SCALAR_INDEX_FIELDS:
        scalar_index_params.add_index(
            field_name=field_name,
            index_name=f"{field_name}_idx",
            index_type="INVERTED",
        )
    return scalar_index_params


def describe_vector_indexes(client: MilvusClient, collection_name: str) -> list[dict[str, Any]]:
    descriptions: list[dict[str, Any]] = []
    for index_name in client.list_indexes(collection_name=collection_name):
        description = client.describe_index(collection_name=collection_name, index_name=index_name)
        if isinstance(description, dict):
            raw_descriptions = [description]
        elif isinstance(description, list):
            raw_descriptions = [item for item in description if isinstance(item, dict)]
        else:
            raw_descriptions = []

        for item in raw_descriptions:
            field_name = item.get("field_name") or item.get("fieldName") or item.get("field")
            if field_name == VECTOR_FIELD_NAME:
                descriptions.append(item)
    return descriptions


def is_expected_vector_index(description: dict[str, Any]) -> bool:
    return (
        str(description.get("index_name") or "").strip() == VECTOR_INDEX_NAME
        and str(description.get("index_type") or "").upper() == VECTOR_INDEX_TYPE
        and str(description.get("metric_type") or "").upper() == VECTOR_METRIC_TYPE
    )


def ensure_vector_index(
    client: MilvusClient,
    *,
    collection_name: str,
    debug: bool,
) -> bool:
    descriptions = describe_vector_indexes(client, collection_name)
    has_expected_index = any(is_expected_vector_index(description) for description in descriptions)
    stale_index_names = sorted(
        {
            str(description.get("index_name")).strip()
            for description in descriptions
            if description.get("index_name") and not is_expected_vector_index(description)
        }
    )

    if not stale_index_names and has_expected_index:
        return False

    with suppress(Exception):
        client.release_collection(collection_name=collection_name)

    for index_name in stale_index_names:
        client.drop_index(collection_name=collection_name, index_name=index_name)
        if debug:
            print(
                f"[phase03] Dropped stale vector index {index_name} on {collection_name}.",
                file=sys.stderr,
                flush=True,
            )

    if not has_expected_index:
        client.create_index(collection_name=collection_name, index_params=build_vector_index_params())
        if debug:
            print(
                f"[phase03] Ensured {VECTOR_INDEX_TYPE} vector index {VECTOR_INDEX_NAME} on {collection_name}.",
                file=sys.stderr,
                flush=True,
            )

    return True


def build_collection_schema(vector_dim: int):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name=PRIMARY_KEY_FIELD, datatype=DataType.VARCHAR, is_primary=True, max_length=MAX_ID_LENGTH)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=MAX_ID_LENGTH)
    schema.add_field(field_name="chunk_order", datatype=DataType.INT64)
    schema.add_field(field_name="parent_unit_order", datatype=DataType.INT64)
    schema.add_field(field_name="parent_unit_kind", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="semantic_hint", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="content_modality", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="heading_path_text", datatype=DataType.VARCHAR, max_length=MAX_TITLE_LENGTH)
    schema.add_field(field_name="document_title", datatype=DataType.VARCHAR, max_length=MAX_TITLE_LENGTH)
    schema.add_field(field_name="section_title", datatype=DataType.VARCHAR, max_length=MAX_TITLE_LENGTH)
    schema.add_field(field_name="page_start", datatype=DataType.INT64)
    schema.add_field(field_name="page_end", datatype=DataType.INT64)
    schema.add_field(field_name="indexable", datatype=DataType.BOOL)
    schema.add_field(field_name="estimated_token_count", datatype=DataType.INT64)
    schema.add_field(field_name="sibling_index", datatype=DataType.INT64)
    schema.add_field(field_name="sibling_count", datatype=DataType.INT64)
    schema.add_field(field_name="prev_chunk_id", datatype=DataType.VARCHAR, max_length=MAX_ID_LENGTH)
    schema.add_field(field_name="next_chunk_id", datatype=DataType.VARCHAR, max_length=MAX_ID_LENGTH)
    schema.add_field(field_name="display_text", datatype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH)
    schema.add_field(field_name="embedding_text", datatype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH)
    schema.add_field(field_name=METADATA_FIELD, datatype=DataType.JSON)
    schema.add_field(field_name=VECTOR_FIELD_NAME, datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

    return schema, build_vector_index_params(), build_scalar_index_params()


def ensure_collection(
    client: MilvusClient,
    *,
    collection_name: str,
    vector_dim: int,
    debug: bool,
) -> bool:
    created = False
    if client.has_collection(collection_name=collection_name):
        existing_dim = describe_collection_dim(client, collection_name)
        if existing_dim is not None and existing_dim != vector_dim:
            raise RuntimeError(
                f"Milvus collection {collection_name} already exists with vector dimension "
                f"{existing_dim}, but phase03 produced dimension {vector_dim}."
            )
        ensure_vector_index(
            client,
            collection_name=collection_name,
            debug=debug,
        )
    else:
        schema, vector_index_params, scalar_index_params = build_collection_schema(vector_dim)
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=vector_index_params,
        )
        try:
            client.create_index(
                collection_name=collection_name,
                index_params=scalar_index_params,
            )
        except Exception as exc:
            if debug:
                print(
                    f"[phase03] Scalar index creation skipped: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        created = True

    client.load_collection(collection_name=collection_name)
    return created


def safe_string(value: Any, *, max_length: int, field_name: str, chunk_id: str) -> str:
    text = "" if value is None else str(value)
    text_size = len(text.encode("utf-8"))
    if text_size > max_length:
        raise ValueError(
            f"Field {field_name} for chunk {chunk_id} exceeds Milvus VARCHAR max_length "
            f"({text_size} > {max_length} bytes)."
        )
    return text


def safe_int(value: Any) -> int:
    if value is None:
        return PAGE_NULL_SENTINEL
    return int(value)


def safe_metadata(value: dict[str, Any], *, chunk_id: str) -> dict[str, Any]:
    payload = json.dumps(value, ensure_ascii=False)
    payload_size = len(payload.encode("utf-8"))
    if payload_size > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Metadata for chunk {chunk_id} exceeds Milvus JSON size guidance "
            f"({payload_size} > {MAX_TEXT_LENGTH} bytes)."
        )
    return value


def build_chunk_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "heading_path": list(row.get("heading_path") or []),
        "source_semantic_hints": list(row.get("source_semantic_hints") or []),
        "source_structural_chunk_ids": list(row.get("source_structural_chunk_ids") or []),
        "source_structural_chunk_orders": list(row.get("source_structural_chunk_orders") or []),
        "source_structural_chunk_count": row.get("source_structural_chunk_count"),
        "source_block_ids": list(row.get("source_block_ids") or []),
        "source_marker_block_ids": list(row.get("source_marker_block_ids") or []),
        "char_start_in_parent": row.get("char_start_in_parent"),
        "char_end_in_parent": row.get("char_end_in_parent"),
        "parent_char_count": row.get("parent_char_count"),
        "char_count": row.get("char_count"),
        "raw_page_start": row.get("page_start"),
        "raw_page_end": row.get("page_end"),
    }


def prepare_chunks(
    semantic_chunks: list[dict[str, Any]],
    *,
    collection_name: str,
    embedding_model: str,
) -> tuple[list[dict[str, Any]], list[PreparedChunk]]:
    enriched_rows: list[dict[str, Any]] = []
    prepared_chunks: list[PreparedChunk] = []

    for row in semantic_chunks:
        chunk_id = str(row["chunk_id"])
        metadata = safe_metadata(build_chunk_metadata(row), chunk_id=chunk_id)
        storage_page_start = safe_int(row.get("page_start"))
        storage_page_end = safe_int(row.get("page_end"))

        insert_row = {
            PRIMARY_KEY_FIELD: safe_string(chunk_id, max_length=MAX_ID_LENGTH, field_name=PRIMARY_KEY_FIELD, chunk_id=chunk_id),
            "doc_id": safe_string(row.get("doc_id"), max_length=MAX_ID_LENGTH, field_name="doc_id", chunk_id=chunk_id),
            "chunk_order": int(row.get("chunk_order") or 0),
            "parent_unit_order": int(row.get("parent_unit_order") or 0),
            "parent_unit_kind": safe_string(row.get("parent_unit_kind"), max_length=64, field_name="parent_unit_kind", chunk_id=chunk_id),
            "chunk_type": safe_string(row.get("chunk_type"), max_length=64, field_name="chunk_type", chunk_id=chunk_id),
            "semantic_hint": safe_string(row.get("semantic_hint"), max_length=64, field_name="semantic_hint", chunk_id=chunk_id),
            "content_modality": safe_string(row.get("content_modality"), max_length=64, field_name="content_modality", chunk_id=chunk_id),
            "heading_path_text": safe_string(row.get("heading_path_text"), max_length=MAX_TITLE_LENGTH, field_name="heading_path_text", chunk_id=chunk_id),
            "document_title": safe_string(row.get("document_title"), max_length=MAX_TITLE_LENGTH, field_name="document_title", chunk_id=chunk_id),
            "section_title": safe_string(row.get("section_title"), max_length=MAX_TITLE_LENGTH, field_name="section_title", chunk_id=chunk_id),
            "page_start": storage_page_start,
            "page_end": storage_page_end,
            "indexable": bool(row.get("indexable")),
            "estimated_token_count": int(row.get("estimated_token_count") or 0),
            "sibling_index": int(row.get("sibling_index") or 0),
            "sibling_count": int(row.get("sibling_count") or 0),
            "prev_chunk_id": safe_string(row.get("prev_chunk_id"), max_length=MAX_ID_LENGTH, field_name="prev_chunk_id", chunk_id=chunk_id),
            "next_chunk_id": safe_string(row.get("next_chunk_id"), max_length=MAX_ID_LENGTH, field_name="next_chunk_id", chunk_id=chunk_id),
            "display_text": safe_string(row.get("display_text"), max_length=MAX_TEXT_LENGTH, field_name="display_text", chunk_id=chunk_id),
            "embedding_text": safe_string(row.get("embedding_text"), max_length=MAX_TEXT_LENGTH, field_name="embedding_text", chunk_id=chunk_id),
            METADATA_FIELD: metadata,
        }

        enriched_row = dict(row)
        enriched_row["phase03"] = {
            "schema_version": PHASE03_SCHEMA_VERSION,
            "collection_name": collection_name,
            "primary_key": chunk_id,
            "embedding_model": embedding_model,
            "storage_page_start": storage_page_start,
            "storage_page_end": storage_page_end,
            "metadata": metadata,
        }
        enriched_rows.append(enriched_row)

        if bool(row.get("indexable")):
            prepared_chunks.append(
                PreparedChunk(
                    chunk_id=chunk_id,
                    embedding_input_text=str(row["embedding_text"]),
                    insert_row=insert_row,
                    enriched_row=enriched_row,
                )
            )

    return enriched_rows, prepared_chunks


def embed_prepared_chunks(
    prepared_chunks: list[PreparedChunk],
    embeddings: OpenAIEmbeddings,
    *,
    batch_size: int,
    debug: bool,
) -> int | None:
    vector_dim: int | None = None

    for batch_number, batch in enumerate(batched(prepared_chunks, batch_size), start=1):
        texts = [chunk.embedding_input_text for chunk in batch]
        vectors = embeddings.embed_documents(texts)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding batch {batch_number} returned {len(vectors)} vectors for "
                f"{len(batch)} inputs."
            )

        for chunk, vector in zip(batch, vectors, strict=False):
            if vector_dim is None:
                vector_dim = len(vector)
            elif len(vector) != vector_dim:
                raise RuntimeError(
                    f"Chunk {chunk.chunk_id} produced vector dimension {len(vector)}; "
                    f"expected {vector_dim}."
                )
            chunk.insert_row[VECTOR_FIELD_NAME] = vector

        if debug:
            print(
                f"[phase03] Embedded batch {batch_number} with {len(batch)} chunks.",
                file=sys.stderr,
                flush=True,
            )

    return vector_dim


def delete_existing_doc_rows(
    client: MilvusClient,
    *,
    collection_name: str,
    doc_id: str,
) -> int:
    filter_expr = f'doc_id == {json.dumps(doc_id)}'
    existing = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=[PRIMARY_KEY_FIELD],
    )
    existing_count = len(existing)
    if existing_count:
        client.delete(
            collection_name=collection_name,
            filter=filter_expr,
        )
    return existing_count


def insert_chunks(
    client: MilvusClient,
    *,
    collection_name: str,
    prepared_chunks: list[PreparedChunk],
    batch_size: int,
    debug: bool,
) -> int:
    inserted = 0
    for batch_number, batch in enumerate(batched(prepared_chunks, batch_size), start=1):
        payload = [chunk.insert_row for chunk in batch]
        client.insert(collection_name=collection_name, data=payload)
        inserted += len(payload)
        if debug:
            print(
                f"[phase03] Inserted batch {batch_number} with {len(payload)} chunks.",
                file=sys.stderr,
                flush=True,
            )
    return inserted


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    env_values = merged_env(load_simple_env(cfg.env_file))
    phase02_manifest, phase02_manifest_path, semantic_chunks_path, semantic_chunks = load_phase02_inputs(cfg)
    doc_id = str(phase02_manifest["doc_id"])
    embedding_dimensions = resolve_embedding_dimensions(cfg, env_values)

    print(f"[phase03] Starting vectorization for {doc_id}", file=sys.stderr, flush=True)
    embeddings = build_embeddings(
        env_values,
        cfg.embedding_batch_size,
        dimensions=embedding_dimensions,
    )
    collection_name = resolve_collection_name(
        cfg,
        env_values,
        embedding_dimensions=embedding_dimensions,
    )
    embedding_model = env_values["OPENAI_EMBEDDING_MODEL"]

    output_root = ensure_dir(cfg.output_root)
    manifests_dir = ensure_dir(output_root / "manifests")
    enriched_dir = ensure_dir(output_root / "enriched_chunks")
    manifest_path = manifests_dir / f"{doc_id}.json"
    enriched_chunks_path = enriched_dir / f"{doc_id}.jsonl"

    if not cfg.force:
        for candidate in (manifest_path, enriched_chunks_path):
            if candidate.exists():
                raise FileExistsError(
                    f"Phase-03 output already exists: {candidate}. Re-run with --force to overwrite."
                )

    enriched_rows, prepared_chunks = prepare_chunks(
        semantic_chunks,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    inserted_chunk_count = 0
    deleted_chunk_count = 0
    created_collection = False
    vector_dim: int | None = None
    milvus_client: MilvusClient | None = None
    milvus_uri: str | None = None
    local_milvus = False
    milvus_db_name: str | None = None

    try:
        if prepared_chunks:
            print(
                f"[phase03] Embedding {len(prepared_chunks)} indexable semantic chunks...",
                file=sys.stderr,
                flush=True,
            )
            vector_dim = embed_prepared_chunks(
                prepared_chunks,
                embeddings,
                batch_size=cfg.embedding_batch_size,
                debug=cfg.debug,
            )
            if vector_dim is None:
                raise RuntimeError("Phase-03 did not produce any embedding vectors.")

            milvus_client, milvus_uri, local_milvus, milvus_db_name = build_milvus_client(cfg, env_values)
            created_collection = ensure_collection(
                milvus_client,
                collection_name=collection_name,
                vector_dim=vector_dim,
                debug=cfg.debug,
            )
            deleted_chunk_count = delete_existing_doc_rows(
                milvus_client,
                collection_name=collection_name,
                doc_id=doc_id,
            )
            inserted_chunk_count = insert_chunks(
                milvus_client,
                collection_name=collection_name,
                prepared_chunks=prepared_chunks,
                batch_size=cfg.embedding_batch_size,
                debug=cfg.debug,
            )
        else:
            milvus_uri = resolve_milvus_uri(cfg, env_values)
            local_milvus = is_local_milvus_uri(milvus_uri)
            if local_milvus:
                ensure_server_milvus_uri(milvus_uri)
            milvus_db_name = None if local_milvus else cfg.milvus_db_name or env_values.get("MILVUS_DB_NAME")
    finally:
        if milvus_client is not None:
            try:
                milvus_client.close()
            except Exception:
                pass

    print("[phase03] Writing enriched chunks and manifest...", file=sys.stderr, flush=True)
    write_jsonl(enriched_chunks_path, enriched_rows)

    manifest = {
        "doc_id": doc_id,
        "processed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_phase02_manifest": str(phase02_manifest_path),
        "source_pdf": phase02_manifest.get("source_pdf"),
        "source_phase01_manifest": phase02_manifest.get("source_phase01_manifest"),
        "phase02_artifacts": phase02_manifest.get("artifacts", {}),
        "phase03_policy": {
            "schema_version": PHASE03_SCHEMA_VERSION,
            "vector_input_from_phase02_embedding_text": True,
            "inherits_phase02_indexable_without_extra_filtering": True,
            "milvus_primary_key": PRIMARY_KEY_FIELD,
            "milvus_doc_filter_field": "doc_id",
            "page_null_sentinel": PAGE_NULL_SENTINEL,
        },
        "embedding": {
            "base_url": env_values.get("OPENAI_BASE_URL"),
            "model": embedding_model,
            "batch_size": cfg.embedding_batch_size,
            "requested_dimensions": embedding_dimensions,
            "dimension": vector_dim,
        },
        "milvus": {
            "uri": sanitize_milvus_uri(milvus_uri or resolve_milvus_uri(cfg, env_values)),
            "mode": "lite" if local_milvus else "server",
            "db_name": milvus_db_name,
            "collection_name": collection_name,
            "vector_field": VECTOR_FIELD_NAME,
            "vector_index_name": VECTOR_INDEX_NAME,
            "vector_metric_type": VECTOR_METRIC_TYPE,
            "vector_index_type_requested": VECTOR_INDEX_TYPE,
            "vector_index_build_params": dict(VECTOR_INDEX_BUILD_PARAMS),
            "created_collection": created_collection,
        },
        "artifacts": {
            "source_semantic_chunks": str(semantic_chunks_path),
            "enriched_chunks": str(enriched_chunks_path),
        },
        "stats": {
            "semantic_chunk_count": len(semantic_chunks),
            "indexable_chunk_count": sum(1 for row in semantic_chunks if row["indexable"]),
            "inserted_chunk_count": inserted_chunk_count,
            "deleted_existing_chunk_count": deleted_chunk_count,
            "skipped_non_indexable_chunk_count": sum(1 for row in semantic_chunks if not row["indexable"]),
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
                "inserted_chunk_count": manifest["stats"]["inserted_chunk_count"],
                "collection_name": manifest["milvus"]["collection_name"],
                "enriched_chunks": manifest["artifacts"]["enriched_chunks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
