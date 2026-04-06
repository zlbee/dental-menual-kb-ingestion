from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
import re
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, ConfigDict, Field
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException
from rank_bm25 import BM25Okapi
from starlette.concurrency import run_in_threadpool


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE03_ROOT = REPO_ROOT / "data" / "processed" / "03_vectorization"
DEFAULT_STANDALONE_MILVUS_URI = "http://localhost:19530"
DEFAULT_COLLECTION_PREFIX = "dental_kb_v1"
VECTOR_FIELD_NAME = "embedding"
PRIMARY_KEY_FIELD = "chunk_id"
METADATA_FIELD = "metadata"
PAGE_NULL_SENTINEL = -1
VECTOR_SEARCH_METRIC_TYPE = "COSINE"
VECTOR_SEARCH_PARAMS = {"ef": 128}
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_DENSE_TOP_K = 60
DEFAULT_LEXICAL_TOP_K = 60
DEFAULT_HEADING_TOP_K = 40
DEFAULT_CANDIDATE_POOL_SIZE = 80
DEFAULT_FINAL_TOP_K = 8
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 0.50
DEFAULT_BODY_WEIGHT = 0.35
DEFAULT_HEADING_WEIGHT = 0.15
DEFAULT_RERANK_FUSION_BOOST = 0.15
QUERY_ITERATOR_BATCH_SIZE = 1000
DEFAULT_STARTUP_RELOAD_MAX_ATTEMPTS = 8
DEFAULT_STARTUP_RELOAD_RETRY_DELAY_SECONDS = 5.0
MAX_TOP_K = 50
MAX_RECALL_K = 500
MAX_CANDIDATE_POOL = 1000
MILVUS_OUTPUT_FIELDS = [
    PRIMARY_KEY_FIELD,
    "doc_id",
    "chunk_order",
    "chunk_type",
    "content_modality",
    "document_title",
    "section_title",
    "page_start",
    "page_end",
    "prev_chunk_id",
    "next_chunk_id",
    "display_text",
    "embedding_text",
    METADATA_FIELD,
]
SEARCH_OUTPUT_FIELDS = [PRIMARY_KEY_FIELD]
VECTOR_GET_OUTPUT_FIELDS = [PRIMARY_KEY_FIELD, VECTOR_FIELD_NAME]
TRANSIENT_MILVUS_ERROR_MARKERS = (
    "channel distribution is not serviceable",
    "channel not available",
)


logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True, frozen=True)
class RuntimeConfig:
    env_file: Path | None
    phase03_root: Path
    milvus_uri: str | None
    milvus_token: str | None
    milvus_db_name: str | None
    collection_name: str | None
    collection_prefix: str
    host: str
    port: int
    dense_top_k: int
    lexical_top_k: int
    heading_top_k: int
    candidate_pool_size: int
    final_top_k: int
    rrf_k: int
    dense_weight: float
    body_weight: float
    heading_weight: float
    rerank_fusion_boost: float
    debug: bool


@dataclasses.dataclass(slots=True, frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_order: int
    chunk_type: str
    content_modality: str
    document_title: str | None
    section_title: str | None
    page_start: int | None
    page_end: int | None
    prev_chunk_id: str | None
    next_chunk_id: str | None
    display_text: str
    embedding_text: str
    metadata: dict[str, Any]
    heading_path: tuple[str, ...]
    heading_path_text: str


@dataclasses.dataclass(slots=True, frozen=True)
class RetrievalIndex:
    rows: tuple[ChunkRecord, ...]
    row_by_chunk_id: dict[str, ChunkRecord]
    body_bm25: BM25Okapi | None
    heading_bm25: BM25Okapi | None
    body_placeholder_rows: frozenset[int]
    heading_placeholder_rows: frozenset[int]
    loaded_at_utc: str


@dataclasses.dataclass(slots=True, frozen=True)
class LaneHit:
    chunk_id: str
    rank: int
    raw_score: float | None = None
    distance: float | None = None


@dataclasses.dataclass(slots=True)
class CandidateScore:
    chunk_id: str
    fused_rrf: float = 0.0
    dense_rrf: float = 0.0
    body_rrf: float = 0.0
    heading_rrf: float = 0.0
    dense_rank: int | None = None
    body_rank: int | None = None
    heading_rank: int | None = None
    dense_raw_score: float | None = None
    dense_distance: float | None = None
    body_bm25: float | None = None
    heading_bm25: float | None = None


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_ids: list[str] | None = None
    chunk_types: list[str] | None = None
    content_modalities: list[str] | None = None
    document_titles: list[str] | None = None
    section_titles: list[str] | None = None
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    dense_top_k: int | None = Field(default=None, ge=1, le=MAX_RECALL_K)
    lexical_top_k: int | None = Field(default=None, ge=1, le=MAX_RECALL_K)
    heading_top_k: int | None = Field(default=None, ge=1, le=MAX_RECALL_K)
    candidate_pool_size: int | None = Field(default=None, ge=1, le=MAX_CANDIDATE_POOL)
    include_neighbors: bool = False
    max_neighbors_per_side: int = Field(default=1, ge=0, le=3)
    filters: RetrievalFilters | None = None


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run the phase-04 online RAG service with Milvus-backed dense recall, "
            "BM25 lexical recall, and embedding reranking."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Optional .env file used to hydrate OpenAI and Milvus settings.",
    )
    parser.add_argument(
        "--phase03-root",
        type=Path,
        default=DEFAULT_PHASE03_ROOT,
        help="Root directory for phase-03 outputs.",
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
        help="Fallback collection prefix when MILVUS_COLLECTION_NAME is unset.",
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Bind host for the HTTP service.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port for the HTTP service.")
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=DEFAULT_DENSE_TOP_K,
        help="Default top-k used for first-stage dense recall.",
    )
    parser.add_argument(
        "--lexical-top-k",
        type=int,
        default=DEFAULT_LEXICAL_TOP_K,
        help="Default top-k used for BM25 recall over embedding_text.",
    )
    parser.add_argument(
        "--heading-top-k",
        type=int,
        default=DEFAULT_HEADING_TOP_K,
        help="Default top-k used for BM25 recall over metadata.heading_path.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help="Default candidate pool size kept after weighted RRF fusion.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=DEFAULT_FINAL_TOP_K,
        help="Default final top-k returned after reranking.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="Reciprocal rank fusion denominator constant.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help="Weighted RRF contribution for dense recall.",
    )
    parser.add_argument(
        "--body-weight",
        type=float,
        default=DEFAULT_BODY_WEIGHT,
        help="Weighted RRF contribution for BM25 over embedding_text.",
    )
    parser.add_argument(
        "--heading-weight",
        type=float,
        default=DEFAULT_HEADING_WEIGHT,
        help="Weighted RRF contribution for BM25 over metadata.heading_path.",
    )
    parser.add_argument(
        "--rerank-fusion-boost",
        type=float,
        default=DEFAULT_RERANK_FUSION_BOOST,
        help="Small fusion prior added on top of the exact rerank cosine score.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging and debug-friendly responses.",
    )
    args = parser.parse_args()

    return RuntimeConfig(
        env_file=args.env_file,
        phase03_root=args.phase03_root,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
        milvus_db_name=args.milvus_db_name,
        collection_name=args.collection_name,
        collection_prefix=args.collection_prefix,
        host=args.host,
        port=args.port,
        dense_top_k=args.dense_top_k,
        lexical_top_k=args.lexical_top_k,
        heading_top_k=args.heading_top_k,
        candidate_pool_size=args.candidate_pool_size,
        final_top_k=args.final_top_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        body_weight=args.body_weight,
        heading_weight=args.heading_weight,
        rerank_fusion_boost=args.rerank_fusion_boost,
        debug=bool(args.debug),
    )


def default_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        env_file=REPO_ROOT / ".env",
        phase03_root=DEFAULT_PHASE03_ROOT,
        milvus_uri=None,
        milvus_token=None,
        milvus_db_name=None,
        collection_name=None,
        collection_prefix=DEFAULT_COLLECTION_PREFIX,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        dense_top_k=DEFAULT_DENSE_TOP_K,
        lexical_top_k=DEFAULT_LEXICAL_TOP_K,
        heading_top_k=DEFAULT_HEADING_TOP_K,
        candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
        final_top_k=DEFAULT_FINAL_TOP_K,
        rrf_k=DEFAULT_RRF_K,
        dense_weight=DEFAULT_DENSE_WEIGHT,
        body_weight=DEFAULT_BODY_WEIGHT,
        heading_weight=DEFAULT_HEADING_WEIGHT,
        rerank_fusion_boost=DEFAULT_RERANK_FUSION_BOOST,
        debug=False,
    )


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


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized.lower() or "default"


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


def resolve_milvus_uri(cfg: RuntimeConfig, env_values: dict[str, str]) -> str:
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


def resolve_collection_name(cfg: RuntimeConfig, env_values: dict[str, str]) -> str:
    explicit_name = cfg.collection_name or env_values.get("MILVUS_COLLECTION_NAME")
    if explicit_name:
        return explicit_name
    prefix = env_values.get("MILVUS_COLLECTION_PREFIX") or cfg.collection_prefix
    embedding_model = env_values["OPENAI_EMBEDDING_MODEL"]
    return f"{slugify(prefix)}_{slugify(embedding_model)}"


def ensure_server_milvus_uri(uri: str) -> None:
    if not is_local_milvus_uri(uri):
        return
    raise RuntimeError(
        "Local Milvus Lite database files are not supported because phase03 now builds "
        "HNSW indexes. Point MILVUS_URI to a Milvus standalone/server endpoint such as "
        "http://localhost:19530 or http://milvus-standalone:19530 in Docker Compose."
    )


def build_embeddings(env_values: dict[str, str], *, model_env_name: str) -> OpenAIEmbeddings:
    required = ("OPENAI_BASE_URL", "OPENAI_API_KEY", model_env_name)
    missing = [name for name in required if not env_values.get(name)]
    if missing:
        raise RuntimeError(
            "Phase-04 online RAG service requires these env vars: " + ", ".join(sorted(missing))
        )

    return OpenAIEmbeddings(
        base_url=env_values["OPENAI_BASE_URL"],
        api_key=env_values["OPENAI_API_KEY"],
        model=env_values[model_env_name],
        max_retries=3,
    )


def build_milvus_client(
    cfg: RuntimeConfig,
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

    return MilvusClient(**client_kwargs), uri, False, db_name


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


def normalize_result_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict") and callable(item.to_dict):
        item = item.to_dict()
    if not isinstance(item, dict):
        raise TypeError(f"Unexpected Milvus result row type: {type(item)!r}")
    return item


def is_transient_milvus_query_error(exc: Exception) -> bool:
    if not isinstance(exc, MilvusException):
        return False
    if getattr(exc, "code", None) != 503:
        return False

    message = str(exc).lower()
    return any(marker in message for marker in TRANSIENT_MILVUS_ERROR_MARKERS)


def empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_page(value: Any) -> int | None:
    if value in (None, "", PAGE_NULL_SENTINEL):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return None if page == PAGE_NULL_SENTINEL else page


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"[0-9A-Za-z]+|[\u4e00-\u9fff]+", text.lower())
    return [token for token in tokens if token]


def heading_path_text_from_metadata(metadata: dict[str, Any], fallback: str | None = None) -> str:
    raw_heading_path = metadata.get("heading_path")
    if isinstance(raw_heading_path, list):
        heading_path = [str(item).strip() for item in raw_heading_path if str(item).strip()]
        if heading_path:
            return " / ".join(heading_path)
    return empty_to_none(fallback) or ""


def build_chunk_record(raw_row: dict[str, Any]) -> ChunkRecord:
    metadata = raw_row.get(METADATA_FIELD) if isinstance(raw_row.get(METADATA_FIELD), dict) else {}
    heading_path_text = heading_path_text_from_metadata(
        metadata,
        fallback=raw_row.get("section_title") or raw_row.get("document_title"),
    )
    heading_path = tuple(part.strip() for part in heading_path_text.split(" / ") if part.strip())

    return ChunkRecord(
        chunk_id=str(raw_row[PRIMARY_KEY_FIELD]),
        doc_id=str(raw_row.get("doc_id") or ""),
        chunk_order=int(raw_row.get("chunk_order") or 0),
        chunk_type=str(raw_row.get("chunk_type") or ""),
        content_modality=str(raw_row.get("content_modality") or ""),
        document_title=empty_to_none(raw_row.get("document_title")),
        section_title=empty_to_none(raw_row.get("section_title")),
        page_start=parse_page(raw_row.get("page_start")),
        page_end=parse_page(raw_row.get("page_end")),
        prev_chunk_id=empty_to_none(raw_row.get("prev_chunk_id")),
        next_chunk_id=empty_to_none(raw_row.get("next_chunk_id")),
        display_text=str(raw_row.get("display_text") or ""),
        embedding_text=str(raw_row.get("embedding_text") or ""),
        metadata=metadata,
        heading_path=heading_path,
        heading_path_text=heading_path_text,
    )


def build_bm25_index(texts: Sequence[str]) -> tuple[BM25Okapi | None, frozenset[int]]:
    tokenized_rows: list[list[str]] = []
    placeholder_rows: set[int] = set()
    has_real_tokens = False

    for index, text in enumerate(texts):
        tokens = tokenize(text)
        if tokens:
            has_real_tokens = True
            tokenized_rows.append(tokens)
            continue
        placeholder_rows.add(index)
        tokenized_rows.append(["__empty__"])

    if not has_real_tokens:
        return None, frozenset(placeholder_rows)
    return BM25Okapi(tokenized_rows), frozenset(placeholder_rows)


def normalize_filter_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [item.strip() for item in values if item and item.strip()]


def build_filter_predicate(filters: RetrievalFilters | None) -> Callable[[ChunkRecord], bool]:
    if filters is None:
        return lambda _: True

    doc_ids = set(normalize_filter_values(filters.doc_ids))
    chunk_types = set(normalize_filter_values(filters.chunk_types))
    content_modalities = set(normalize_filter_values(filters.content_modalities))
    document_titles = set(normalize_filter_values(filters.document_titles))
    section_titles = set(normalize_filter_values(filters.section_titles))
    page_from = filters.page_from
    page_to = filters.page_to

    def predicate(row: ChunkRecord) -> bool:
        if doc_ids and row.doc_id not in doc_ids:
            return False
        if chunk_types and row.chunk_type not in chunk_types:
            return False
        if content_modalities and row.content_modality not in content_modalities:
            return False
        if document_titles and (row.document_title or "") not in document_titles:
            return False
        if section_titles and (row.section_title or "") not in section_titles:
            return False
        if page_from is not None:
            row_end = row.page_end if row.page_end is not None else row.page_start
            if row_end is None or row_end < page_from:
                return False
        if page_to is not None:
            row_start = row.page_start if row.page_start is not None else row.page_end
            if row_start is None or row_start > page_to:
                return False
        return True

    return predicate


def build_milvus_filter(filters: RetrievalFilters | None) -> str:
    clauses = ["indexable == true"]
    if filters is None:
        return " and ".join(clauses)

    doc_ids = normalize_filter_values(filters.doc_ids)
    chunk_types = normalize_filter_values(filters.chunk_types)
    content_modalities = normalize_filter_values(filters.content_modalities)
    document_titles = normalize_filter_values(filters.document_titles)
    section_titles = normalize_filter_values(filters.section_titles)

    if doc_ids:
        clauses.append(f"doc_id in {json.dumps(sorted(doc_ids), ensure_ascii=False)}")
    if chunk_types:
        clauses.append(f"chunk_type in {json.dumps(sorted(chunk_types), ensure_ascii=False)}")
    if content_modalities:
        clauses.append(
            f"content_modality in {json.dumps(sorted(content_modalities), ensure_ascii=False)}"
        )
    if document_titles:
        clauses.append(
            f"document_title in {json.dumps(sorted(document_titles), ensure_ascii=False)}"
        )
    if section_titles:
        clauses.append(
            f"section_title in {json.dumps(sorted(section_titles), ensure_ascii=False)}"
        )
    if filters.page_from is not None:
        clauses.append(f"page_end >= {int(filters.page_from)}")
    if filters.page_to is not None:
        clauses.append(f"page_start <= {int(filters.page_to)}")
    return " and ".join(clauses)


def top_k_bm25_hits(
    bm25: BM25Okapi | None,
    placeholder_rows: frozenset[int],
    rows: Sequence[ChunkRecord],
    *,
    query_tokens: list[str],
    top_k: int,
    predicate: Callable[[ChunkRecord], bool],
) -> list[LaneHit]:
    if bm25 is None or not query_tokens or top_k <= 0:
        return []

    raw_scores = np.asarray(bm25.get_scores(query_tokens), dtype=float)
    scored_indices: list[tuple[int, float]] = []
    for index, score in enumerate(raw_scores):
        if index in placeholder_rows or score <= 0.0:
            continue
        row = rows[index]
        if not predicate(row):
            continue
        scored_indices.append((index, float(score)))

    scored_indices.sort(key=lambda item: item[1], reverse=True)
    return [
        LaneHit(chunk_id=rows[index].chunk_id, rank=rank, raw_score=score)
        for rank, (index, score) in enumerate(scored_indices[:top_k], start=1)
    ]


def ensure_cosine_vector(vector: Sequence[float], *, expected_dim: int | None = None) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"Expected a one-dimensional vector, got shape {array.shape!r}.")
    if expected_dim is not None and array.shape[0] != expected_dim:
        raise ValueError(
            f"Vector dimension mismatch: expected {expected_dim}, got {array.shape[0]}."
        )
    return array


def cosine_similarity(query_vector: np.ndarray, candidate_vector: Sequence[float]) -> float:
    candidate = ensure_cosine_vector(candidate_vector, expected_dim=int(query_vector.shape[0]))
    query_norm = float(np.linalg.norm(query_vector))
    candidate_norm = float(np.linalg.norm(candidate))
    if query_norm == 0.0 or candidate_norm == 0.0:
        return 0.0
    return float(np.dot(query_vector, candidate) / (query_norm * candidate_norm))


class OnlineRAGService:
    def __init__(self, cfg: RuntimeConfig, env_values: dict[str, str]) -> None:
        self.cfg = cfg
        self.env_values = env_values
        self.collection_name = resolve_collection_name(cfg, env_values)
        self.client, self.milvus_uri, self.local_milvus, self.milvus_db_name = build_milvus_client(
            cfg, env_values
        )
        self.vector_dim = describe_collection_dim(self.client, self.collection_name)
        self.recall_model = env_values["OPENAI_RECALL_MODEL"]
        self.rerank_model = env_values["OPENAI_EMBEDDING_MODEL"]
        self.recall_embeddings = build_embeddings(env_values, model_env_name="OPENAI_RECALL_MODEL")
        if self.recall_model == self.rerank_model:
            self.rerank_embeddings = self.recall_embeddings
        else:
            self.rerank_embeddings = build_embeddings(
                env_values, model_env_name="OPENAI_EMBEDDING_MODEL"
            )
        self._state_lock = threading.RLock()
        self._milvus_lock = threading.RLock()
        self._index: RetrievalIndex | None = None

    def close(self) -> None:
        with suppress(Exception):
            self.client.close()

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "milvus": {
                "uri": sanitize_milvus_uri(self.milvus_uri),
                "mode": "lite" if self.local_milvus else "server",
                "db_name": self.milvus_db_name,
                "collection_name": self.collection_name,
                "vector_field": VECTOR_FIELD_NAME,
                "vector_dim": self.vector_dim,
            },
            "models": {
                "recall": self.recall_model,
                "rerank": self.rerank_model,
            },
            "defaults": {
                "dense_top_k": self.cfg.dense_top_k,
                "lexical_top_k": self.cfg.lexical_top_k,
                "heading_top_k": self.cfg.heading_top_k,
                "candidate_pool_size": self.cfg.candidate_pool_size,
                "final_top_k": self.cfg.final_top_k,
                "rrf_k": self.cfg.rrf_k,
                "dense_weight": self.cfg.dense_weight,
                "body_weight": self.cfg.body_weight,
                "heading_weight": self.cfg.heading_weight,
                "rerank_fusion_boost": self.cfg.rerank_fusion_boost,
            },
        }

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            index = self._index
        return {
            "status": "ok" if index is not None else "loading",
            "loaded_chunk_count": len(index.rows) if index is not None else 0,
            "loaded_at_utc": index.loaded_at_utc if index is not None else None,
            "config": self.config_snapshot(),
        }

    def reload(self) -> dict[str, Any]:
        with self._milvus_lock:
            self.client.load_collection(collection_name=self.collection_name)
            iterator = self.client.query_iterator(
                collection_name=self.collection_name,
                batch_size=QUERY_ITERATOR_BATCH_SIZE,
                filter="indexable == true",
                output_fields=MILVUS_OUTPUT_FIELDS,
            )

            rows: list[ChunkRecord] = []
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for item in batch:
                        rows.append(build_chunk_record(normalize_result_item(item)))
            finally:
                with suppress(Exception):
                    iterator.close()

        rows.sort(key=lambda row: (row.doc_id, row.chunk_order, row.chunk_id))
        if not rows:
            raise RuntimeError(
                f"Milvus collection {self.collection_name} did not return any indexable rows."
            )

        body_bm25, body_placeholder_rows = build_bm25_index(
            [row.embedding_text for row in rows]
        )
        heading_bm25, heading_placeholder_rows = build_bm25_index(
            [row.heading_path_text for row in rows]
        )

        index = RetrievalIndex(
            rows=tuple(rows),
            row_by_chunk_id={row.chunk_id: row for row in rows},
            body_bm25=body_bm25,
            heading_bm25=heading_bm25,
            body_placeholder_rows=body_placeholder_rows,
            heading_placeholder_rows=heading_placeholder_rows,
            loaded_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        )

        with self._state_lock:
            self._index = index

        return {
            "status": "reloaded",
            "loaded_chunk_count": len(index.rows),
            "loaded_at_utc": index.loaded_at_utc,
            "config": self.config_snapshot(),
        }

    def reload_with_retry(
        self,
        *,
        max_attempts: int = DEFAULT_STARTUP_RELOAD_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_STARTUP_RELOAD_RETRY_DELAY_SECONDS,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self.reload()
            except Exception as exc:
                last_exc = exc
                if not is_transient_milvus_query_error(exc) or attempt >= max_attempts:
                    raise
                logger.warning(
                    "Milvus collection %s is not query-ready yet (%s/%s): %s. Retrying in %.1f seconds.",
                    self.collection_name,
                    attempt,
                    max_attempts,
                    exc,
                    retry_delay_seconds,
                )
                time.sleep(retry_delay_seconds)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("reload_with_retry exhausted attempts without a result.")

    def _require_index(self) -> RetrievalIndex:
        with self._state_lock:
            index = self._index
        if index is None:
            raise RuntimeError("Retrieval index is not loaded yet.")
        return index

    def _embed_query(self, query: str, *, for_rerank: bool) -> np.ndarray:
        embeddings = self.rerank_embeddings if for_rerank else self.recall_embeddings
        vector = embeddings.embed_query(query)
        return ensure_cosine_vector(vector, expected_dim=self.vector_dim)

    def _dense_search(self, query_vector: np.ndarray, *, milvus_filter: str, top_k: int) -> list[LaneHit]:
        if top_k <= 0:
            return []

        with self._milvus_lock:
            results = self.client.search(
                collection_name=self.collection_name,
                anns_field=VECTOR_FIELD_NAME,
                data=[query_vector.tolist()],
                filter=milvus_filter,
                limit=top_k,
                output_fields=SEARCH_OUTPUT_FIELDS,
                search_params={"metric_type": VECTOR_SEARCH_METRIC_TYPE, "params": dict(VECTOR_SEARCH_PARAMS)},
            )

        first_result_set: list[Any]
        if isinstance(results, list) and results and isinstance(results[0], list):
            first_result_set = results[0]
        elif isinstance(results, list):
            first_result_set = results
        else:
            first_result_set = []

        dense_hits: list[LaneHit] = []
        for rank, raw_hit in enumerate(first_result_set, start=1):
            hit = normalize_result_item(raw_hit)
            entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
            chunk_id = (
                entity.get(PRIMARY_KEY_FIELD)
                or hit.get(PRIMARY_KEY_FIELD)
                or hit.get("id")
                or hit.get("pk")
            )
            if chunk_id is None:
                continue
            raw_score = hit.get("score")
            distance = hit.get("distance")
            dense_hits.append(
                LaneHit(
                    chunk_id=str(chunk_id),
                    rank=rank,
                    raw_score=float(raw_score) if raw_score is not None else None,
                    distance=float(distance) if distance is not None else None,
                )
            )
        return dense_hits

    def _fetch_candidate_vectors(self, chunk_ids: list[str]) -> dict[str, Sequence[float]]:
        if not chunk_ids:
            return {}
        with self._milvus_lock:
            rows = self.client.get(
                collection_name=self.collection_name,
                ids=chunk_ids,
                output_fields=VECTOR_GET_OUTPUT_FIELDS,
            )

        vectors: dict[str, Sequence[float]] = {}
        for item in rows:
            row = normalize_result_item(item)
            chunk_id = row.get(PRIMARY_KEY_FIELD) or row.get("id") or row.get("pk")
            vector = row.get(VECTOR_FIELD_NAME)
            if chunk_id is None or vector is None:
                continue
            vectors[str(chunk_id)] = vector
        return vectors

    def retrieve(self, request: RetrieveRequest) -> dict[str, Any]:
        if request.filters is not None and request.filters.page_from and request.filters.page_to:
            if request.filters.page_from > request.filters.page_to:
                raise ValueError("filters.page_from must be less than or equal to filters.page_to.")

        query = request.query.strip()
        if not query:
            raise ValueError("query must not be blank.")

        index = self._require_index()
        top_k = request.top_k or self.cfg.final_top_k
        dense_top_k = request.dense_top_k or self.cfg.dense_top_k
        lexical_top_k = request.lexical_top_k or self.cfg.lexical_top_k
        heading_top_k = request.heading_top_k or self.cfg.heading_top_k
        candidate_pool_size = request.candidate_pool_size or self.cfg.candidate_pool_size
        candidate_pool_size = max(candidate_pool_size, top_k)

        filter_predicate = build_filter_predicate(request.filters)
        milvus_filter = build_milvus_filter(request.filters)
        started_at = time.perf_counter()

        recall_query_vector = self._embed_query(query, for_rerank=False)
        dense_hits = self._dense_search(
            recall_query_vector,
            milvus_filter=milvus_filter,
            top_k=dense_top_k,
        )

        query_tokens = tokenize(query)
        body_hits = top_k_bm25_hits(
            index.body_bm25,
            index.body_placeholder_rows,
            index.rows,
            query_tokens=query_tokens,
            top_k=lexical_top_k,
            predicate=filter_predicate,
        )
        heading_hits = top_k_bm25_hits(
            index.heading_bm25,
            index.heading_placeholder_rows,
            index.rows,
            query_tokens=query_tokens,
            top_k=heading_top_k,
            predicate=filter_predicate,
        )

        fused_candidates: dict[str, CandidateScore] = {}
        self._apply_rrf_lane(
            fused_candidates,
            lane_hits=dense_hits,
            lane_name="dense",
            lane_weight=self.cfg.dense_weight,
        )
        self._apply_rrf_lane(
            fused_candidates,
            lane_hits=body_hits,
            lane_name="body",
            lane_weight=self.cfg.body_weight,
        )
        self._apply_rrf_lane(
            fused_candidates,
            lane_hits=heading_hits,
            lane_name="heading",
            lane_weight=self.cfg.heading_weight,
        )

        ranked_candidates = sorted(
            fused_candidates.values(),
            key=lambda item: (item.fused_rrf, -(item.dense_rank or 10**9), -(item.body_rank or 10**9)),
            reverse=True,
        )
        ranked_candidates = ranked_candidates[:candidate_pool_size]

        if not ranked_candidates:
            return {
                "query": query,
                "took_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "config": self.config_snapshot(),
                "applied_filters": request.filters.model_dump(mode="json") if request.filters else None,
                "counts": {
                    "dense_hits": len(dense_hits),
                    "body_hits": len(body_hits),
                    "heading_hits": len(heading_hits),
                    "candidate_pool": 0,
                    "returned": 0,
                },
                "results": [],
            }

        rerank_query_vector = (
            recall_query_vector
            if self.recall_model == self.rerank_model
            else self._embed_query(query, for_rerank=True)
        )
        candidate_ids = [candidate.chunk_id for candidate in ranked_candidates]
        candidate_vectors = self._fetch_candidate_vectors(candidate_ids)

        results: list[dict[str, Any]] = []
        for candidate in ranked_candidates:
            row = index.row_by_chunk_id.get(candidate.chunk_id)
            vector = candidate_vectors.get(candidate.chunk_id)
            if row is None or vector is None:
                continue
            rerank_cosine = cosine_similarity(rerank_query_vector, vector)
            final_score = rerank_cosine + (self.cfg.rerank_fusion_boost * candidate.fused_rrf)
            results.append(
                {
                    "chunk": row,
                    "scores": {
                        "final_score": float(final_score),
                        "rerank_cosine": float(rerank_cosine),
                        "fused_rrf": float(candidate.fused_rrf),
                        "dense_rrf": float(candidate.dense_rrf),
                        "body_rrf": float(candidate.body_rrf),
                        "heading_rrf": float(candidate.heading_rrf),
                        "dense_rank": candidate.dense_rank,
                        "body_rank": candidate.body_rank,
                        "heading_rank": candidate.heading_rank,
                        "dense_raw_score": candidate.dense_raw_score,
                        "dense_distance": candidate.dense_distance,
                        "body_bm25": candidate.body_bm25,
                        "heading_bm25": candidate.heading_bm25,
                    },
                }
            )

        results.sort(
            key=lambda item: (
                item["scores"]["final_score"],
                item["scores"]["rerank_cosine"],
                item["scores"]["fused_rrf"],
            ),
            reverse=True,
        )
        results = results[:top_k]

        payload_results: list[dict[str, Any]] = []
        for rank, item in enumerate(results, start=1):
            row = item["chunk"]
            scores = item["scores"]
            payload = {
                "rank": rank,
                "chunk_id": row.chunk_id,
                "doc_id": row.doc_id,
                "chunk_type": row.chunk_type,
                "content_modality": row.content_modality,
                "document_title": row.document_title,
                "section_title": row.section_title,
                "heading_path": list(row.heading_path),
                "page_start": row.page_start,
                "page_end": row.page_end,
                "display_text": row.display_text,
                "retrieval_text": row.embedding_text,
                "prev_chunk_id": row.prev_chunk_id,
                "next_chunk_id": row.next_chunk_id,
                "metadata": row.metadata,
                "citation": {
                    "doc_id": row.doc_id,
                    "page_start": row.page_start,
                    "page_end": row.page_end,
                    "heading_path": list(row.heading_path),
                    "source_block_ids": row.metadata.get("source_block_ids") or [],
                    "source_marker_block_ids": row.metadata.get("source_marker_block_ids") or [],
                },
                "scores": scores,
            }
            if request.include_neighbors and request.max_neighbors_per_side > 0:
                payload["neighbors"] = self._serialize_neighbors(
                    index,
                    row,
                    max_neighbors_per_side=request.max_neighbors_per_side,
                )
            payload_results.append(payload)

        return {
            "query": query,
            "took_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "config": self.config_snapshot(),
            "applied_filters": request.filters.model_dump(mode="json") if request.filters else None,
            "counts": {
                "dense_hits": len(dense_hits),
                "body_hits": len(body_hits),
                "heading_hits": len(heading_hits),
                "candidate_pool": len(ranked_candidates),
                "returned": len(payload_results),
            },
            "results": payload_results,
        }

    def _apply_rrf_lane(
        self,
        candidates: dict[str, CandidateScore],
        *,
        lane_hits: Sequence[LaneHit],
        lane_name: str,
        lane_weight: float,
    ) -> None:
        for lane_hit in lane_hits:
            candidate = candidates.setdefault(lane_hit.chunk_id, CandidateScore(chunk_id=lane_hit.chunk_id))
            contribution = lane_weight / (self.cfg.rrf_k + lane_hit.rank)
            candidate.fused_rrf += contribution
            if lane_name == "dense":
                candidate.dense_rrf += contribution
                candidate.dense_rank = lane_hit.rank
                candidate.dense_raw_score = lane_hit.raw_score
                candidate.dense_distance = lane_hit.distance
            elif lane_name == "body":
                candidate.body_rrf += contribution
                candidate.body_rank = lane_hit.rank
                candidate.body_bm25 = lane_hit.raw_score
            elif lane_name == "heading":
                candidate.heading_rrf += contribution
                candidate.heading_rank = lane_hit.rank
                candidate.heading_bm25 = lane_hit.raw_score

    def _serialize_neighbors(
        self,
        index: RetrievalIndex,
        row: ChunkRecord,
        *,
        max_neighbors_per_side: int,
    ) -> dict[str, list[dict[str, Any]]]:
        previous: list[dict[str, Any]] = []
        next_rows: list[dict[str, Any]] = []

        cursor = row.prev_chunk_id
        for _ in range(max_neighbors_per_side):
            if not cursor:
                break
            neighbor = index.row_by_chunk_id.get(cursor)
            if neighbor is None:
                break
            previous.insert(0, self._serialize_neighbor_row(neighbor))
            cursor = neighbor.prev_chunk_id

        cursor = row.next_chunk_id
        for _ in range(max_neighbors_per_side):
            if not cursor:
                break
            neighbor = index.row_by_chunk_id.get(cursor)
            if neighbor is None:
                break
            next_rows.append(self._serialize_neighbor_row(neighbor))
            cursor = neighbor.next_chunk_id

        return {
            "previous": previous,
            "next": next_rows,
        }

    @staticmethod
    def _serialize_neighbor_row(row: ChunkRecord) -> dict[str, Any]:
        return {
            "chunk_id": row.chunk_id,
            "doc_id": row.doc_id,
            "chunk_type": row.chunk_type,
            "content_modality": row.content_modality,
            "section_title": row.section_title,
            "heading_path": list(row.heading_path),
            "page_start": row.page_start,
            "page_end": row.page_end,
            "display_text": row.display_text,
        }


def get_service_from_request(request: Request) -> OnlineRAGService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service is not ready.")
    return service


def create_app(cfg: RuntimeConfig) -> FastAPI:
    env_values = merged_env(load_simple_env(cfg.env_file))
    app = FastAPI(
        title="Dental Manual Online RAG Service",
        version="1.0.0",
        summary="Milvus-backed hybrid retrieval service for phase-04 online RAG.",
    )

    @app.on_event("startup")
    async def on_startup() -> None:
        service = OnlineRAGService(cfg, env_values)
        app.state.rag_service = service
        await run_in_threadpool(service.reload_with_retry)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        service = getattr(app.state, "rag_service", None)
        if service is None:
            return
        await run_in_threadpool(service.close)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return get_service_from_request(request).health()

    @app.post("/reload")
    async def reload_indexes(request: Request) -> dict[str, Any]:
        service = get_service_from_request(request)
        try:
            return await run_in_threadpool(service.reload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/retrieve")
    async def retrieve(request: Request, payload: RetrieveRequest) -> dict[str, Any]:
        service = get_service_from_request(request)
        try:
            return await run_in_threadpool(service.retrieve, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app(default_runtime_config())


def main() -> int:
    cfg = parse_args()
    uvicorn.run(
        create_app(cfg),
        host=cfg.host,
        port=cfg.port,
        log_level="debug" if cfg.debug else "info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
