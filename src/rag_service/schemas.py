"""Immutable domain records; offsets refer to the stored normalized document text."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceDocument(Record):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    title: str = ""
    page: int | None = Field(default=None, ge=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class Document(SourceDocument):
    raw_text: str
    content_hash: str
    version: str


class Chunk(Record):
    chunk_id: str
    document_id: str
    document_version: str
    position: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    source_uri: str
    title: str
    heading: str = ""
    page: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    semantic_group_id: str | None = None
    split_reason: str | None = None


class EmbeddingSpec(Record):
    model: str
    revision: str
    dimension: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    normalized: bool = True

    @property
    def cache_namespace(self) -> str:
        return fingerprint(self.model_dump())


class IngestionResult(Record):
    generation_id: str
    collection: str
    status: Literal["ready", "unchanged"]
    documents: int
    chunks: int
    embedded_chunks: int = 0
    cached_chunks: int = 0
    semantic_embedded_texts: int = 0
    semantic_cached_texts: int = 0


class EvaluationQuestion(Record):
    query_id: str
    query: str
    reference_answer: str
    relevant_document_ids: list[str]
    missing_document_ids: list[str] = Field(default_factory=list)


class IndexMetadata(Record):
    """Stored in Qdrant collection metadata so query workers need no local manifest."""

    generation_id: str
    collection: str
    fingerprint: str
    embedding: EmbeddingSpec
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)


class SearchHit(Record):
    score: float
    chunk: Chunk
