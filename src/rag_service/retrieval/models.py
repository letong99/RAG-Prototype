"""Requests and reproducible retrieval results, independent of transport."""

from pydantic import Field, model_validator

from rag_service.schemas import Chunk, IndexMetadata, Record


class SearchRequest(Record):
    collection: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=16000)
    candidate_limit: int = Field(default=40, ge=1, le=100)
    top_n: int = Field(default=6, ge=1, le=100)
    document_ids: list[str] | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "SearchRequest":
        if not self.query.strip():
            raise ValueError("Query must contain non-whitespace text")
        if self.top_n > self.candidate_limit:
            raise ValueError("top_n cannot exceed candidate_limit")
        return self


class RankerSpec(Record):
    model: str
    revision: str
    max_pair_tokens: int = Field(gt=0)


class RankedPassage(Record):
    rank: int
    dense_rank: int
    dense_score: float = Field(allow_inf_nan=False)
    rerank_score: float | None = Field(default=None, allow_inf_nan=False)
    chunk: Chunk


class SearchResult(Record):
    request: SearchRequest
    index: IndexMetadata
    ranker: RankerSpec | None
    candidate_count: int
    passages: list[RankedPassage]
    elapsed_seconds: float
