"""Rank-preserving context assembly with explicit provenance and wire limits."""

from pydantic import Field

from rag_service.ports import Tokenizer
from rag_service.retrieval.models import SearchResult
from rag_service.schemas import Chunk, IndexMetadata, Record


class ContextLimits(Record):
    max_context_tokens: int = Field(default=2500, ge=1, le=16000)
    max_response_bytes: int = Field(default=100000, ge=4096, le=1000000)
    max_passages: int = Field(default=6, ge=1, le=100)
    overlap_fraction: float = Field(default=0.8, gt=0, le=1)
    tokenizer_name: str = Field(default="embedding tokenizer", min_length=1)
    token_counts_estimated: bool = True


class ContextPassage(Record):
    citation_id: str
    chunk: Chunk
    original_end_char: int
    truncated: bool = False


class ContextResult(Record):
    query: str
    index: IndexMetadata
    passages: list[ContextPassage]
    context_tokens: int
    tokenizer_name: str
    token_counts_estimated: bool
    omitted_count: int
    duplicate_count: int
    truncated: bool
    warnings: list[str]


def evidence_text(passages: list[ContextPassage]) -> str:
    """JSON escaping keeps source boundaries unambiguous, including hostile source text."""
    import json

    return json.dumps([p.model_dump() for p in passages], ensure_ascii=False)


class ContextAssembler:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def assemble(self, result: SearchResult, limits: ContextLimits) -> ContextResult:
        selected: list[ContextPassage] = []
        seen: list[Chunk] = []
        duplicates = 0
        omitted = 0

        def response() -> ContextResult:
            truncated = bool(omitted or any(p.truncated for p in selected))
            return ContextResult(
                query=result.request.query,
                index=result.index,
                passages=selected.copy(),
                context_tokens=self.tokenizer.count(evidence_text(selected)),
                tokenizer_name=limits.tokenizer_name,
                token_counts_estimated=limits.token_counts_estimated,
                omitted_count=omitted,
                duplicate_count=duplicates,
                truncated=truncated,
                warnings=["Some ranked evidence was omitted or shortened."] if truncated else [],
            )

        def fits() -> bool:
            candidate = response()
            # Reserve space for final omission counters and warning text.
            return (
                candidate.context_tokens <= limits.max_context_tokens
                and len(candidate.model_dump_json().encode("utf-8")) + 256
                <= limits.max_response_bytes
            )

        for ranked in result.passages:
            chunk = ranked.chunk
            if any(self._duplicate(chunk, prior, limits.overlap_fraction) for prior in seen):
                duplicates += 1
                continue
            seen.append(chunk)
            if len(selected) >= limits.max_passages:
                omitted += 1
                continue
            passage = ContextPassage(
                citation_id=f"S{len(selected) + 1}",
                chunk=chunk,
                original_end_char=chunk.end_char,
            )
            selected.append(passage)
            if fits():
                continue
            selected.pop()
            # Check actual serialized prefixes; token counts need not be monotonic.
            for _, end in reversed(self.tokenizer.offsets(chunk.text)[:-1]):
                text = chunk.text[:end]
                shortened = chunk.model_copy(
                    update={
                        "text": text,
                        "end_char": chunk.start_char + end,
                        "token_count": self.tokenizer.count(text),
                    }
                )
                selected.append(passage.model_copy(update={"chunk": shortened, "truncated": True}))
                if fits():
                    break
                selected.pop()
            else:
                omitted += 1
        assembled = response()
        if len(assembled.model_dump_json().encode("utf-8")) > limits.max_response_bytes:
            raise ValueError("Response metadata exceeds serialized response budget")
        return assembled

    @staticmethod
    def _duplicate(chunk: Chunk, prior: Chunk, fraction: float) -> bool:
        if chunk.chunk_id == prior.chunk_id or chunk.text.split() == prior.text.split():
            return True
        if (chunk.document_id, chunk.document_version, chunk.page) != (
            prior.document_id,
            prior.document_version,
            prior.page,
        ):
            return False
        overlap = max(
            0, min(chunk.end_char, prior.end_char) - max(chunk.start_char, prior.start_char)
        )
        shortest = min(chunk.end_char - chunk.start_char, prior.end_char - prior.start_char)
        return shortest > 0 and overlap / shortest >= fraction
