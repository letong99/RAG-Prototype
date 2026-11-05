"""Detect topic changes between sentence contexts, then recursively bound each semantic group."""

import math
from collections.abc import Iterator
from itertools import pairwise

from rag_service.chunking.recursive import RecursiveCharacterSplitter
from rag_service.chunking.spans import Span, heading, sections, sentences, trim
from rag_service.ports import Embedder
from rag_service.schemas import Chunk, Document, fingerprint


def cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Semantic embeddings must have the same nonzero dimension")
    if not all(math.isfinite(value) for value in [*left, *right]):
        raise ValueError("Semantic embeddings must be finite")
    norm = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    if norm == 0:
        raise ValueError("Semantic embeddings must be nonzero")
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / norm
    return 1.0 - max(-1.0, min(1.0, similarity))


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentage / 100
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


class SemanticChunker:
    version = "semantic-recursive-v1"

    def __init__(
        self,
        embedder: Embedder,
        max_tokens: int,
        overlap_tokens: int = 0,
        *,
        breakpoint_percentile: float = 85,
        min_distance: float = 0.1,
        window_sentences: int = 2,
    ) -> None:
        if not 0 <= breakpoint_percentile <= 100 or not 0 <= min_distance <= 2:
            raise ValueError("Invalid semantic breakpoint settings")
        if window_sentences < 1:
            raise ValueError("window_sentences must be positive")
        if max_tokens > embedder.spec.max_tokens:
            raise ValueError("Chunk budget exceeds the embedding model input limit")
        self.embedder = embedder
        self.splitter = RecursiveCharacterSplitter(embedder.tokenizer, max_tokens, overlap_tokens)
        self.input_splitter = RecursiveCharacterSplitter(
            embedder.tokenizer, embedder.spec.max_tokens
        )
        self.breakpoint_percentile = breakpoint_percentile
        self.min_distance = min_distance
        self.window_sentences = window_sentences
        self.identity = fingerprint(
            {
                "version": self.version,
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
                "percentile": breakpoint_percentile,
                "min_distance": min_distance,
                "window_sentences": window_sentences,
                "embedding": embedder.spec.model_dump(),
                "recursive_version": self.splitter.version,
            }
        )

    def chunks(self, document: Document) -> Iterator[Chunk]:
        position = 0
        source_sections = sections(document.text)
        for section in source_sections:
            groups = self._groups(document.text, section)
            for group in groups:
                bounded = self.splitter.split(document.text, group)
                group_id = fingerprint(
                    [document.document_id, document.version, self.identity, group.start, group.end]
                )
                reason = "semantic" if len(groups) > 1 else "structure"
                if len(groups) == 1 and len(source_sections) == 1:
                    reason = "document"
                for span in bounded:
                    text = span.text(document.text)
                    yield Chunk(
                        chunk_id=fingerprint([group_id, span.start, span.end]),
                        document_id=document.document_id,
                        document_version=document.version,
                        position=position,
                        text=text,
                        start_char=span.start,
                        end_char=span.end,
                        token_count=self.embedder.tokenizer.count(text),
                        source_uri=document.source_uri,
                        title=document.title,
                        heading=heading(document.text, section),
                        page=document.page,
                        metadata=document.metadata,
                        semantic_group_id=group_id,
                        split_reason="size" if len(bounded) > 1 else reason,
                    )
                    position += 1

    def _groups(self, source: str, section: Span) -> list[Span]:
        body = section
        if heading(source, section):
            newline = source.find("\n", section.start, section.end)
            if newline < 0 or (content := trim(source, Span(newline + 1, section.end))) is None:
                return [section]
            body = content
        units = [
            piece
            for sentence in sentences(source, body)
            for piece in self.input_splitter.split(source, sentence)
        ]
        if len(units) < 2:
            return [section]
        contexts = []
        for boundary in range(1, len(units)):
            contexts.append(self._context(source, units, boundary, left=True))
            contexts.append(self._context(source, units, boundary, left=False))
        vectors = self.embedder.embed(contexts)
        if len(vectors) != len(contexts):
            raise ValueError("Semantic embedder returned an incorrect number of vectors")
        distances = [
            cosine_distance(vectors[index], vectors[index + 1])
            for index in range(0, len(vectors), 2)
        ]
        threshold = percentile(distances, self.breakpoint_percentile)
        breaks = [
            index + 1
            for index, distance in enumerate(distances)
            if distance > self.min_distance and distance >= threshold
        ]
        boundaries = [0, *breaks, len(units)]
        return [
            Span(section.start if start == 0 else units[start].start, units[end - 1].end)
            for start, end in pairwise(boundaries)
        ]

    def _context(self, source: str, units: list[Span], boundary: int, *, left: bool) -> str:
        start = max(0, boundary - self.window_sentences) if left else boundary
        end = boundary if left else min(len(units), boundary + self.window_sentences)
        # Drop distant whole sentences rather than truncate the sentence nearest the boundary.
        while (
            self.embedder.tokenizer.count(source[units[start].start : units[end - 1].end])
            > self.embedder.spec.max_tokens
        ):
            if left:
                start += 1
            else:
                end -= 1
        return source[units[start].start : units[end - 1].end]
