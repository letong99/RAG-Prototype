"""Prefer paragraph/sentence boundaries within token windows; never silently truncate."""

import re
from bisect import bisect_right
from collections.abc import Iterator

from rag_service.ports import Tokenizer
from rag_service.schemas import Chunk, Document, fingerprint


class TokenChunker:
    version = "boundary-token-v1"

    def __init__(self, tokenizer: Tokenizer, size: int, overlap: int) -> None:
        if size < 1 or not 0 <= overlap < size:
            raise ValueError("Expected size > 0 and 0 <= overlap < size")
        self.tokenizer = tokenizer
        self.size = size
        self.overlap = overlap

    def chunks(self, document: Document) -> Iterator[Chunk]:
        text = document.text
        offsets = self.tokenizer.offsets(text)
        if not offsets:
            raise ValueError(f"No tokens in {document.document_id!r}")
        boundaries = [match.end() for match in re.finditer(r"\n\s*\n|[.!?](?=\s)", text)]
        headings = list(re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE))
        heading_starts = [match.start() for match in headings]
        token_ends = [end for _, end in offsets]
        start = 0
        position = 0
        while start < len(offsets):
            stop = min(start + self.size, len(offsets))
            start_char = offsets[start][0]
            end_char = offsets[stop - 1][1]
            if stop < len(offsets):
                boundary_index = bisect_right(boundaries, end_char) - 1
                if boundary_index >= 0:
                    candidate = boundaries[boundary_index]
                    boundary_stop = bisect_right(token_ends, candidate)
                    if boundary_stop - start >= max(self.size // 2, self.overlap + 1):
                        stop = boundary_stop
                        end_char = offsets[stop - 1][1]
            # Retokenizing a substring can change tokenization at its edges.
            while self.tokenizer.count(text[start_char:end_char]) > self.size:
                stop -= 1
                if stop <= start:
                    raise ValueError("A source token cannot fit the chunk budget")
                end_char = offsets[stop - 1][1]
            chunk_text = text[start_char:end_char]
            heading = ""
            heading_index = bisect_right(heading_starts, start_char) - 1
            if heading_index >= 0:
                heading = headings[heading_index].group(1)
            chunk_id = fingerprint(
                [
                    document.document_id,
                    document.version,
                    self.version,
                    self.size,
                    self.overlap,
                    start_char,
                    end_char,
                ]
            )
            yield Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                document_version=document.version,
                position=position,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                token_count=self.tokenizer.count(chunk_text),
                source_uri=document.source_uri,
                title=document.title,
                heading=heading,
                page=document.page,
                metadata=document.metadata,
            )
            if stop == len(offsets):
                break
            start = max(start + 1, stop - self.overlap)
            position += 1
