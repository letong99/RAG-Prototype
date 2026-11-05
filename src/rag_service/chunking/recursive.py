"""LangChain recursive splitting with exact source offsets and tokenizer-budget checks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_service.chunking.spans import Span, sentences, trim
from rag_service.ports import Tokenizer


class RecursiveCharacterSplitter:
    version = "langchain-recursive-v1"

    def __init__(self, tokenizer: Tokenizer, max_tokens: int, overlap_tokens: int = 0) -> None:
        if max_tokens < 1 or not 0 <= overlap_tokens < max_tokens:
            raise ValueError("Expected max_tokens > 0 and 0 <= overlap_tokens < max_tokens")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=0,
            length_function=tokenizer.count,
            separators=[
                r"\n[ \t]*\n",
                r"\n",
                r"(?<=[.!?。！？])\s+|(?<=[。！？])",
                r"\s+",
                r"(?<=[，、；;])",
                "",
            ],
            is_separator_regex=True,
            keep_separator="end",
            strip_whitespace=True,
        )

    def split(self, source: str, span: Span) -> list[Span]:
        cleaned = trim(source, span)
        if cleaned is None:
            return []
        if self.tokenizer.count(cleaned.text(source)) <= self.max_tokens:
            return [cleaned]
        bounded: list[Span] = []
        cursor = cleaned.start
        for text in self._splitter.split_text(cleaned.text(source)):
            start = source.find(text, cursor, cleaned.end)
            if start < 0 or source[cursor:start].strip():
                raise ValueError("Recursive splitter output cannot be mapped to source text")
            if self.tokenizer.count(text) > self.max_tokens:
                raise ValueError("Recursive splitter output exceeds the tokenizer budget")
            bounded.append(Span(start, start + len(text)))
            cursor = start + len(text)
        if source[cursor : cleaned.end].strip():
            raise ValueError("Recursive splitter dropped source content")
        return self._overlap(source, bounded)

    def _overlap(self, source: str, bounded: list[Span]) -> list[Span]:
        """Best-effort whole-sentence overlap, confined to this semantic group."""
        if not self.overlap_tokens:
            return bounded
        if not bounded:
            return []
        complete_units = sentences(source, Span(bounded[0].start, bounded[-1].end))
        result = []
        for index, current in enumerate(bounded):
            start = current.start
            if index:
                previous_start = bounded[index - 1].start
                candidates = [
                    unit
                    for unit in complete_units
                    if unit.start >= previous_start and unit.end <= current.start
                ]
                # A size cut may end halfway through a sentence. Do not call that fragment overlap.
                if candidates and source[candidates[-1].end : current.start].strip():
                    candidates = []
                for sentence in reversed(candidates):
                    overlap = source[sentence.start : current.start]
                    combined = source[sentence.start : current.end]
                    if (
                        self.tokenizer.count(overlap) > self.overlap_tokens
                        or self.tokenizer.count(combined) > self.max_tokens
                    ):
                        break
                    start = sentence.start
            result.append(Span(start, current.end))
        return result
