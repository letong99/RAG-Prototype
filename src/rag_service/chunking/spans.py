"""Source-preserving text boundaries shared by semantic and recursive splitting."""

import re
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start : self.end]


def trim(source: str, span: Span) -> Span | None:
    start, end = span.start, span.end
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return Span(start, end) if start < end else None


def partition(source: str, span: Span, boundaries: list[int]) -> list[Span]:
    cuts = [
        span.start,
        *sorted({cut for cut in boundaries if span.start < cut < span.end}),
        span.end,
    ]
    return [part for start, end in pairwise(cuts) if (part := trim(source, Span(start, end)))]


_SENTENCE_END = re.compile(r"""[.!?。！？]+[”’"'）)\]]*(?=\s|$)|[。！？]+|\n[ \t]*\n""")
_ABBREVIATION = re.compile(
    r"(?:\b(?:dr|mr|mrs|ms|prof|fig|figs|eq|eqs|vs|etc|al|no|st|inc)\.|"
    r"\b(?:[a-z]\.){2,})$",
    re.IGNORECASE,
)


def sentences(source: str, span: Span) -> list[Span]:
    """Conservative English/CJK segmentation, including common biomedical abbreviations.

    This is a deterministic boundary heuristic, not a language-specific linguistic parser.
    Decimals do not match because a period must be followed by whitespace or end of text.
    """
    boundaries = []
    for match in _SENTENCE_END.finditer(source, span.start, span.end):
        if match.group().startswith("."):
            prefix = source[span.start : match.end()].rstrip()
            if _ABBREVIATION.search(prefix) or re.search(r"\b[A-Z]\.$", prefix):
                continue
        boundaries.append(match.end())
    return partition(source, span, boundaries)


def sections(source: str) -> list[Span]:
    """Markdown headings are hard boundaries; heading-like lines inside fences are ignored."""
    boundaries = []
    fence: str | None = None
    offset = 0
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
        elif fence is None and re.match(r"#{1,6}\s+", line):
            boundaries.append(offset)
        offset += len(line)
    return partition(source, Span(0, len(source)), boundaries)


def heading(source: str, span: Span) -> str:
    match = re.match(r"#{1,6}\s+([^\n]+)", span.text(source))
    return match.group(1).strip() if match else ""
