from conftest import WordTokenizer, document

from rag_service.chunking.token_chunker import TokenChunker
from rag_service.ingestion.normalization import normalize


def test_chunk_budget_coverage_overlap_and_provenance() -> None:
    source = normalize(document("a", "# Heading\n\n" + " ".join(f"word{i}" for i in range(70))))
    chunker = TokenChunker(WordTokenizer(), size=12, overlap=3)
    chunks = list(chunker.chunks(source))
    assert len(chunks) > 1
    assert chunks == list(chunker.chunks(source))
    covered: set[int] = set()
    for chunk in chunks:
        assert chunk.text == source.text[chunk.start_char : chunk.end_char]
        assert chunk.token_count <= 12
        covered.update(range(chunk.start_char, chunk.end_char))
    assert all(index in covered for index, char in enumerate(source.text) if not char.isspace())
    assert chunks[1].start_char < chunks[0].end_char
    assert chunks[-1].heading == "Heading"


def test_normalization_preserves_raw_text() -> None:
    raw = "  alpha  \r\n\r\nbeta\x00  "
    normalized = normalize(document("a", raw))
    assert normalized.raw_text == raw
    assert normalized.text == "alpha\n\nbeta"
    assert normalized.version != normalize(document("a", "other")).version


def test_boundary_preference() -> None:
    source = normalize(
        document("a", "one two three four five six.\n\nseven eight nine ten eleven twelve")
    )
    chunks = list(TokenChunker(WordTokenizer(), 10, 1).chunks(source))
    assert chunks[0].text == "one two three four five six."
