from collections.abc import Sequence
from pathlib import Path

import pytest
from conftest import FakeEmbedder, WordTokenizer, document

from rag_service.chunking.factory import create_chunker
from rag_service.chunking.preview import preview_file
from rag_service.chunking.recursive import RecursiveCharacterSplitter
from rag_service.chunking.semantic import SemanticChunker
from rag_service.chunking.spans import Span, sections, sentences
from rag_service.config import IngestionConfig
from rag_service.embeddings.cached import CachedEmbedder
from rag_service.ingestion.normalization import normalize
from rag_service.ingestion.pipeline import IngestionPipeline
from rag_service.schemas import EmbeddingSpec
from rag_service.storage.manifest import Manifest
from rag_service.storage.qdrant import QdrantVectorStore

TOPICS = (
    "Gene repair protects DNA. Cancer mutations affect genes. "
    "Database backup protects records. Backup storage supports restore."
)


class TopicEmbedder(FakeEmbedder):
    """Known semantic geometry tests boundary logic without pretending to evaluate a model."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        assert all(self.tokenizer.count(text) <= self.spec.max_tokens for text in texts)
        self.calls.extend(texts)
        vectors = []
        for text in texts:
            text = text.lower()
            biology = sum(text.count(word) for word in ("gene", "dna", "cancer", "mutation"))
            storage = sum(text.count(word) for word in ("database", "backup", "storage", "restore"))
            vectors.append([float(biology), float(storage), 0.01, 0.0])
        return vectors


def test_topic_change_splits_below_token_ceiling() -> None:
    source = normalize(document("topics", TOPICS))
    chunks = list(SemanticChunker(TopicEmbedder(), 100).chunks(source))
    assert [chunk.text for chunk in chunks] == [
        "Gene repair protects DNA. Cancer mutations affect genes.",
        "Database backup protects records. Backup storage supports restore.",
    ]
    assert all(chunk.split_reason == "semantic" for chunk in chunks)
    assert chunks[0].semantic_group_id != chunks[1].semantic_group_id


def test_short_coherent_section_stays_intact() -> None:
    source = normalize(
        document("one-topic", "Gene repair protects DNA. Cancer mutations affect genes.")
    )
    chunks = list(SemanticChunker(TopicEmbedder(), 100).chunks(source))
    assert len(chunks) == 1
    assert chunks[0].text == source.text


def test_two_sentence_contexts_do_not_overlap_across_topic_boundary() -> None:
    source = normalize(
        document("two", "Gene repair protects DNA. Database backup protects storage.")
    )
    assert len(list(SemanticChunker(TopicEmbedder(), 100).chunks(source))) == 2


def test_headings_stay_with_body_and_separate_sections() -> None:
    source = normalize(
        document(
            "headings", "# Genes\n\nGene repair protects DNA.\n\n# Cancer\n\nCancer affects genes."
        )
    )
    chunks = list(SemanticChunker(TopicEmbedder(), 100).chunks(source))
    assert len(chunks) == 2
    assert [chunk.heading for chunk in chunks] == ["Genes", "Cancer"]
    assert all(chunk.text.count("\n") >= 1 for chunk in chunks)
    assert all(chunk.split_reason == "structure" for chunk in chunks)


def test_oversized_semantic_group_uses_recursive_fallback_and_keeps_offsets() -> None:
    text = "Gene repair protects DNA. " * 20
    source = normalize(document("large", text))
    chunks = list(SemanticChunker(TopicEmbedder(), 12).chunks(source))
    assert len(chunks) > 1
    assert len({chunk.semantic_group_id for chunk in chunks}) == 1
    covered: set[int] = set()
    for chunk in chunks:
        assert chunk.token_count <= 12
        assert chunk.split_reason == "size"
        assert chunk.text == source.text[chunk.start_char : chunk.end_char]
        covered.update(range(chunk.start_char, chunk.end_char))
    assert all(index in covered for index, char in enumerate(source.text) if not char.isspace())


def test_giant_sentence_is_bounded_before_semantic_embedding() -> None:
    embedder = TopicEmbedder()
    embedder.spec = EmbeddingSpec(model="test", revision="v1", dimension=4, max_tokens=16)
    source = normalize(document("long", "gene " * 100 + ". Database backup protects storage."))
    chunks = list(SemanticChunker(embedder, 8).chunks(source))
    assert all(chunk.token_count <= 8 for chunk in chunks)
    assert all(embedder.tokenizer.count(text) <= 16 for text in embedder.calls)


def test_recursive_character_fallback_for_unbroken_unicode() -> None:
    class CharacterTokenizer:
        def count(self, text: str) -> int:
            return len(text)

        def offsets(self, text: str) -> list[tuple[int, int]]:
            return [(index, index + 1) for index in range(len(text))]

    text = "基因修复β" * 20
    spans = RecursiveCharacterSplitter(CharacterTokenizer(), 12).split(text, Span(0, len(text)))
    assert "".join(span.text(text) for span in spans) == text
    assert all(len(span.text(text)) <= 12 for span in spans)


def test_biomedical_abbreviations_and_decimals_do_not_create_sentence_boundaries() -> None:
    text = "Dr. Chen measured 3.14 mg. Fig. 2 shows DNA repair. Another result follows."
    assert [span.text(text) for span in sentences(text, Span(0, len(text)))] == [
        "Dr. Chen measured 3.14 mg.",
        "Fig. 2 shows DNA repair.",
        "Another result follows.",
    ]


def test_code_fences_do_not_create_markdown_sections() -> None:
    text = "# Real\n\n```python\n# comment\nprint(1)\n```\n\n# Next\nBody."
    assert len(sections(text)) == 2


def test_overlap_does_not_cross_semantic_groups() -> None:
    chunks = list(
        SemanticChunker(TopicEmbedder(), 100, overlap_tokens=20).chunks(
            normalize(document("x", TOPICS))
        )
    )
    assert len(chunks) == 2
    assert "DNA" not in chunks[1].text


def test_semantic_pipeline_cache_and_configuration_versioning(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite3")
    store = QdrantVectorStore(path=tmp_path / "qdrant")
    embedder = TopicEmbedder()
    config = IngestionConfig(data_dir=tmp_path, chunk_tokens=100, batch_size=2)
    try:
        pipeline = IngestionPipeline(config, embedder, store, manifest)
        first = pipeline.ingest([document("x", TOPICS)])
        calls = len(embedder.calls)
        second = pipeline.ingest([document("x", TOPICS)])
        assert first.chunks == 2
        assert first.semantic_embedded_texts > 0
        assert second.status == "unchanged"
        assert second.semantic_embedded_texts == 0
        assert len(embedder.calls) == calls
        changed = config.model_copy(update={"semantic_min_distance": 2})
        third = IngestionPipeline(changed, embedder, store, manifest).ingest(
            [document("x", TOPICS)]
        )
        assert third.generation_id != first.generation_id
        assert third.chunks == 1
    finally:
        store.close()
        manifest.close()


def test_preview_does_not_publish_a_generation(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text(TOPICS, encoding="utf-8")
    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        embedder = CachedEmbedder(TopicEmbedder(), manifest, 2)
        report = preview_file(source, IngestionConfig(data_dir=tmp_path), embedder)
        assert not report["indexed"]
        assert len(report["chunks"]) == 2
        assert manifest.generations("default") == []
        assert not (tmp_path / "qdrant").exists()
    finally:
        manifest.close()


def test_default_strategy_is_semantic() -> None:
    assert isinstance(create_chunker(IngestionConfig(), TopicEmbedder()), SemanticChunker)


def test_recursive_preserves_repeated_text_and_whole_sentence_overlap() -> None:
    text = "Gene repair. Gene repair.\n\nGene repair. Gene repair.\n\nGene repair. Gene repair."
    spans = RecursiveCharacterSplitter(WordTokenizer(), 6, overlap_tokens=2).split(
        text, Span(0, len(text))
    )
    assert len(spans) > 1
    assert spans[1].start < spans[0].end
    assert all(WordTokenizer().count(span.text(text)) <= 6 for span in spans)


def test_overlap_never_copies_a_mid_sentence_fragment() -> None:
    text = "Gene repair. unfinished fragment\nnext four words here\nlast four words here."
    spans = RecursiveCharacterSplitter(WordTokenizer(), 6, overlap_tokens=2).split(
        text, Span(0, len(text))
    )
    assert all(right.start >= left.end for left, right in zip(spans, spans[1:], strict=False))


@pytest.mark.parametrize("value", [-1, 101])
def test_invalid_percentile_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        IngestionConfig(semantic_percentile=value)
