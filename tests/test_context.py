import json
from unittest.mock import Mock

import pytest
from conftest import document

from rag_service.retrieval.context import ContextLimits
from rag_service.retrieval.generation import GeneratedAnswer, generate_answer
from rag_service.retrieval.models import SearchRequest
from rag_service.retrieval.service import RetrievalService


def setup(pipeline):
    pipeline.ingest([document("a"), document("b", "Vaccines prevent infections.")])
    return RetrievalService(pipeline.config.data_dir, pipeline.store)


def test_context_provenance_and_budgets(pipeline):
    service = setup(pipeline)
    request = SearchRequest(collection="default", query="backups")
    context = service.retrieve_context(request, pipeline.embedder)
    assert [p.citation_id for p in context.passages] == ["S1", "S2"]
    assert context.index.ingestion_revision == 1
    assert context.passages[0].chunk.document_version
    assert context.context_tokens == pipeline.embedder.tokenizer.count(
        json.dumps([p.model_dump() for p in context.passages], ensure_ascii=False)
    )
    tiny = service.retrieve_context(
        request, pipeline.embedder, limits=ContextLimits(max_context_tokens=1)
    )
    assert tiny.passages == [] and tiny.omitted_count == 2 and tiny.truncated
    assert tiny.context_tokens <= 1


def test_generation_citations_and_empty_context(pipeline):
    service = setup(pipeline)
    request = SearchRequest(collection="default", query="backups")
    generator = Mock()
    generator.generate.return_value = GeneratedAnswer(status="answered", answer="Thirty days [S1].")
    result = service.answer_question(request, pipeline.embedder, generator)
    assert result.prompt_version == "grounded-answer-v1"
    assert "untrusted" in generator.generate.call_args.kwargs["system"]
    for text in ["Thirty days.", "Thirty days [S99]."]:
        generator.generate.return_value = GeneratedAnswer(status="answered", answer=text)
        with pytest.raises(ValueError, match="citations"):
            service.answer_question(request, pipeline.embedder, generator)
    empty = service.retrieve_context(
        request.model_copy(update={"document_ids": []}), pipeline.embedder
    )
    generator.reset_mock()
    assert generate_answer(empty, generator).result.status == "insufficient_evidence"
    generator.generate.assert_not_called()


def test_dataset_inventory_without_embedding(pipeline):
    service = setup(pipeline)
    pipeline.embedder.fail = True
    overview = service.describe_dataset("default")
    assert overview.documents == 2 and overview.chunks == 2
    assert overview.index.ingestion_revision == 1


def test_overlap_and_prefix_truncation(pipeline):
    from rag_service.retrieval.context import ContextAssembler

    service = setup(pipeline)
    result = service.search(SearchRequest(collection="default", query="backups"), pipeline.embedder)
    first = result.passages[0]
    duplicate = first.model_copy(
        update={
            "chunk": first.chunk.model_copy(
                update={"chunk_id": "overlap", "text": first.chunk.text + " extra"}
            )
        }
    )
    pool = result.model_copy(update={"passages": [first, duplicate, result.passages[1]]})
    assembler = ContextAssembler(pipeline.embedder.tokenizer)
    full = assembler.assemble(pool, ContextLimits())
    assert full.duplicate_count == 1 and len(full.passages) == 2
    one = result.model_copy(update={"passages": [first]})
    complete = assembler.assemble(one, ContextLimits())
    short = assembler.assemble(one, ContextLimits(max_context_tokens=complete.context_tokens - 2))
    assert short.passages and short.passages[0].truncated
    passage = short.passages[0]
    assert first.chunk.text.startswith(passage.chunk.text)
    assert passage.chunk.end_char == passage.chunk.start_char + len(passage.chunk.text)
    assert passage.original_end_char == first.chunk.end_char
    assert short.context_tokens <= complete.context_tokens - 2


def test_wire_size_with_unicode_metadata(pipeline):
    from rag_service.retrieval.context import ContextAssembler

    result = setup(pipeline).search(
        SearchRequest(collection="default", query="backups"), pipeline.embedder
    )
    first = result.passages[0]
    large = first.model_copy(
        update={"chunk": first.chunk.model_copy(update={"title": "文" * 2000})}
    )
    result = result.model_copy(update={"passages": [large, result.passages[1]]})
    context = ContextAssembler(pipeline.embedder.tokenizer).assemble(
        result, ContextLimits(max_response_bytes=4096)
    )
    assert len(context.model_dump_json().encode("utf-8")) <= 4096
    assert context.omitted_count == 1
    assert context.passages[0].chunk == result.passages[1].chunk
