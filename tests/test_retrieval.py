"""Offline retrieval/ranking tests using embedded Qdrant and deterministic doubles."""

import json
from unittest.mock import Mock

import pytest
from conftest import document
from filelock import FileLock, Timeout

from rag_service.retrieval.models import RankerSpec, SearchRequest
from rag_service.retrieval.ranking import CrossEncoderReranker
from rag_service.retrieval.review import review
from rag_service.retrieval.service import RetrievalService


def setup_search(pipeline):
    pipeline.ingest([document("a"), document("b", "Vaccines prevent infections.")])
    return RetrievalService(pipeline.config.data_dir, pipeline.store)


def request(**kwargs):
    return SearchRequest(collection="default", query="backups", **kwargs)


def test_rerank_pool_then_select_and_preserve_provenance(pipeline):
    service = setup_search(pipeline)
    baseline = service.search(request(top_n=2), pipeline.embedder)
    ranker = Mock(spec=["score", "spec"])
    ranker.spec = RankerSpec(model="fake", revision="1", max_pair_tokens=512)

    def score(query, texts):
        with (
            pytest.raises(Timeout),
            FileLock(pipeline.config.data_dir / "retrieval.lock", timeout=0),
        ):
            pass
        assert len(texts) == 2
        return [-1.0, 3.0]

    ranker.score.side_effect = score
    result = service.search(request(top_n=1), pipeline.embedder, reranker=ranker)
    assert result.candidate_count == 2
    assert result.passages[0].chunk == baseline.passages[1].chunk
    assert result.passages[0].dense_rank == 2
    assert result.passages[0].rerank_score == 3
    assert result.index.ingestion_revision == 1


def test_document_filter_and_empty_filter(pipeline):
    service = setup_search(pipeline)
    result = service.search(request(document_ids=["a"]), pipeline.embedder)
    assert {p.chunk.document_id for p in result.passages} == {"a"}
    assert service.search(request(document_ids=[]), pipeline.embedder).passages == []


@pytest.mark.parametrize("scores", [[1.0], [float("nan"), 1.0], [1.0, float("inf")]])
def test_invalid_ranker_output_fails_closed(pipeline, scores):
    service = setup_search(pipeline)
    ranker = Mock()
    ranker.score.return_value = scores
    with pytest.raises(RuntimeError, match="invalid scores"):
        service.search(request(), pipeline.embedder, reranker=ranker)


@pytest.mark.parametrize("values", [{"query": " "}, {"top_n": 41}, {"candidate_limit": 101}])
def test_invalid_requests(values):
    with pytest.raises(ValueError):
        SearchRequest(**({"collection": "default", "query": "test"} | values))


def test_query_budget_before_search(pipeline, monkeypatch):
    service = setup_search(pipeline)
    search = Mock(side_effect=AssertionError("DB must not be queried"))
    monkeypatch.setattr(pipeline.store, "search", search)
    with pytest.raises(ValueError, match="token budget"):
        service.search(SearchRequest(collection="default", query="word " * 513), pipeline.embedder)
    search.assert_not_called()


def test_offline_human_review(pipeline, tmp_path, monkeypatch):
    result = setup_search(pipeline).search(request(top_n=2), pipeline.embedder)
    answers = iter(["invalid", "3", "direct evidence", "s"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    output = tmp_path / "judgments.jsonl"
    review(result, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["grade"] == 3
    assert rows[0]["chunk_id"] == result.passages[0].chunk.chunk_id
    assert rows[0]["document_version"] == result.passages[0].chunk.document_version


def test_cross_encoder_adapter_pair_budget_and_prediction_without_model_download():
    ranker = object.__new__(CrossEncoderReranker)
    ranker._spec = RankerSpec(model="fake", revision="1", max_pair_tokens=5)
    ranker._batch_size = 2
    ranker._model = Mock()
    ranker._model.tokenizer.return_value = {"input_ids": [1, 2, 3]}
    ranker._model.predict.return_value = [2.5]
    assert ranker.score("q", ["p"]) == [2.5]
    ranker._model.tokenizer.return_value = {"input_ids": [1] * 6}
    with pytest.raises(ValueError, match="pair exceeds"):
        ranker.score("q", ["long"])
    assert ranker._model.predict.call_count == 1
    assert ranker.score("q", []) == []
