import json

import pytest

from rag_service.ingestion.bioasq import export_rows, read_documents
from rag_service.ingestion.normalization import normalize
from rag_service.ingestion.validation import BIOASQ_DATASET_ID
from rag_service.schemas import SourceDocument


@pytest.mark.parametrize("text", ["nan", " NaN\n", "None", "null", None, float("nan"), " "])
def test_placeholder_is_rejected_and_missing_evidence_retained(tmp_path, text):
    output = tmp_path / "export"
    summary = export_rows(
        [{"id": "bad", "passage": text}, {"id": "good", "passage": "Valid evidence."}],
        [
            {
                "id": "q",
                "question": "Question?",
                "answer": "Answer",
                "relevant_passage_ids": ["bad", "good"],
            }
        ],
        output,
        revision="test",
    )
    assert summary["documents"] == summary["rejected_passages"] == 1
    assert summary["questions_with_missing_evidence"] == 1
    assert json.loads((output / "questions.jsonl").read_text())["missing_document_ids"] == ["bad"]
    assert json.loads((output / "rejected.jsonl").read_text())["document_id"] == "bad"


def test_old_exports_and_direct_pipeline_input_cannot_bypass_validation(tmp_path):
    source = SourceDocument(
        document_id="bad",
        text="nan",
        source_uri="test://bad",
        metadata={"dataset": BIOASQ_DATASET_ID},
    )
    path = tmp_path / "old.jsonl"
    path.write_text(source.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="placeholder"):
        list(read_documents(path))
    with pytest.raises(ValueError, match="placeholder"):
        normalize(source)
    assert normalize(source.model_copy(update={"text": "NaN is a missing numerical value."})).text
