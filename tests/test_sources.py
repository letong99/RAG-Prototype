import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from rag_service.ingestion.bioasq import export_rows, passage_ids, read_documents
from rag_service.ingestion.files import read_directory


def test_bioasq_separates_labels_and_reports_missing_text(tmp_path: Path) -> None:
    output = tmp_path / "export"
    summary = export_rows(
        [
            {"id": 1, "passage": "Evidence passage."},
            {"id": 2, "passage": None},
            {"id": 3, "passage": float("nan")},
        ],
        [
            {
                "id": 0,
                "question": "A question?",
                "answer": "Secret reference answer",
                "relevant_passage_ids": "[1, 2]",
            }
        ],
        output,
        revision="test-sha",
    )
    docs = list(read_documents(output / "documents.jsonl"))
    assert [doc.document_id for doc in docs] == ["1"]
    assert "Secret reference answer" not in (output / "documents.jsonl").read_text()
    assert summary["rejected_passages"] == 2
    label = json.loads((output / "questions.jsonl").read_text())
    assert label["missing_document_ids"] == ["2"]
    assert docs[0].metadata["revision"] == "test-sha"


def test_ids_are_safely_parsed() -> None:
    assert passage_ids("[1, 2, 1]") == ["1", "2"]
    assert passage_ids(["1", 2]) == ["1", "2"]
    with pytest.raises(ValueError):
        passage_ids("__import__('os').system('false')")


def test_dataset_subset_marks_missing_evidence(tmp_path: Path) -> None:
    summary = export_rows(
        [{"id": 1, "passage": "one"}, {"id": 2, "passage": "two"}],
        [{"id": 0, "question": "which?", "answer": "two", "relevant_passage_ids": [2]}],
        tmp_path / "export",
        revision="sha",
        max_passages=1,
    )
    assert summary["partial_corpus"]
    assert summary["documents"] == 1
    assert summary["questions_with_missing_evidence"] == 1


def test_text_and_markdown_sources(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("# βeta", encoding="utf-8")
    documents = list(read_directory(tmp_path))
    assert [doc.document_id for doc in documents] == ["a.txt", "b.md"]
    assert documents[1].text == "# βeta"


def test_empty_pdf_page_fails_explicitly(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(tmp_path / "blank.pdf")
    with pytest.raises(ValueError, match="OCR"):
        list(read_directory(tmp_path))


def test_symlinks_cannot_escape_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    (source / "link.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(ValueError, match="escapes"):
        list(read_directory(source))
