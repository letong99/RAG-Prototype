"""Import corpus passages separately from QA labels; retain rejected rows and missing evidence."""

import ast
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from rag_service.ingestion.validation import BIOASQ_DATASET_ID, missing_passage
from rag_service.schemas import EvaluationQuestion, SourceDocument

DATASET_ID = BIOASQ_DATASET_ID


def passage_ids(value: object) -> list[str]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("relevant_passage_ids must be a list or a serialized list")
    if any(not isinstance(item, (str, int)) or isinstance(item, bool) for item in value):
        raise ValueError("Passage IDs must be strings or integers")
    return list(dict.fromkeys(str(item) for item in value))


def export_rows(
    corpus: Iterable[Mapping[str, Any]],
    questions: Iterable[Mapping[str, Any]],
    output: Path,
    *,
    revision: str,
    max_passages: int | None = None,
) -> dict[str, Any]:
    """Export to a new directory. A summary file marks successful completion."""
    if max_passages is not None and max_passages < 1:
        raise ValueError("max_passages must be positive")
    output.mkdir(parents=True, exist_ok=False)
    accepted: set[str] = set()
    seen: set[str] = set()
    rejected = question_count = missing_count = 0
    with (
        (output / "documents.jsonl").open("w", encoding="utf-8") as documents,
        (output / "rejected.jsonl").open("w", encoding="utf-8") as failures,
    ):
        for row in corpus:
            if max_passages is not None and len(accepted) >= max_passages:
                break
            raw_id = row["id"]
            if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool):
                raise ValueError("Invalid corpus passage ID")
            document_id = str(raw_id)
            if document_id in seen:
                raise ValueError(f"Duplicate corpus passage ID: {document_id}")
            seen.add(document_id)
            text = row["passage"]
            if missing_passage(text):
                failures.write(
                    json.dumps({"document_id": document_id, "reason": "missing_text"}) + "\n"
                )
                rejected += 1
                continue
            document = SourceDocument(
                document_id=document_id,
                text=text,
                source_uri=f"https://pubmed.ncbi.nlm.nih.gov/{document_id}/",
                title=f"BioASQ passage {document_id}",
                metadata={"dataset": DATASET_ID, "revision": revision, "license": "CC-BY-2.5"},
            )
            documents.write(document.model_dump_json() + "\n")
            accepted.add(document_id)
    with (output / "questions.jsonl").open("w", encoding="utf-8") as labels:
        for row in questions:
            relevant = passage_ids(row["relevant_passage_ids"])
            missing = [document_id for document_id in relevant if document_id not in accepted]
            question = EvaluationQuestion(
                query_id=str(row["id"]),
                query=row["question"],
                reference_answer=row["answer"],
                relevant_document_ids=relevant,
                missing_document_ids=missing,
            )
            labels.write(question.model_dump_json() + "\n")
            question_count += 1
            missing_count += bool(missing)
    summary = {
        "dataset": DATASET_ID,
        "revision": revision,
        "documents": len(accepted),
        "rejected_passages": rejected,
        "questions": question_count,
        "questions_with_missing_evidence": missing_count,
        "max_passages": max_passages,
        "partial_corpus": max_passages is not None,
        "license": "CC-BY-2.5",
        "attribution": "rag-datasets/rag-mini-bioasq, derived from BioASQ Task 11b",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def prepare_bioasq(
    data_dir: Path, *, revision: str = "main", max_passages: int | None = None
) -> Path:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    resolved = HfApi().dataset_info(DATASET_ID, revision=revision, timeout=30).sha
    if not resolved:
        raise ValueError("Could not resolve the dataset revision")
    cache = str(data_dir / "cache" / "datasets")
    corpus = load_dataset(
        DATASET_ID, "text-corpus", split="passages", revision=resolved, cache_dir=cache
    )
    questions = load_dataset(
        DATASET_ID, "question-answer-passages", split="test", revision=resolved, cache_dir=cache
    )
    output = data_dir / "datasets" / "bioasq" / uuid4().hex
    export_rows(corpus, questions, output, revision=resolved, max_passages=max_passages)
    return output


def read_documents(path: Path) -> Iterator[SourceDocument]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                document = SourceDocument.model_validate_json(line)
                if document.metadata.get("dataset") == DATASET_ID and missing_passage(
                    document.text
                ):
                    raise ValueError(
                        "BioASQ passage contains placeholder text; re-export the dataset"
                    )
                yield document
            except ValueError as error:
                raise ValueError(f"Invalid document at {path}:{line_number}: {error}") from error
