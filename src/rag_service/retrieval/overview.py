"""Factual collection inventory, separate from query-selected evidence."""

from rag_service.schemas import IndexMetadata, Record


class DatasetDescription(Record):
    index: IndexMetadata
    documents: int
    chunks: int
    languages: list[str]
    scope: str = "Full published collection inventory; no semantic summary has been generated."
