"""Local file readers. PDFs become page documents to preserve precise page provenance."""

from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader

from rag_service.schemas import SourceDocument

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def read_directory(root: Path) -> Iterator[SourceDocument]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Input must be a directory: {root}")
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise ValueError("No supported .txt, .md, or .pdf documents found")
    for path in paths:
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source symlink escapes the input directory: {path}")
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            yield from _read_pdf(path, relative)
        else:
            yield SourceDocument(
                document_id=relative,
                text=path.read_text(encoding="utf-8"),
                source_uri=path.as_uri(),
                title=path.stem,
                metadata={"parser": "utf8-v1"},
            )


def _read_pdf(path: Path, relative: str) -> Iterator[SourceDocument]:
    reader = PdfReader(path)
    if not reader.pages:
        raise ValueError(f"PDF has no pages: {relative}")
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            raise ValueError(f"No text on {relative} page {number}; OCR may be required")
        yield SourceDocument(
            document_id=f"{relative}#page={number}",
            text=text,
            source_uri=path.as_uri(),
            title=path.stem,
            page=number,
            metadata={"parser": "pypdf-page-v1"},
        )
