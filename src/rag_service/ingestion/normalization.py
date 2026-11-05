"""Conservative normalization with both raw and normalized text retained."""

from rag_service.schemas import Document, SourceDocument, fingerprint

NORMALIZER_VERSION = "whitespace-v1"


def normalize(source: SourceDocument) -> Document:
    text = source.text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    if not text:
        raise ValueError(f"Document {source.document_id!r} contains no usable text")
    content_hash = fingerprint(text)
    version = fingerprint({**source.model_dump(), "normalizer": NORMALIZER_VERSION})
    return Document(
        **{**source.model_dump(), "text": text},
        raw_text=source.text,
        content_hash=content_hash,
        version=version,
    )
