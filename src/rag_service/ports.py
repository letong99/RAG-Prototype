"""Small interfaces at model and storage boundaries."""

from collections.abc import Iterator, Sequence
from typing import Protocol

from rag_service.schemas import Chunk, Document, EmbeddingSpec, IndexMetadata


class Chunker(Protocol):
    version: str

    def chunks(self, document: Document) -> Iterator[Chunk]: ...


class Tokenizer(Protocol):
    def offsets(self, text: str) -> list[tuple[int, int]]: ...

    def count(self, text: str) -> int: ...


class Embedder(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    @property
    def tokenizer(self) -> Tokenizer: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def create(
        self, generation_id: str, dimension: int, *, metadata: IndexMetadata | None = None
    ) -> None: ...

    def write(
        self, generation_id: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None: ...

    def count(self, generation_id: str) -> int: ...

    def active_index(self, collection: str) -> IndexMetadata | None: ...

    def publish(self, index: IndexMetadata) -> None: ...
