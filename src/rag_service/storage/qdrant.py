"""One immutable Qdrant collection per ingestion generation."""

from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from rag_service.config import QdrantSettings
from rag_service.schemas import Chunk, IndexMetadata, SearchHit


class QdrantVectorStore:
    def __init__(
        self,
        *,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        if (path is None) == (url is None):
            raise ValueError("Specify exactly one of path or url")
        if path is not None and api_key is not None:
            raise ValueError("API keys are only supported for remote Qdrant")
        self._remote = url is not None
        self._client = (
            QdrantClient(url=url, api_key=api_key, timeout=timeout)
            if url
            else QdrantClient(path=str(path))
        )

    @classmethod
    def connect(cls, settings: QdrantSettings, data_dir: Path) -> "QdrantVectorStore":
        if settings.mode == "embedded":
            return cls(path=data_dir / "qdrant")
        return cls(
            url=str(settings.url),
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            timeout=settings.timeout_seconds,
        )

    @staticmethod
    def collection_name(generation_id: str) -> str:
        return f"rag_{generation_id}"

    def create(
        self, generation_id: str, dimension: int, *, metadata: IndexMetadata | None = None
    ) -> None:
        if metadata and (
            metadata.generation_id != generation_id or metadata.embedding.dimension != dimension
        ):
            raise ValueError("Index metadata does not match the generation or vector dimension")
        self._client.create_collection(
            collection_name=self.collection_name(generation_id),
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            metadata={"rag": metadata.model_dump()} if metadata else None,
        )
        if self._remote:
            # The server maintains these indexes and its vector/HNSW indexes.
            for field in ("document_id", "semantic_group_id"):
                self._client.create_payload_index(
                    collection_name=self.collection_name(generation_id),
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )

    @staticmethod
    def alias_name(collection: str) -> str:
        return f"rag_active_{collection}"

    def active_index(self, collection: str) -> IndexMetadata | None:
        alias = self.alias_name(collection)
        target = next(
            (
                item.collection_name
                for item in self._client.get_aliases().aliases
                if item.alias_name == alias
            ),
            None,
        )
        if target is None:
            return None
        # Resolve once to a physical generation. Later requests can pin it across alias switches.
        metadata = self._client.get_collection(target).config.metadata or {}
        index = IndexMetadata.model_validate(metadata.get("rag"))
        if index.collection != collection or target != self.collection_name(index.generation_id):
            raise ValueError("Qdrant alias and index metadata disagree")
        return index

    def publish(self, index: IndexMetadata) -> None:
        """Single-writer publication using Qdrant's atomic alias operations."""
        if self.count(index.generation_id) != index.chunks:
            raise RuntimeError("Stored vector count does not match staged chunk count")
        alias = self.alias_name(index.collection)
        operations: list[models.CreateAliasOperation | models.DeleteAliasOperation] = []
        if any(item.alias_name == alias for item in self._client.get_aliases().aliases):
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=self.collection_name(index.generation_id),
                    alias_name=alias,
                )
            )
        )
        self._client.update_collection_aliases(change_aliases_operations=operations)

    def search(
        self,
        generation_id: str,
        vector: list[float],
        *,
        limit: int = 6,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """Delegate nearest-neighbor search and filtering to Qdrant; no custom ranking loop."""
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100")
        query_filter = None
        if document_ids is not None:
            if not document_ids:
                return []
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=list(document_ids)),
                    )
                ]
            )
        response = self._client.query_points(
            collection_name=self.collection_name(generation_id),
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            SearchHit(score=point.score, chunk=Chunk.model_validate(point.payload))
            for point in response.points
        ]

    def write(
        self, generation_id: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(
            collection_name=self.collection_name(generation_id), points=points, wait=True
        )

    def count(self, generation_id: str) -> int:
        return int(
            self._client.count(
                collection_name=self.collection_name(generation_id), exact=True
            ).count
        )

    def close(self) -> None:
        self._client.close()
