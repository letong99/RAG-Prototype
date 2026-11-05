"""Validated, batched embeddings shared by semantic segmentation and final indexing."""

import math
from collections.abc import Sequence
from itertools import batched

from rag_service.ports import Embedder, Tokenizer
from rag_service.schemas import EmbeddingSpec
from rag_service.storage.manifest import Manifest


class CachedEmbedder:
    def __init__(self, embedder: Embedder, manifest: Manifest, batch_size: int) -> None:
        self._embedder = embedder
        self._manifest = manifest
        self.batch_size = batch_size
        self.computed_texts = 0
        self.reused_texts = 0

    @property
    def spec(self) -> EmbeddingSpec:
        return self._embedder.spec

    @property
    def tokenizer(self) -> Tokenizer:
        return self._embedder.tokenizer

    def reset_stats(self) -> None:
        self.computed_texts = self.reused_texts = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        namespace = self.spec.cache_namespace
        available: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):
            if self.tokenizer.count(text) > self.spec.max_tokens:
                raise ValueError("Embedding input exceeds the model token budget")
            cached = self._manifest.cached_vector(namespace, text)
            if cached is None:
                missing.append(text)
            else:
                self._validate(cached)
                available[text] = cached
        for batch in batched(missing, self.batch_size):
            vectors = self._embedder.embed(batch)
            if len(vectors) != len(batch):
                raise ValueError("Embedder returned an incorrect number of vectors")
            for vector in vectors:
                self._validate(vector)
            new_vectors = dict(zip(batch, vectors, strict=True))
            self._manifest.cache_vectors(namespace, new_vectors)
            available.update(new_vectors)
            self.computed_texts += len(batch)
        self.reused_texts += len(texts) - len(missing)
        return [available[text] for text in texts]

    def _validate(self, vector: list[float]) -> None:
        if len(vector) != self.spec.dimension:
            raise ValueError("Embedding dimension does not match the index configuration")
        if not all(math.isfinite(value) for value in vector) or not any(vector):
            raise ValueError("Embedding must be finite and nonzero")
