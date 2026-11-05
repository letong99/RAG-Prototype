"""Local document embeddings with a pinned Hub revision and matching fast tokenizer."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_service.schemas import EmbeddingSpec


class HuggingFaceTokenizer:
    def __init__(self, tokenizer: Any) -> None:
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for source offsets")
        self._tokenizer = tokenizer

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )
        return [(int(a), int(b)) for a, b in encoded["offset_mapping"] if b > a]

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False, truncation=False))


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model: str,
        revision: str,
        cache_dir: Path,
        device: str = "cpu",
        *,
        offline: bool = False,
    ) -> None:
        from huggingface_hub import HfApi
        from sentence_transformers import SentenceTransformer

        pinned = bool(re.fullmatch(r"[0-9a-f]{40}", revision))
        if offline and not pinned:
            raise ValueError("Offline mode requires a full model commit SHA via --model-revision")
        resolved = (
            revision if pinned else HfApi().model_info(model, revision=revision, timeout=30).sha
        )
        if not resolved:
            raise ValueError("Could not resolve the embedding model revision")
        self._model = SentenceTransformer(
            model,
            revision=resolved,
            cache_folder=str(cache_dir),
            device=device,
            trust_remote_code=False,
            local_files_only=offline,
        )
        self._tokenizer = HuggingFaceTokenizer(self._model.tokenizer)
        special_tokens = self._model.tokenizer.num_special_tokens_to_add(pair=False)
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError("Embedding dimension is unknown")
        self._spec = EmbeddingSpec(
            model=model,
            revision=resolved,
            dimension=dimension,
            max_tokens=int(self._model.max_seq_length) - special_tokens,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @property
    def tokenizer(self) -> HuggingFaceTokenizer:
        return self._tokenizer

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if any(self.tokenizer.count(text) > self.spec.max_tokens for text in texts):
            raise ValueError("Embedding input exceeds the model token budget")
        # Empty prompt prevents model defaults from silently consuming the input budget.
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False, prompt=""
        )
        return [[float(value) for value in vector] for vector in vectors]
