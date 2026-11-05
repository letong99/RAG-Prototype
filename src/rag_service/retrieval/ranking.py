"""Sentence Transformers cross-encoder adapter; scores are logits, not confidence."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rag_service.retrieval.models import RankerSpec


class Reranker(Protocol):
    @property
    def spec(self) -> RankerSpec: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        cache_dir: Path,
        *,
        model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        revision: str = "main",
        device: str = "cpu",
        offline: bool = False,
        batch_size: int = 16,
        max_pair_tokens: int = 512,
    ) -> None:
        if batch_size < 1 or max_pair_tokens < 1:
            raise ValueError("Batch size and pair token budget must be positive")
        pinned = bool(re.fullmatch(r"[0-9a-f]{40}", revision))
        if offline and not pinned:
            raise ValueError("Offline reranking requires a full model commit SHA")
        from huggingface_hub import HfApi
        from sentence_transformers import CrossEncoder
        from torch.nn import Identity

        resolved = revision if pinned else HfApi().model_info(model, revision=revision).sha
        if not resolved:
            raise ValueError("Could not resolve reranker revision")
        self._model = CrossEncoder(
            model,
            revision=resolved,
            cache_folder=str(cache_dir),
            device=device,
            local_files_only=offline,
            trust_remote_code=False,
            max_length=max_pair_tokens,
            activation_fn=Identity(),
        )
        if self._model.num_labels != 1:
            raise ValueError("Reranking requires a single-score cross encoder")
        tokenizer_limit = int(self._model.tokenizer.model_max_length)
        self._spec = RankerSpec(
            model=model,
            revision=resolved,
            max_pair_tokens=min(max_pair_tokens, tokenizer_limit),
        )
        self._batch_size = batch_size

    @property
    def spec(self) -> RankerSpec:
        return self._spec

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        # Reject instead of silently discarding the end of biomedical evidence.
        for passage in passages:
            tokens = self._model.tokenizer(query, passage, truncation=False)["input_ids"]
            if len(tokens) > self.spec.max_pair_tokens:
                raise ValueError("Query/passage pair exceeds reranker token budget")
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            prompt="",
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]
