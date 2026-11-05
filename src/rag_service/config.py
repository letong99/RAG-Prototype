"""Validated runtime configuration, independent of the CLI."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, model_validator


class IngestionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_dir: Path = Path(".data")
    collection: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    chunk_tokens: int = Field(default=300, ge=8, le=8192)
    overlap_tokens: int = Field(default=0, ge=0)
    chunking_strategy: Literal["semantic", "token"] = "semantic"
    semantic_percentile: float = Field(default=85, ge=0, le=100)
    semantic_min_distance: float = Field(default=0.1, ge=0, le=2)
    semantic_window_sentences: int = Field(default=2, ge=1, le=10)
    batch_size: int = Field(default=32, ge=1, le=512)
    model_name: str = "BAAI/bge-small-en-v1.5"
    model_revision: str = "main"
    device: str = "cpu"
    offline: bool = False

    @model_validator(mode="after")
    def validate_overlap(self) -> "IngestionConfig":
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        return self


class QdrantSettings(BaseModel):
    """Remote storage by default; embedded mode is an explicit development choice."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    mode: Literal["remote", "embedded"] = "remote"
    url: HttpUrl | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    @model_validator(mode="after")
    def validate_connection(self) -> "QdrantSettings":
        if self.mode == "remote" and self.url is None:
            raise ValueError(
                "Set QDRANT_URL or --qdrant-url; use --storage embedded only for development"
            )
        if self.mode == "embedded" and (self.url is not None or self.api_key is not None):
            raise ValueError("Embedded mode cannot be combined with remote Qdrant credentials")
        return self
