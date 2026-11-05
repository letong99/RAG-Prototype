"""Thin command-line composition layer; domain modules do not depend on argparse."""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from rag_service.config import IngestionConfig, QdrantSettings
from rag_service.ingestion.bioasq import prepare_bioasq, read_documents
from rag_service.ingestion.files import read_directory
from rag_service.storage.manifest import Manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Versioned RAG document ingestion")
    root.add_argument("--data-dir", type=Path, default=Path(".data"))
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-bioasq", help="Download passages and separate QA labels")
    prepare.add_argument("--revision", default="main")
    prepare.add_argument(
        "--max-passages", type=int, help="Smoke-test subset; unsuitable for evaluation"
    )
    ingest = commands.add_parser(
        "ingest", help="Replace a collection with a complete source snapshot"
    )
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory", type=Path)
    source.add_argument("--documents", type=Path, help="SourceDocument JSONL, not QA labels")
    ingest.add_argument("--collection", default="default")
    _chunking_arguments(ingest)
    _storage_arguments(ingest)
    ingest.add_argument("--allow-empty", action="store_true", help="Permit clearing the collection")
    preview = commands.add_parser(
        "preview-chunks", help="Inspect one text file without indexing it"
    )
    preview.add_argument("--file", type=Path, required=True)
    preview.add_argument("--output", type=Path)
    preview.set_defaults(collection="preview", qdrant_url=None)
    _chunking_arguments(preview)
    status = commands.add_parser("status", help="Show active generation and ingestion history")
    status.add_argument("--collection", default="default")
    remote_status = commands.add_parser(
        "storage-status", help="Inspect Qdrant's active index without a model"
    )
    remote_status.add_argument("--collection", default="default")
    _storage_arguments(remote_status)
    return root


def _storage_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--storage", choices=["remote", "embedded"], default="remote")
    command.add_argument("--qdrant-url", help="Qdrant server URL; defaults to QDRANT_URL")
    command.add_argument("--qdrant-timeout", type=int, default=30)


def _storage_settings(args: argparse.Namespace) -> QdrantSettings:
    return QdrantSettings.model_validate(
        {
            "mode": args.storage,
            "url": args.qdrant_url or os.environ.get("QDRANT_URL"),
            "api_key": os.environ.get("QDRANT_API_KEY"),
            "timeout_seconds": args.qdrant_timeout,
        }
    )


def _chunking_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--chunk-tokens", type=int, default=300, help="Maximum tokens, not a target size"
    )
    command.add_argument("--overlap-tokens", type=int, default=0)
    command.add_argument("--chunking-strategy", choices=["semantic", "token"], default="semantic")
    command.add_argument("--semantic-percentile", type=float, default=85)
    command.add_argument("--semantic-min-distance", type=float, default=0.1)
    command.add_argument("--semantic-window-sentences", type=int, default=2)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    command.add_argument("--model-revision", default="main")
    command.add_argument("--device", default="cpu")
    command.add_argument(
        "--offline", action="store_true", help="Use cached model at a pinned commit"
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    # Keep dependency caches inside the selected data directory unless explicitly configured.
    os.environ.setdefault("HF_HOME", str(args.data_dir.resolve() / "cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.command == "prepare-bioasq":
        output = prepare_bioasq(
            args.data_dir, revision=args.revision, max_passages=args.max_passages
        )
        return {"output": str(output), "summary": json.loads((output / "summary.json").read_text())}
    if args.command == "storage-status":
        from rag_service.storage.qdrant import QdrantVectorStore

        store = QdrantVectorStore.connect(_storage_settings(args), args.data_dir)
        try:
            active = store.active_index(args.collection)
            return {"active": active.model_dump() if active else None}
        finally:
            store.close()

    manifest = Manifest(args.data_dir / "manifest.sqlite3")
    try:
        if args.command == "status":
            return {
                "active": manifest.active(args.collection),
                "history": manifest.generations(args.collection),
            }
        return _process(args, manifest)
    finally:
        manifest.close()


def _process(args: argparse.Namespace, manifest: Manifest) -> dict[str, object]:
    from rag_service.chunking.preview import preview_file
    from rag_service.embeddings.cached import CachedEmbedder
    from rag_service.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from rag_service.ingestion.pipeline import IngestionPipeline
    from rag_service.storage.qdrant import QdrantVectorStore

    # Validate the target before loading or downloading an embedding model.
    storage_settings = None if args.command == "preview-chunks" else _storage_settings(args)
    config = IngestionConfig(
        data_dir=args.data_dir,
        collection=args.collection,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        chunking_strategy=args.chunking_strategy,
        semantic_percentile=args.semantic_percentile,
        semantic_min_distance=args.semantic_min_distance,
        semantic_window_sentences=args.semantic_window_sentences,
        batch_size=args.batch_size,
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        offline=args.offline,
    )
    embedder = SentenceTransformerEmbedder(
        config.model_name,
        config.model_revision,
        config.data_dir / "cache" / "models",
        config.device,
        offline=config.offline,
    )
    if args.command == "preview-chunks":
        return preview_file(
            args.file, config, CachedEmbedder(embedder, manifest, config.batch_size), args.output
        )
    assert storage_settings is not None
    store = QdrantVectorStore.connect(storage_settings, config.data_dir)
    try:
        sources = (
            read_directory(args.directory) if args.directory else read_documents(args.documents)
        )
        result = IngestionPipeline(config, embedder, store, manifest).ingest(
            sources, allow_empty=args.allow_empty
        )
        return result.model_dump()
    finally:
        store.close()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parser().parse_args(argv)
    try:
        result = run(arguments)
    except Exception as error:
        logging.getLogger(__name__).error("%s: %s", type(error).__name__, error)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
