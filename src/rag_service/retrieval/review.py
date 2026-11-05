"""Export retrieval runs, then collect human judgments without accessing Qdrant."""

import argparse
import os
from pathlib import Path

from pydantic import Field

from rag_service.retrieval.models import SearchRequest, SearchResult
from rag_service.schemas import Record, fingerprint


class Judgment(Record):
    run_id: str
    query: str
    chunk_id: str
    document_version: str
    rank: int
    grade: int = Field(ge=0, le=3)
    note: str = ""


def review(result: SearchResult, destination: Path) -> None:
    """Append judgments; skipped passages remain unjudged, never assumed irrelevant."""
    run_id = fingerprint(result.model_dump(mode="json"))
    print(f"Query: {result.request.query}")
    print("Grades: 0 irrelevant, 1 related, 2 partially answers, 3 directly answers; s skip")
    with destination.open("a", encoding="utf-8") as stream:
        for passage in result.passages:
            chunk = passage.chunk
            print(f"\nRank {passage.rank} | {chunk.source_uri} | {chunk.document_id}")
            print(chunk.text)
            while True:
                value = input("Grade [0-3/s]: ").strip().lower()
                if value in {"0", "1", "2", "3", "s"}:
                    break
            if value == "s":
                continue
            judgment = Judgment(
                run_id=run_id,
                query=result.request.query,
                chunk_id=chunk.chunk_id,
                document_version=chunk.document_version,
                rank=passage.rank,
                grade=int(value),
                note=input("Note (optional): "),
            )
            stream.write(judgment.model_dump_json() + "\n")
            stream.flush()


def export(args: argparse.Namespace) -> None:
    # Imports are deferred: grading a saved run never loads a model or a DB client.
    from rag_service.config import QdrantSettings
    from rag_service.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from rag_service.ingestion.lifecycle import Ledger
    from rag_service.retrieval.ranking import CrossEncoderReranker
    from rag_service.retrieval.service import RetrievalService
    from rag_service.storage.qdrant import QdrantVectorStore

    request = SearchRequest(
        collection=args.collection,
        query=args.query,
        top_n=args.top_n,
        candidate_limit=args.candidates,
        document_ids=args.document_id,
    )
    ledger = Ledger(args.data_dir)
    try:
        with ledger.admission(args.collection) as index:
            spec = index.embedding
    finally:
        ledger.close()
    embedder = SentenceTransformerEmbedder(
        spec.model,
        spec.revision,
        args.cache_dir,
        offline=args.offline,
    )
    reranker = (
        CrossEncoderReranker(
            args.cache_dir,
            revision=args.reranker_revision,
            offline=args.offline,
        )
        if args.rerank
        else None
    )
    settings = QdrantSettings.model_validate(
        {
            "mode": args.storage,
            "url": os.environ.get("QDRANT_URL") if args.storage == "remote" else None,
            "api_key": os.environ.get("QDRANT_API_KEY") if args.storage == "remote" else None,
        }
    )
    store = QdrantVectorStore.connect(settings, args.data_dir)
    try:
        result = RetrievalService(args.data_dir, store).search(request, embedder, reranker=reranker)
        # Exclusive creation prevents overwriting a previously judged run.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(result.model_dump_json(indent=2) + "\n")
    finally:
        store.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("export", help="Query the active index and save top-N evidence")
    fetch.add_argument("--data-dir", type=Path, required=True)
    fetch.add_argument("--collection", required=True)
    fetch.add_argument("--query", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--top-n", type=int, default=6)
    fetch.add_argument("--candidates", type=int, default=40)
    fetch.add_argument("--document-id", action="append")
    fetch.add_argument("--storage", choices=["remote", "embedded"], default="remote")
    fetch.add_argument("--cache-dir", type=Path, default=Path(".data/cache/models"))
    fetch.add_argument("--rerank", action="store_true")
    fetch.add_argument("--reranker-revision", default="main")
    fetch.add_argument("--offline", action="store_true")
    grade = commands.add_parser("grade", help="Grade saved results offline; no DB/model access")
    grade.add_argument("--run", type=Path, required=True)
    grade.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args)
    else:
        review(SearchResult.model_validate_json(args.run.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
