# Python RAG service

Foundation and ingestion implementation for [the project plan](RAG_IMPLEMENTATION_PLAN.md).

Implemented lifecycle: [incremental ingestion, versioning, and active/fallback lifecycle](docs/INCREMENTAL_INGESTION.md).
Normal ingestion applies document patches; full generation rebuilds are explicit.

Implemented: typed configuration and records, text/Markdown/PDF parsing, semantic chunking with
LangChain recursive character splitting,
local embeddings, persistent Qdrant storage, SQLite manifest and embedding cache, atomic publication
of corpus generations, and a BioASQ dataset adapter. The vector query adapter delegates to Qdrant's
SDK. A guarded dense retrieval context now embeds queries and verifies versions. Hybrid retrieval,
reranking, MCP, and answer generation remain later milestones.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --locked
uv run rag --help
```

On the initial development machine, uv is bootstrapped at `.tools/bin/uv`; substitute that path for
`uv` if it is not installed on your PATH. The first ingestion downloads the embedding model.
To activate the environment directly, run `source .venv/bin/activate`; commands such as `rag` and
`pytest` then work without the `uv run` prefix. This project currently targets Python 3.12.
Intel macOS receives platform-specific PyTorch, NumPy, Arrow, and Transformers compatibility constraints.

## Ingest local documents

Set `QDRANT_URL` and `QDRANT_API_KEY` in the process environment to target Qdrant Cloud or another
Qdrant server. See [remote storage and deployment](docs/REMOTE_STORAGE.md) and `.env.example`.
The remote service stores vectors **and chunk text**; it does not read this machine's files during queries.

```sh
uv run rag storage-status --collection example
uv run rag ingest --directory tests/fixtures/documents --collection example \
  --available-storage-bytes "$QDRANT_FREE_BYTES"
uv run rag status --collection example
```

Defaults: semantic boundaries with a 300-token **ceiling**, zero overlap, BGE-small-en-v1.5 on CPU, batch size 32,
and **remote Qdrant**. A missing remote URL fails before model loading. `--qdrant-url` overrides
`QDRANT_URL`; keys are read only from the environment. Embedded Qdrant at `.data/qdrant` remains an
explicit development/test option: `--storage embedded`. It is not the production server's indexing engine.

All CLI global options precede the command:

```sh
uv run rag --data-dir .data-experiment ingest --directory ./documents --collection papers \
  --available-storage-bytes "$QDRANT_FREE_BYTES"
```

**Ingestion is incremental by default.** Supply new or changed documents; missing IDs are retained.
Unchanged normalized documents skip chunking, embedding, and vector writes. Citation-only metadata
changes update payloads without embedding. Supply `--delete-document-id ID` for explicit deletions.
Deleting the last document additionally requires `--allow-empty`.

Set `QDRANT_FREE_BYTES` above to measured free disk space from the hosting dashboard. Remote bootstrap
and full rebuilds require `--available-storage-bytes`; this value is checked against a conservative
staged-size estimate, not treated as an automatic measurement. It is optional for ordinary patches; when supplied, the staged additional-storage estimate is also checked before mutation.
The CLI does not automatically load `.env`; export its variables in your shell or deployment environment.

```sh
# Patch and explicit deletion; repeat the deletion option for multiple IDs.
uv run rag ingest --documents changed-documents.jsonl --collection papers --delete-document-id old-id

# Complete input set, explicit new generation (required for model/chunking changes).
uv run rag ingest --documents all-documents.jsonl --collection papers --full-rebuild \
  --available-storage-bytes "$QDRANT_FREE_BYTES"

# Inspect maintenance state or replay a failed operation without loading a model.
uv run rag status --collection papers
uv run rag recover --collection papers

# Explicitly restore the older inactive snapshot; newer knowledge may be lost.
uv run rag rollback --collection papers
uv run rag cleanup --collection papers
```

Bootstrap creates one active collection. Full rebuilds retain the former active as the inactive
fallback, then retire the older fallback. The fallback does not receive incremental changes.
Interrupted mutations keep retrieval paused until `recover` or explicit `rollback` succeeds.
A preparation failure before remote mutation leaves the existing index available.

## Preview and configure chunking

```sh
uv run rag preview-chunks --file tests/fixtures/chunking/topics.txt --output .data/preview.json
```

Preview computes sentence-context embeddings but does not create a vector index or publish a
generation. The default strategy detects topic changes between adjacent sentence contexts, then
uses LangChain's `RecursiveCharacterTextSplitter` to fit oversized semantic groups within the
embedding model's input limit. It preserves Markdown section boundaries, source offsets, and
semantic group IDs. See [chunking design and the measured preview](docs/CHUNKING.md).

Configuration: `--semantic-percentile 85`, `--semantic-min-distance 0.1`, and
`--semantic-window-sentences 2`. These are uncalibrated starting values. `--chunk-tokens 300` is a
maximum, not a target size. Optional overlap copies complete sentences only within a semantic
group when both the overlap and total token budgets permit it. It never crosses a detected topic
boundary. The old `--chunking-strategy token` remains available as an evaluation baseline.

Semantic embeddings and final chunk embeddings share a validated, batched cache. CLI results report
their computation counts separately. Unchanged documents skip semantic segmentation entirely.
Changing chunking settings requires an explicit `--full-rebuild`.

For the already downloaded model, preview can run entirely offline:

```sh
uv run rag preview-chunks --file tests/fixtures/chunking/topics.txt \
  --offline --model-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
```

## Latest hosted smoke result

A small real-server run completed with **21 valid passages / 60 chunks**. The unchanged rerun skipped
all documents, and three selected questions retrieved labelled evidence. The run exposed placeholder
passages in the source export; these are now rejected. Use the corrected export at
`.data/datasets/bioasq/validated-224a87f/` or re-export with the updated adapter.
See [the full smoke-test report](docs/SMOKE_TEST_RESULTS.md), including data-quality and evaluation limits.

## Prepare BioASQ

```sh
uv run rag prepare-bioasq
```

The command loads both configurations from `rag-datasets/rag-mini-bioasq`:

- `text-corpus`, split `passages`: source documents to index.
- `question-answer-passages`, split `test`: questions, reference answers, and relevant passage IDs.

The returned `output` directory contains:

- `documents.jsonl`: validated passage documents.
- `questions.jsonl`: evaluation labels, including missing-evidence IDs.
- `rejected.jsonl`: passage IDs with missing or empty text.
- `summary.json`: revision, attribution, counts, and partial-corpus flag; present only after export completes.

Use the actual output path printed by the command:

```sh
uv run rag ingest --documents .data/datasets/bioasq/EXPORT_ID/documents.jsonl --collection bioasq \
  --available-storage-bytes "$QDRANT_FREE_BYTES"
```

For a small ingestion smoke test, run `prepare-bioasq --max-passages 100`. It still downloads the
dataset but exports only the first 100 valid passages. Its QA file retains all questions and marks
unavailable evidence; do not use this subset for retrieval quality claims. Missing evidence is
reported even in the full corpus. QA answers are never indexed as source evidence.

The dataset revision is resolved to a commit before loading. Pass `--revision COMMIT` to reproduce
an export. The embedding model is similarly resolved and recorded; use `--model-revision COMMIT`
to reproduce it. The initial model is an English baseline and has not been evaluated for biomedical
retrieval quality.

Dataset attribution: [rag-datasets/rag-mini-bioasq](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq),
derived from BioASQ Task 11b, distributed under [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/).
The adapter preserves the requested dataset rather than silently replacing it with another version.

## Layout and persistence

```text
src/rag_service/
  config.py, schemas.py, ports.py  # validated contracts and narrow interfaces
  ingestion/                     # source adapters, normalization, orchestration
  chunking/                      # semantic boundaries, LangChain fallback, previews, source spans
  embeddings/                    # Sentence Transformers adapter and shared embedding cache
  storage/                       # SQLite embedding/audit cache and Qdrant adapter
  retrieval/                     # guarded query embedding and integrity checks
  cli.py                         # dependency composition and commands
tests/                           # offline correctness and local Qdrant integration tests
```

Ingestion writes durable, checksummed replay artifacts and document manifests in
`.data/operations/ID/`. `.data/lifecycle.sqlite3` holds coordinator identity, counters, publication
pointers, and the operation journal; `.data/manifest.sqlite3` retains the embedding cache and legacy
run audit. Back up both databases and their artifacts together. Do not delete the lifecycle ledger
and attempt to adopt an existing remote index: ownership/publication mismatch is rejected.
Legacy indexes require a new corpus name or a separately designed migration.

Chunk offsets refer to normalized text. PDF pages retain page provenance; Markdown sections retain
headings. Oversized code/tables may split. Titles and citation metadata are not embedded.

All readers and writers must use **one coordinator directory on one host**. `RetrievalService.retrieve`
is a context manager: keep it open through reranking/context assembly. It checks the exact model,
published revision, document versions, and payload identities against the committed manifest. It
serializes retrieval requests using the same gate that ingestion drains before mutation. Direct
`store.search` is a low-level SDK adapter and bypasses those guarantees. Multiple independent MCP
hosts require shared database coordination before deployment; local file locks are insufficient.

```python
from rag_service.retrieval.service import RetrievalService

service = RetrievalService(data_dir, store)
with service.retrieve("papers", "What is the backup policy?", matching_embedder) as (index, hits):
    # Build context inside this block. Future result caches must include index.ingestion_revision.
    context = "\n\n".join(hit.chunk.text for hit in hits)
```

Every incremental mutation verifies the complete payload/ID set (a deliberate prototype tradeoff).
Full builds prepare embeddings while reads remain available, then hold the gate during remote upload,
validation, and publication. There is no distributed transaction between SQLite and Qdrant: durable
replay and fail-closed retrieval handle interruptions. Cleanup removes retired remote collections;
local audit/artifact retention and secure physical erasure are not automated.

Normalized source records and chunk JSONL can feed the future sparse/BM25 encoder. Qdrant will own
the sparse index and hybrid rank fusion as well. Those stages and the remote MCP service are not
implemented yet. Only dense vectors and payloads are currently published to the configured Qdrant service.

## Verification

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Default tests use deterministic fake embeddings to verify lifecycle behavior and real embedded
Qdrant to test persistence. They do not download models or datasets. An opt-in real-model test can
be run with `RAG_TEST_MODEL=1 uv run pytest -m model`; it requires network access or cached assets.
The opt-in `remote` test requires a separate `RAG_TEST_QDRANT_URL`; it creates and cleans up a tiny
test collection. It has not been run against a hosted cluster. See the remote-storage document.

Resolving a symbolic Hub revision such as `main` requires network access. With cached weights,
`--offline --model-revision FULL_COMMIT_SHA` avoids network access.
See [the validation report](docs/INGESTION_VALIDATION.md) for the verified dataset revision and results.

Dependency caches and downloaded data are ignored by Git. Runtime source data is not published with
the code. Run status inspection without loading the embedding model.

Retrieval and optional CrossEncoder reranking are available with an offline human-review
workflow. See [retrieval usage and evaluation](docs/RETRIEVAL.md). Wait for the full ingestion
to finish before exporting results from the hosted collection.

### Context assembly and optional answers

The Python service exposes `retrieve_context`, `answer_question` (with an injected generation
provider), and `describe_dataset` for exact published inventory. Context includes stable citations,
versioned source locations, token/byte limits, and omission information. See
[context, generation, and dataset overview design](docs/CONTEXT_AND_GENERATION.md) for usage,
provider responsibilities, and the recommended hierarchical approach to corpus-wide summaries.

### Local MCP access

Run `rag-mcp` as a stdio MCP server from Codex or Claude Code. Its `retrieve_context` tool returns
assembled, cited evidence plus response latency and query/result token estimates, and saves query
metrics to `.data/query_metrics.jsonl`. See [local MCP setup](docs/LOCAL_MCP.md) for installation,
client registration commands, limits, and metric definitions.
