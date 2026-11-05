# Python RAG service

Foundation and ingestion implementation for [the project plan](RAG_IMPLEMENTATION_PLAN.md).

Accepted next design: [incremental ingestion, versioning, and active/fallback lifecycle](docs/INCREMENTAL_INGESTION.md).
This design is documented but not implemented; the ingestion commands below still use full snapshots.

Implemented: typed configuration and records, text/Markdown/PDF parsing, semantic chunking with
LangChain recursive character splitting,
local embeddings, persistent Qdrant storage, SQLite manifest and embedding cache, atomic publication
of corpus generations, and a BioASQ dataset adapter. The vector query adapter delegates to Qdrant's
SDK. Query-text orchestration, hybrid retrieval, reranking, MCP, and answer generation are later milestones.

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
uv run rag ingest --directory tests/fixtures/documents --collection example
uv run rag status --collection example
```

Defaults: semantic boundaries with a 300-token **ceiling**, zero overlap, BGE-small-en-v1.5 on CPU, batch size 32,
and **remote Qdrant**. A missing remote URL fails before model loading. `--qdrant-url` overrides
`QDRANT_URL`; keys are read only from the environment. Embedded Qdrant at `.data/qdrant` remains an
explicit development/test option: `--storage embedded`. It is not the production server's indexing engine.

All CLI global options precede the command:

```sh
uv run rag --data-dir .data-experiment ingest --directory ./documents --collection papers
```

**Each ingestion replaces the collection's entire active snapshot.** Supply the complete desired
source set on every run. Removing a document from that set removes it from the new active generation.
Reingesting unchanged inputs is a no-op; changed snapshots reuse cached embeddings for unchanged
text. An empty JSONL requires `--allow-empty` to clear a collection. Empty or unreadable local
documents fail the run and leave the previous generation active.

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
their computation counts separately. An unchanged semantic run can repeat segmentation using cached
sentence contexts without new inference. Changing chunking settings produces a new index generation.

For the already downloaded model, preview can run entirely offline:

```sh
uv run rag preview-chunks --file tests/fixtures/chunking/topics.txt \
  --offline --model-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
```

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
uv run rag ingest --documents .data/datasets/bioasq/EXPORT_ID/documents.jsonl --collection bioasq
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
  storage/                       # SQLite and Qdrant adapters
  cli.py                         # dependency composition and commands
tests/                           # offline correctness and local Qdrant integration tests
```

Ingestion stages raw and normalized documents plus chunks in `.data/generations/ID/`. Chunk offsets
refer to the normalized text, not raw PDF bytes. PDF pages are separate documents with explicit
page numbers. Markdown headings are retained as metadata and stay attached to their section body
where the budget allows. Oversized code blocks and tables may be split. Titles/headings are not prepended to
embedding input in this version, so the token budget applies directly to chunk text.

Each generation gets its own Qdrant collection. After vector-count verification, Qdrant atomically
switches the active alias. Collection metadata contains the model revision and corpus identity, so
query workers need neither the local SQLite manifest nor the ingestion machine's directory. Readers
resolve the alias once per request and pin the physical generation throughout. SQLite remains a local
audit and embedding cache. Remote publication timeouts are retained as `publication_pending` for
reconciliation. A local file lock protects one work directory; deployment must allow only one writer
per corpus. Old and failed generations are retained for inspection; garbage
collection and secure physical deletion are not implemented yet. Logical deletion from a new
snapshot does not erase retained historical data.

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
