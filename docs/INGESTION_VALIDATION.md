# Foundation and ingestion validation

## Latest hosted smoke test

See [the hosted smoke report](SMOKE_TEST_RESULTS.md): 21 valid passages / 60 chunks are now indexed
remotely. It uncovered 12,220 placeholder-only rows in the old 40,221-row export; the corrected local
export contains **28,001 valid passages**. There are 3,547 questions with at least one missing labelled
passage. The full corpus remains unembedded. Source validation is fixed; 81 default tests pass.

The sections below record earlier milestones. Their original "no missing evidence" finding was
superseded by the placeholder audit; source-ID presence alone did not mean usable passage text.

## Historical dataset and index state

- Dataset: `rag-datasets/rag-mini-bioasq`.
- Dataset revision: `224a87f64a5c3a720b5bc627cf760f543b2f1e79`.
- Full export: 40,221 passage documents and 4,719 evaluation questions.
- The original validator reported no rejected text or missing IDs; the later placeholder audit above corrects that finding.
- Only the original **100-passage smoke subset** was embedded and indexed with the token-window
  baseline: 118 chunks, 95 distinct new chunk embeddings, and 23 reused embeddings.
- Its collection is `bioasq-smoke`; its active generation is `2f1297c109004817802914f0b8f19df2`.
- Repeating that baseline ingestion returned `unchanged` with no new embeddings.
- **The full corpus has not been embedded or indexed.** No further dataset ingestion was run after
  the request to revise chunking.

## Semantic revision

Final verification: **44 tests passed**, including both token and semantic chunking against the
cached real embedding model. Ruff lint/format checks and strict mypy checks passed. The model tests
ran with `HF_HUB_OFFLINE=1`; no further BioASQ ingestion was performed.

The default is now semantic grouping with LangChain recursive character splitting. The old smoke
index remains a baseline; changing the default does not silently rewrite an existing index.

The new strategy is verified with deterministic topic-vector tests, source-offset and tokenizer-limit
checks, lifecycle/cache tests using temporary Qdrant storage, and a small real-model preview on a
synthetic file. The preview generated two coherent topic chunks without indexing them. See
[CHUNKING.md](CHUNKING.md).

Retrieval-quality evaluation, full-corpus indexing with the new strategy, hybrid search, reranking,
and MCP remain later work. These correctness tests do not establish improved retrieval quality.

## Remote-storage revision

Remote Qdrant is now the default CLI target, with an explicit embedded mode for development/tests.
The official SDK handles vector writes, payload indexing, filtered vector queries, and atomic alias
publication. Collection metadata provides model revision and corpus identity to independent readers.
SQLite is an ingestion audit/cache rather than a query-service dependency.

The original `bioasq-smoke` index predates the new alias metadata and was not migrated or uploaded.
Remote configuration, SDK delegation, alias switching, and lost-response recovery have been tested
using local fixtures and mocks. A subsequent read-only check confirmed the existing server and client are both 1.20.0, with no
collections or aliases. No hosted mutation/lifecycle test has been run. A separate opt-in server test is supplied. See [REMOTE_STORAGE.md](REMOTE_STORAGE.md).

Historical remote-storage verification: **50 passed, 3 skipped** (two opt-in real-model cases and the
unconfigured real-server test). Ruff lint/format and mypy passed. The real-model cases were already
verified during the semantic revision above; the hosted-server test remains outstanding.


## Incremental lifecycle implementation

Latest default verification: **73 passed, 4 skipped** (two opt-in real-model cases and two
unconfigured real-server tests). The two real-model cases also passed separately with
`RAG_TEST_MODEL=1 HF_HUB_OFFLINE=1` (2 passed, 75 deselected). Ruff lint/format and strict mypy checks pass.

Implemented and tested with the official embedded Qdrant SDK:

- Early normalized-document skipping with no chunker/embedder/vector-write calls.
- Metadata-only updates, explicit deletions, stale chunk removal, and patch semantics.
- Per-document versions, generation numbers, monotonic committed revisions, and processing checks.
- Active/inactive rotation, older-fallback cleanup, explicit rollback, and capacity preflight.
- Durable journals, checksummed replay artifacts, same-operation replay after lost acknowledgements,
  and recovery after closing and reopening storage.
- Reader draining before writes, maintenance blocking, and strict retrieval payload/version checks.

The coordinator is single-host and requires a shared data directory. Readers are serialized and
must use the retrieval context through context assembly. Full payload verification scans the
collection on every changed ingestion. These are deliberate prototype limits, not a distributed
availability or throughput claim. Schema version 1 initializes a new lifecycle database beside the
legacy audit/cache; it does not silently migrate a pre-existing remotely published index.

At this implementation milestone, no production knowledge-base ingestion or hosted write had been performed.
The subsequent limited hosted smoke run is documented above. The optional server lifecycle
test uses only `RAG_TEST_QDRANT_URL`, uniquely named fixture collections, and explicit fixture cleanup.
The full BioASQ corpus remains unembedded. Local audit/replay files are retained; remote cleanup does
not erase historical local source data.
