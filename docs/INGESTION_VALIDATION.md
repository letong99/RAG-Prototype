# Foundation and ingestion validation

## Dataset and index state

- Dataset: `rag-datasets/rag-mini-bioasq`.
- Dataset revision: `224a87f64a5c3a720b5bc627cf760f543b2f1e79`.
- Full export: 40,221 passage documents and 4,719 evaluation questions.
- This downloaded revision had no rejected passage text and no missing relevant passage IDs.
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
using local fixtures and mocks. No real cloud connection has been tested; that requires the cluster
endpoint and credentials. A separate opt-in server test is supplied. See [REMOTE_STORAGE.md](REMOTE_STORAGE.md).

Latest remote-storage verification: **50 passed, 3 skipped** (two opt-in real-model cases and the
unconfigured real-server test). Ruff lint/format and mypy passed. The real-model cases were already
verified during the semantic revision above; the hosted-server test remains outstanding.
