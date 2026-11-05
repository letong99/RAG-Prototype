# Local MCP server for Codex and Claude Code

The server exposes one read-only `retrieve_context` tool over stdio. The agent launches
and manages the process; there is no HTTP listener. It uses the existing published index,
local coordinator directory, and Qdrant configuration. It never starts ingestion. The calling
agent owns final answer generation.

## Install and connect

From the repository run `.tools/bin/uv sync` to install the MCP Python SDK (v1 API,
`mcp>=1.26,<2`) and the `rag-mcp` entry point. Keep `.env` configured with `QDRANT_URL`
and `QDRANT_API_KEY`. No secrets need to go in client configuration.

Determine the published collection with `rag --data-dir .data status --collection COLLECTION`.
Replace `COLLECTION` below with that collection name. Paths here match this checkout; change
all paths if the repository moves. `uv run --env-file` loads credentials without shell evaluation.

Codex registration:

```sh
codex mcp add local-rag -- /Users/letong/Desktop/code/RAG/.tools/bin/uv run --project /Users/letong/Desktop/code/RAG --env-file /Users/letong/Desktop/code/RAG/.env rag-mcp --data-dir /Users/letong/Desktop/code/RAG/.data --collection COLLECTION
```

Claude Code registration (local scope):

```sh
claude mcp add --transport stdio local-rag -- /Users/letong/Desktop/code/RAG/.tools/bin/uv run --project /Users/letong/Desktop/code/RAG --env-file /Users/letong/Desktop/code/RAG/.env rag-mcp --data-dir /Users/letong/Desktop/code/RAG/.data --collection COLLECTION
```

Use `codex mcp list` or `claude mcp list` to inspect registration, then start a new agent session
and ask it to call `local-rag` with a question. These commands are documented here; this change
does not modify your personal client configuration. See the official
[Codex MCP setup](https://learn.chatgpt.com/docs/extend/mcp) and
[Claude Code MCP setup](https://code.claude.com/docs/en/mcp).

For embedded development storage, omit `--env-file` and pass `--storage embedded`. Do not open
the same embedded Qdrant directory from multiple processes. The default uses your Qdrant server;
local stdio describes the MCP connection, independently of where Qdrant runs.

The server loads the published embedding model/revision on first query and reuses it. First-call
latency includes loading (and downloads unless `--offline` is set). For offline use, cache the
published model first and pass `--offline`. Restart after publishing a different embedding model.
Library/model stdout is redirected to stderr during inference to preserve MCP framing.

## Tool contract

```json
{
  "query": "How long are backups retained?",
  "top_k": 6,
  "max_context_tokens": 2500,
  "document_ids": null
}
```

`query` must contain text and fit the embedding model; `top_k` is 1–40, context budget is
1–16000. The dense candidate pool is 40; context assembly removes duplicate/overlapping evidence
and enforces a 98000-byte context envelope. `document_ids` only narrows the configured collection,
is limited to 100 IDs, and `[]` returns no passages. Transport registration is local only and
provides no remote authentication layer. Current transport uses dense ranking; the Python service
still accepts an optional reranker. Semantic corpus-wide summaries remain a separate future feature.

The structured response contains `context` (passages, citations, version, source locations,
truncation information) and `metrics`:

| Metric | Meaning |
|---|---|
| `request_id` | Correlates the response with the local metrics log |
| `response_time_ms` | Server wall time from handler entry through assembly and token counting, including model loading and waiting for its worker lock; excludes metrics-file write and MCP/client transport |
| `query_tokens` | Query text counted by the published embedding tokenizer |
| `context_tokens` | Serialized evidence array, including citation/provenance metadata |
| `result_tokens` | Full serialized `context` object, excluding the metrics wrapper and MCP framing |
| `total_tokens` | `query_tokens + result_tokens`; context tokens are already included in result tokens |
| `generation_tokens` | Zero: this tool retrieves evidence and performs no answer generation |
| `tokenizer` | Embedding model name and revision |
| `token_counts_estimated` | True for token usage in the calling agent, whose tokenizer and tool framing are outside this service |

These counts are not provider billing or aggregate model-compute tokens; they exclude reranker
pairs, system prompts, client reasoning, tool definitions, and final generated answers. The client
may render both MCP structured and text representations, so caller context usage can differ.

Each successful call appends matching metrics plus timestamp, collection, generation and ingestion
revision to `DATA_DIR/query_metrics.jsonl`. Errors in the handler append latency and error category
and return a sanitized MCP error with a request ID. SDK schema rejection happens before the handler
and does not create a query record. The log excludes query text and source passages; file locking
coordinates multiple server processes. Failure to persist metrics fails the call. Rotate/archive
this append-only file operationally as needed.

## Validation

`pytest -q tests/test_mcp.py` covers usage calculations, persisted metrics, model reuse, empty
results, sanitized failures, SDK initialize/list/call/schema validation, and a real stdio subprocess
roundtrip. The subprocess uses deterministic test-only embeddings and embedded Qdrant; no model
or credentials are required. SDK protocol tests skip when the dependency is unavailable.
