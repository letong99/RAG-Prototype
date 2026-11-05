# Hosted BioASQ semantic ingestion smoke test

## Outcome

Passed ingestion, incremental insertion, unchanged-document skipping, stored-payload integrity,
and guarded dense retrieval against the user's existing Qdrant 1.20.0 server.

- Logical collection: `bioasq-semantic-smoke`.
- Physical collection: `rag_3f4ea467d85448f8b8722b3a24c7b9e4`.
- Final state: **21 valid documents, 60 chunks, generation 1, ingestion revision 4**.
- Maintenance is off. No inactive fallback exists because no explicit full rebuild was requested.
- The small collection is retained for inspection. The full corpus has not been embedded or uploaded.
- Model: `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`,
  384 dimensions, cached weights, CPU inference, offline model loading.
- Semantic chunking with recursive character fallback; maximum 300 tokens, zero overlap.

## Data-quality issue found and corrected

The first 20-passage sample produced 40 chunks in 12.777 seconds. Its unchanged rerun skipped all
20 documents in 0.217 seconds. Adding one passage took 1.406 seconds and retained generation 1.
However, seven sample passages contained literal `nan`; one test question's only evidence was one
of those placeholders. The run passed operational checks but exposed an input-validation defect.

An audit of the complete local export found:

| Source/evaluation item | Count |
|---|---:|
| Original exported rows | 40,221 |
| Placeholder-only passage rows | 12,220 |
| Usable text passages after validation | 28,001 |
| Evaluation questions | 4,719 |
| Questions missing at least one labelled passage after validation | 3,547 |
| Questions with all labelled passages missing | 332 |
| Questions with all labelled passages available | 1,172 |

Dataset revision: `224a87f64a5c3a720b5bc627cf760f543b2f1e79`.
The earlier claim that all exported rows contained usable evidence was incorrect: the old check
rejected null/empty values but accepted string placeholders. Source adapters now reject entire
placeholder strings (`nan`, `none`, `null`, ignoring case/whitespace), and direct normalization
also rejects them for BioASQ-tagged sources. Embedded occurrences in ordinary prose remain valid.
QA labels are retained with missing-evidence IDs rather than silently dropping difficult questions.

A corrected full **local export only** was written to
`.data/datasets/bioasq/validated-224a87f/`. Original exports remain for audit but should not be used
for ingestion. Old exports now fail validation on placeholder rows.

## Repaired sample and incremental check

Selected three questions with complete, valid labelled evidence, included all ten distinct evidence
documents, and filled the sample to 20 with valid passages. This is an evidence-selected functional
sample, not a random or representative evaluation corpus. Replaced the earlier sample via an
incremental patch (19 updated/inserted, 20 explicitly deleted, one unchanged), then repeated ingestion
and inserted one additional valid passage. All operations used the same physical collection.

| Operation | Documents after operation | Chunks | Elapsed | Revision |
|---|---:|---:|---:|---:|
| repair_sample | 20 | 58 | 17.597 s | 3 |
| unchanged_repeat | 20 | 58 | 0.207 s | 3 |
| incremental_insertion | 21 | 60 | 1.336 s | 4 |

The unchanged rerun performed **zero semantic or final-chunk embedding calls** and skipped all
20 documents. The final incremental insertion reused two cached chunk embeddings and computed four
sentence-context embeddings. Timings exclude model initialization and include server calls; cache
reuse means these are not cold-start throughput measurements.

## Chunk and storage observations

- Token counts: minimum 10, median 69.0, maximum 297.
- Split reasons: 52 semantic and 8 size-limited chunks.
- All chunks respected the 300-token ceiling and passed stored payload/identity checks.
- Raw float32 vectors: 92,160 bytes; serialized chunk payloads: 74,925 bytes.
  These are calculated content sizes, **not measured Qdrant disk usage**; indexes, WAL, and other
  overhead are additional. Do not extrapolate a final 4 GB capacity guarantee from this tiny sample.
- The server was empty at start and `/metrics` reported no disk-quota breach, but did not expose
  free bytes. A conservative 256 MB smoke-test allowance was supplied within the user-stated 4 GB
  tier. It was explicitly an allowance, not a measured free-space value.

## Retrieval sanity checks

Retrieval returned five chunks per question, potentially containing multiple chunks from the same
document. Counts below are unique labelled documents found, not number of relevant chunks.

| Query ID | Question | Labelled documents found / available | Retrieval time |
|---|---|---:|---:|
| 40 | Treatment of which disease was investigated in the MR CLEAN study? | 2/4 | 0.333 s |
| 41 | Which factors activate zygotic gene expression during the maternal-to-zygotic transition in zebrafish? | 1/1 | 0.355 s |
| 46 | Are there web based self management strategies for chronic pain ? | 3/5 | 0.336 s |

All three queries found labelled evidence in the top five chunks. Mean document recall over these
three selected questions was 0.70. This is a basic functionality check, not a benchmark or evidence
that semantic chunking improves quality. No reranker, hybrid retrieval, or answer-generation model
was used. Guarded retrieval checked generation, document version, processing fingerprint, and
payload identity against the committed manifest before returning context.

## Reproducibility and validation

Ignored local artifacts in `.data/smoke-remote/` include:

- `result.json`: original smoke result, including the failed evidence query.
- `source-audit.json`: corrected full-export counts.
- `clean-documents.jsonl`, `clean-questions.jsonl`, `clean-incremental-document.jsonl`: corrected inputs.
- `clean-result.json`, `clean-chunks.jsonl`: corrected run metrics and actual chunks.
- `coordinator/`: persistent lifecycle ledger, embedding cache, and replay artifacts.

The hosted runner loaded credentials from ignored `.env` without printing them; no key or endpoint
is included in this report. Bootstrap, incremental writes/deletes, no-op ingestion, metadata
publication, and guarded queries were exercised on the actual server. Full-rebuild/fallback and
failure-recovery behavior remain covered by local tests; they were not exercised on the hosted
server during this smoke test.

Regression validation after the placeholder fix: **81 passed, 4 skipped**; Ruff lint/format,
strict mypy (31 source files), and diff whitespace checks pass. The skipped tests are opt-in model
and disposable-server tests; the real cached-model and hosted runs described here were separate.
