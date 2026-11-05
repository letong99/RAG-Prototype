# Full BioASQ incremental ingestion

Status: in progress. This is not a completion report.

- Source: `.data/datasets/bioasq/validated-224a87f/documents.jsonl` (28,001 valid passages).
- Excluded upstream placeholders: 12,220, retained in the corrected export's rejection audit.
- Target: existing `bioasq-semantic-smoke` collection, generation 1.
- Physical collection: `rag_3f4ea467d85448f8b8722b3a24c7b9e4`.
- Starting publication: revision 4, 21 documents, 60 chunks.
- Expected patch: 27,980 new documents; 21 unchanged documents skipped.
- Coordinator: `.data/smoke-remote/coordinator/` (the same ledger as the smoke run).
- Model: cached `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- Chunking: semantic percentile 85, distance floor 0.1, two-sentence windows, recursive character
  fallback, 300-token maximum, zero overlap. No embedding/chunking configuration changes.
- Additional-storage allowance: 2 GB within the user-stated 4 GB tier, not measured free space.
  The staged conservative estimate is checked before remote mutation.
- Remote writes batch up to 256 consecutive vector upserts while preserving mutation ordering.

Progress is recorded in `.data/full-ingestion/run.log`; machine-readable state and final results
are written atomically to `.data/full-ingestion/result.json`. Preparation caches embeddings locally
while the existing published index remains readable. Remote upload and verification run inside the
controlled retrieval window. A failure after mutation requires journal replay with `rag recover`.

After publication the runner verifies the exact remote point count, checks all stored chunk payloads
against the committed manifest, reruns the entire input to prove unchanged-document skipping, and
runs three guarded retrieval sanity checks. Successful completion is reported only after these
checks pass. These three questions do not constitute a full retrieval-quality evaluation.
