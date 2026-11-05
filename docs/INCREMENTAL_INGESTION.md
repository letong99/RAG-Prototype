# Incremental ingestion and generation lifecycle

Status: implemented for a single-host coordinator. Incremental patches are the default; explicit
full rebuilds rotate active/fallback collections. The distributed deployment described below remains
future work. See the [README](../README.md) for commands.

Implementation boundaries: retrieval requests are serialized by a process-safe file lock and must
share the coordinator directory. Preparation runs before maintenance; remote upload, validation, and
publication all run inside the window, including full rebuild uploads. Each mutation validates the
complete stored payload/ID set. Remote bootstrap/full builds require operator-supplied measured free
space, checked against a conservative estimate. Incremental runs can supply the same allowance;
consecutive vector upserts are batched into groups of at most 256 without crossing deletion or
metadata-update boundaries. Staging and upload progress are logged. Full-source deletion synchronization is not exposed;
use explicit deletion IDs or a full rebuild. The lifecycle schema is additive beside the legacy audit
DB; legacy remote generations cannot be silently adopted. Local artifact retention is not automated.

## Decisions

- Routine additions and updates modify only the active Qdrant collection.
- Keep one inactive fallback collection from the previous successful full rebuild. It is immutable
  during incremental ingestion and can lack subsequent corrections, additions, and deletions.
- Full rebuilds require an explicit operation. Embedding or chunking configuration changes must
  reject incremental ingestion and explain that a full rebuild is required.
- Pause retrieval during incremental mutation, drain existing requests, and serialize writers.
- Keep document versions on every chunk and verify them against a committed document manifest.
- Use existing database, indexing, and locking primitives; implement only application orchestration.

The first ingestion bootstraps the active collection; no previous fallback exists yet. Creating a
duplicate initial fallback is not implicit. A fallback is also not an independent disaster-recovery
backup: cluster loss requires separate snapshots or reproducible source artifacts.

## Identity and version model

| Field | Meaning |
|---|---|
| `corpus_id` | Stable logical knowledge-base identity |
| `generation_number` | Monotonically allocated number for each full rebuild; never reused |
| `ingestion_revision` | Monotonically allocated corpus-wide publication revision for updates, rebuilds, and rollback |
| `document_id` | Stable source identity, independent of its text |
| `content_hash` | Hash of normalized text |
| `metadata_hash` | Hash of citation and other relevant source metadata |
| `document_version` | Deterministic identity derived from content and relevant metadata |
| `processing_fingerprint` | Embedding model/revision, normalization and chunking algorithms/settings |
| `written_revision` | Revision that last wrote a chunk |

Store generation, document identity/version, processing fingerprint, and written revision in chunk
payloads alongside existing citations and offsets. Unchanged chunks retain their written revision.
Different documents need not have equal versions or written revisions. IDs may have gaps after
failed attempts; retry the same journaled operation rather than allocating a fresh identity.

The manifest records each document's committed version, hashes, expected chunk IDs/count, source
artifact reference, and processing fingerprint. Retain a matching manifest snapshot for the fallback.
An embedding-cache key includes exact embedding input and model specification; sharing an embedding
must not discard separate source identities or citations.

## Early change detection

Read and normalize each supplied source, then compare its identity and hashes before semantic
chunking. Trust source version tokens only where the source guarantees their meaning.

| Change | Work |
|---|---|
| No relevant change | Skip chunking, embedding, and Qdrant writes |
| Citation-only metadata change | Update payload/version and manifest without embedding |
| Text or chunk-affecting metadata change | Rechunk that document; reuse cached embeddings; replace its chunk set |
| New document | Chunk, embed/cache, and insert |
| Explicit deletion | Remove that document's chunks and commit a manifest tombstone |
| Processing fingerprint change | Refuse incremental mode; require explicit full rebuild |

Classify metadata by whether it participates in splitting or embedding before enabling metadata-only
updates. Use deterministic point IDs derived from document/version, processing fingerprint, and chunk
position/identity. Source metadata-only updates can retain existing point IDs; persist the exact IDs
in the manifest rather than assuming they can always be reconstructed from the current version.

Default incremental batches are patches: absent documents are not deletions. A separately explicit
full-source synchronization operation may calculate deletions from missing IDs, after validating that
the supplied source listing is complete. Reject duplicate conflicting document IDs in a batch.

## Controlled ingestion window

1. Acquire exclusive corpus writer ownership. Read the committed revision and prepare a durable
   operation journal, intended document changes, and replayable artifacts. Expensive chunking and
   embedding may happen here while retrieval still serves the unchanged active collection.
2. Persist maintenance state and stop admitting new retrieval requests across all service instances.
   Drain requests already admitted, including their reranking and context assembly. Abort before
   mutation if draining cannot be confirmed; new requests receive a retryable maintenance response.
3. Upsert prepared new chunks, apply payload changes, and delete obsolete exact point IDs through
   Qdrant's SDK. Wait for applied writes. Never use a broad deletion filter that can delete newly
   inserted chunks of the same document.
4. Verify affected documents against expected IDs, versions, counts, and processing fingerprints;
   verify deletions and corpus totals. Counts alone cannot establish completeness.
5. Commit manifest changes and publication revision, reconcile Qdrant collection metadata, and mark
   the journal committed. Resume retrieval only when both stores agree and verification succeeds.

Every retrieval entry point must participate in admission control, including future MCP tools.
Direct Qdrant clients bypass this guarantee. Maintenance state survives process restart and must
not automatically expire into a readable state after a crash.

For the first implementation, enforce one service deployment with a single writer and a persistent
SQLite journal/manifest, using standard SQLite transactions and existing `filelock`. A shared local
lock alone does not coordinate remote query workers. Before deploying multiple service instances,
move coordination and the authoritative manifest to PostgreSQL with SQLAlchemy/Alembic and database
locking, or route all admission through one coordinator. Reader admission and writer exclusion must
be synchronized, not implemented as an unsynchronized check of a boolean flag.

Loss of a distributed lock does not fence an old writer from Qdrant. Do not launch a replacement
writer until the previous worker and its outstanding writes are stopped or reconciled. This is a
deployment constraint, not a guarantee supplied by advisory locks.

## Recovery and fallback

Use journal states such as `prepared`, `maintenance`, `applying`, `verifying`, `committing`,
`committed`, and `recovery_required`. Journal before writes; retain intended point IDs and replayable
payload/vector artifacts. Retry the same deterministic operations after checking remote state.

Qdrant and the manifest database do not share a transaction. A timeout can mean a write succeeded
without an acknowledgement. Recovery must inspect actual state and complete the operation, rather
than assume failure or reopen retrieval. Failures after mutation keep maintenance enabled.

Normal recovery completes the interrupted revision. Explicit fallback activation is an alternative:
switch to the inactive collection together with its manifest and matching embedding configuration,
validate, then reopen retrieval. Record the rollback as a new publication revision while retaining
the fallback's original generation identity. Do not silently discard newer knowledge; report the
restored snapshot revision. Quarantine a partially modified former active collection rather than
label it a healthy fallback. Cleanup requires confirmed ownership and absence of readers/writers.

## Explicit full rebuild

Serialize against incremental writers and freeze the input corpus revision. Existing retrieval may
continue against the active collection while a new physical collection is built. Reuse existing
chunking, embedding-cache, and official Qdrant ingestion adapters.

Validate the entire candidate, persist a publication intent, then briefly gate/drain retrieval for
publication. Use Qdrant's atomic alias operations to make the candidate active and the previous
active collection inactive. Commit the corresponding manifest and revision; reconcile ambiguous
responses before reopening retrieval. Alias atomicity does not include the external manifest.

After successful publication and reader draining, retire the older fallback. Steady state is two
collections; rebuilding temporarily requires three, plus construction overhead. Check actual free
space before starting on the 4 GB tier. If insufficient, stop with a capacity explanation; do not
silently remove the fallback. Current size estimates are provisional until a semantic pilot is measured.

## Retrieval integrity

Pin the committed generation/revision and matching model specification at request admission.
Compare every returned chunk's document version against that generation's manifest and check
processing fingerprints. Chunks from the same document must agree on document version; document
versions are not compared across different documents. Returned chunks alone cannot establish that
an entire document was indexed completely, so ingestion performs completeness checks.

On integrity mismatch, fail the request with a retryable integrity/recovery error and record the
affected IDs; do not assemble potentially contradictory context or silently filter away corruption.
Key retrieval caches by published revision so incremental commits invalidate stale contexts.

## Library reuse and boundaries

| Responsibility | Existing tool / boundary |
|---|---|
| Upserts, payload updates, exact-ID deletions, counts, scroll, aliases | Official `qdrant-client`; server owns indexing/search |
| Retry-safe point writes | Qdrant upserts with application-supplied deterministic IDs |
| Recursive splitting | Existing LangChain `RecursiveCharacterTextSplitter` integration |
| Embedding inference | Existing Sentence Transformers adapter and embedding cache |
| Durable single-deployment manifest/journal | Python SQLite transactions and existing file lock |
| Future shared manifest and schema migrations | PostgreSQL, SQLAlchemy, Alembic |
| State validation and provenance | Existing Pydantic schemas |

Qdrant supplies point operations and atomic alias changes, not an end-to-end atomic document
replacement spanning our manifest. A generic indexing helper is reusable only if its deletion,
versioning, and recovery semantics match this design; avoid a second competing record manager.

## Implementation sequence and acceptance checks

1. Add versioned manifest schemas, deterministic identities, migration support, and change planning.
   Test normalized-equivalent sources, metadata classification, duplicate IDs, and input reordering.
2. Add incremental Qdrant operations and early skipping. With spy embedders/chunkers, assert unchanged
   documents invoke neither; metadata-only changes perform no embeddings. Verify changed documents
   leave no stale chunks and patch batches do not delete absent documents.
3. Add durable admission control and journals. Test reader draining, concurrent writer rejection,
   restart during each mutation/commit boundary, lost acknowledgements, and idempotent replay.
4. Add explicit full rebuild, active/fallback publication, fallback manifest retention, rollback,
   capacity preflight, and bounded cleanup. Test no fallback on bootstrap, fingerprint rejection,
   rollback model compatibility, three-collection peak, and ambiguous alias publication.
5. Integrate retrieval integrity and revision-aware caches. Test mixed versions, uniformly stale
   results, missing chunks at ingestion validation, and stale cached contexts.
6. Run disposable real-server integration tests for transport and recovery behavior, then a
   representative BioASQ pilot. Measure skip rate, embedding calls, update duration, downtime, disk
   growth, and retrieval recall/citation correctness against a clean full rebuild of the same corpus.

Local Qdrant tests cannot establish distributed coordination or server disk usage. No full BioASQ ingestion or hosted mutation was performed during this implementation. Runtime code
and offline fixture tests implement the lifecycle; real-server mutation tests remain opt-in.

## References

- [Qdrant point operations and idempotence](https://qdrant.tech/documentation/manage-data/points/)
- [Qdrant upsert API and wait semantics](https://api.qdrant.tech/api-reference/points/upsert-points)
- [Qdrant collections and atomic aliases](https://qdrant.tech/documentation/manage-data/collections/)
- [PostgreSQL locking primitives](https://www.postgresql.org/docs/current/explicit-locking.html)
- [Qdrant capacity planning](https://qdrant.tech/documentation/capacity-planning/)
