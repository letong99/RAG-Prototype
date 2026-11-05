# Retrieval, reranking, and human review

Implemented baseline: Qdrant dense candidate retrieval → integrity validation → optional
Sentence Transformers CrossEncoder reranking → top-N passages. Qdrant owns nearest-neighbor
search and indexing; Sentence Transformers owns model inference. No custom vector index.
The running ingestion configuration and collection schema are unchanged.

## Service API

```python
from rag_service.retrieval.models import SearchRequest
from rag_service.retrieval.service import RetrievalService

result = RetrievalService(coordinator_dir, store).search(
    SearchRequest(
        collection="bioasq-semantic-smoke", query="What causes ...?", candidate_limit=40, top_n=6
    ),
    embedder,
    reranker=reranker,  # omit for the dense baseline
)
```

Results include the request, committed generation/revision, embedding specification,
optional pinned reranker specification, candidate count, elapsed time, and ranked passages.
Each passage retains its dense rank/score, separate reranker score, complete text, source URI,
character offsets, title, document version, and chunk identity. Scores are not probabilities.
Candidate limits are 1–100; top-N cannot exceed the candidate limit. Document-ID filtering
happens in Qdrant before candidate selection. An empty document filter returns no results.
Exact duplicate chunk IDs are removed; distinct chunks from the same document are retained.
Tied reranker scores retain dense order.

The query embedder must exactly match the committed embedding specification. Queries exceeding
its token budget fail before vector search. CrossEncoder pairs exceeding the configured token
budget also fail, avoiding silent evidence truncation. Reranker failures and malformed scores
fail the request; there is no silent fallback to dense ranking.

The coordinator's admission lock covers retrieval, version/hash checks, and reranking.
Pending ingestion mutations block reads. All candidates are checked against the committed
manifest before ranking, not just the returned top-N. Unchanged documents may legitimately
have older `written_revision` values: their document version and payload must match the
active manifest. This is a **single-coordinator** design: workers must share the same local
coordinator directory. An endpoint URL alone is insufficient for safe remote MCP deployment;
a distributed admission/manifest implementation remains future work.

## Human evaluation script

**Wait until full ingestion finishes before running export against the remote database.**
These commands were documented, not executed against the hosted collection during development.
Use the activated project environment. Export expects QDRANT_URL and QDRANT_API_KEY in the
environment; it does not load `.env` automatically. Never include credentials in output files.

Dense baseline, using the existing coordinator and cached embedding model:

```bash
python scripts/review_retrieval.py export \
  --data-dir .data/smoke-remote/coordinator \
  --collection bioasq-semantic-smoke \
  --query 'What is the role of insulin in glucose metabolism?' \
  --candidates 40 --top-n 10 --offline \
  --output .data/retrieval-dense.json
```

For reranking, add `--rerank`. The initial online run resolves and records a model commit and
may download `cross-encoder/ms-marco-MiniLM-L6-v2`; omit `--offline` for that first run.
Subsequent offline runs require `--reranker-revision FULL_COMMIT_SHA` from the exported result.
The default is a general web-search model, not a validated biomedical model. Compare its
results with the dense baseline before adopting it. Reranking requires no document reembedding.

Grade a saved run entirely offline:

```bash
python scripts/review_retrieval.py grade \
  --run .data/retrieval-dense.json \
  --output .data/retrieval-judgments.jsonl
```

The script shows the query, passage, rank, and source, hiding numeric scores during grading.
Grades: 0 irrelevant, 1 related, 2 partially answers, 3 directly answers; `s` leaves a passage
unjudged. Optional notes are saved with a run fingerprint, chunk ID, document version, and rank.
Judgments append and flush after each answer. Repeated reviews append additional judgments;
consumers must choose a reviewer/adjudication policy rather than count duplicates as new evidence.
Exports refuse to overwrite existing files. Saved text is untrusted source content, not instructions.

## Evaluation and limitations

1. Freeze the published generation/revision and reranker commit; use identical queries and
   candidate limits for dense and reranked runs.
2. Review a representative set of BioASQ questions, including difficult biomedical terminology
   and no-answer cases. Use complete-evidence questions as the primary labelled benchmark.
   The validated dataset records missing passages; do not silently treat missing gold evidence
   as retrieval failure or inflate recall by ignoring it without reporting the affected count.
3. Compare candidate document Recall@40, final document Recall@N and MRR, and human graded
   nDCG@N. Deduplicate document IDs for document metrics. Grade pooled, shuffled candidates
   blind to system identity for comparative assessment; the interactive script is a basic
   inspection aid, not a blinded experiment or an automatic metric runner.
4. Measure latency and failure rates separately; reranking cannot recover evidence absent
   from the candidate pool. Report unjudged results explicitly.

This milestone returns ranked evidence, not an assembled LLM prompt. Context budgeting,
lexical/hybrid retrieval, automated benchmark reports, and MCP transport remain separate work.
No sparse vectors were added to the collection while ingestion was running.

Tests use embedded Qdrant in temporary directories and deterministic model doubles; no remote
retrieval or model downloads. Model inference quality and hosted end-to-end performance are
intentionally unverified until ingestion completes.

Library reference: [Sentence Transformers CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).
