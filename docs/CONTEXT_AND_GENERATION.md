# Context assembly, generation, and dataset questions

`RetrievalService.retrieve_context(request, embedder, limits=..., tokenizer=..., reranker=...)`
uses the complete ranked candidate pool before selecting evidence. `ContextLimits` bounds
passage count, serialized evidence tokens (including provenance), and UTF-8 response bytes.
The default tokenizer is the embedding tokenizer and counts are explicitly estimates for
an unknown caller model. Pass the target model's tokenizer and label via `ContextLimits`
when available. This budget covers evidence, not the entire generation prompt; the provider
must reserve room for instructions, question, and output in its own model window.

Assembly preserves ranking, suppresses identical text and same-version/page passages with
at least 80% overlap relative to the shorter span, and shortens oversized text only at tokenizer
offsets. It preserves chunk identity, original end offset, version, source URI, title, heading,
page, and metadata. A shortened chunk's text, end offset, and token count describe the returned
excerpt. Citation IDs are assigned to selected passages. Omission counts refer to the ranked
candidate pool, not all documents in the collection. Duplicate counts are separate. Large
metadata can cause omission; an envelope exceeding the byte limit raises an error.

`RetrievalService.answer_question(request, embedder, generator, ...)` is optional. Supply an
`AnswerGenerator` implementation whose `generate(system=..., question=..., evidence=...)`
returns `GeneratedAnswer`. No provider, credentials, model download, or external request is
implicitly enabled. The adapter must enforce its model window, output limit, request timeout,
and provider configuration. `grounded-answer-v1` requires evidence-only answers, citations,
insufficiency acknowledgement, and treating sources as untrusted data. Source content is JSON
escaped. The service rejects unknown citations and answered output without citations; this is
structural validation, not verification that a citation supports a claim. Prompt instructions
are not a security boundary. Empty evidence abstains without calling the generator; nonempty
but insufficient evidence is a distinct generator decision, never a similarity-score threshold.

The evidence snapshot is immutable and revision-labelled. Retrieval verifies it under the
existing ingestion admission gate, then assembly and provider inference use the copied snapshot
without blocking ingestion. These Python entry points inherit the existing single-host ledger
requirement and corpus boundary. Authentication/MCP transport remains a separate milestone;
these methods do not add per-user authorization.

## Dataset-wide questions

Use explicit operations rather than routing every question through nearest-neighbor search:

- `describe_dataset(collection)` returns exact published document/chunk counts, recorded
  languages, and index identity. It reads the entire committed manifest under admission and
  needs no embedding model. Missing language metadata is `unknown`. It does not infer topics.
- Factual questions use `retrieve_context` or optional `answer_question`.
- Semantic requests such as “summarize this dataset” should use a dedicated overview index.
  Build document summaries, aggregate topic/group summaries, then a corpus summary during
  ingestion. Keep supporting document IDs, versions, citations, processing/prompt/model versions,
  and coverage counts at each level. Publish summaries with the corresponding corpus revision;
  recompute changed documents and affected ancestors after updates/deletes. Fail closed or label
  unavailable/stale summaries instead of mixing revisions. Apply the same access scope before
  aggregation: a whole-corpus summary must never serve a restricted document subset.

Hierarchical semantic summaries are the recommended next phase, not implemented by the inventory
method. They require choosing a generation provider and evaluating summary coverage. A bounded,
diverse sample can support an explicitly labelled preview, but must not be presented as a full
corpus summary. Use a caller-selected overview/factual intent initially; an eventual classifier
can suggest a route, with an explicit override for ambiguous questions. The calling agent can
already answer inventory questions from `describe_dataset` and should explain that a content
summary is unavailable until revision-matched summaries exist.

Validation uses deterministic generators and embedded Qdrant. Live provider quality, prompt
injection resistance, groundedness, and citation completeness still require labelled evaluation.
