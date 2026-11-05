# Semantic chunking and recursive fallback

The default strategy is `semantic`. A token ceiling limits model input; it does not determine where
normal chunks end. The original token-window implementation remains selectable as `token` for
controlled evaluation.

## Pipeline

1. Separate Markdown sections; ignore heading-like lines inside fenced code blocks.
2. Segment each body into sentence spans. Keep common biomedical abbreviations and decimal numbers
   together. The deterministic sentence heuristic supports English and Chinese punctuation but is
   not a complete linguistic parser.
3. At each candidate boundary, embed up to two complete sentences on the left and two on the right.
   These contexts never overlap across the candidate boundary. Drop distant sentences when necessary
   to fit the embedding model. An individually oversized sentence is recursively split first.
4. Calculate cosine distance between those contexts. Break where distance is at least the section's
   85th percentile and greater than 0.1. The floor prevents forced percentile splits in homogeneous text.
5. For each semantic group, use LangChain's `RecursiveCharacterTextSplitter` only if the group exceeds
   the configured ceiling. The separator order prefers paragraphs, lines, sentence punctuation,
   whitespace, punctuation, and finally individual characters. The length function is the embedding
   tokenizer, rather than character count. Retokenize and verify each result.
6. Preserve exact normalized-source offsets, section headings, a shared semantic group ID for size-split
   fragments, and a split reason. Overlap is off by default; optional overlap uses whole sentences
   within one semantic group and is omitted if it cannot fit.

The semantic algorithm is an explicit, separately testable module. The recursive splitting engine is
the maintained [LangChain text-splitter library](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).
No LLM call or external embedding API is used.

## Real-model preview

Source: `tests/fixtures/chunking/topics.txt`, a short synthetic document with six sentences about
two unrelated topics. Model: BGE-small-en-v1.5 at commit
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. Maximum chunk budget: 300 tokens.

| Chunk | Tokens | Content | Reason |
|---|---:|---|---|
| 1 | 33 | BRCA1, DNA repair, inherited cancer risk, genetic testing | Semantic boundary |
| 2 | 37 | Database backups, retention, restoration testing | Semantic boundary |

All three sentences in each topic stayed together. Both chunks are far below the 300-token ceiling;
the split was driven by semantics. Seven distinct context texts were embedded, with three duplicate
context requests reused. No vector index or corpus generation was created. The detailed output is
in [chunking-preview.json](chunking-preview.json).

## Cost and limitations

Semantic chunking requires embeddings before final chunk embeddings are generated. The shared cache
deduplicates and batches those inputs, records their model revision, and can reuse them on subsequent
runs. CLI results distinguish semantic embedding work from final chunk indexing work.

Semantic similarity is a heuristic, not proof that a passage is self-contained. Pronouns, citations,
tables, and long arguments can span boundaries. Oversized semantic groups still require splitting;
the model's context limit cannot be removed by a chunker. Semantic group IDs support later neighbor
or parent-context expansion, but that retrieval feature is not implemented yet. PDF pages are still
separate documents, so cross-page context is another later retrieval concern.

These thresholds have not been tuned for BioASQ. The synthetic preview demonstrates behavior, not
retrieval-quality improvement. Before full-corpus indexing, compare a representative development
sample using the token baseline and semantic strategy. Review sentence fragmentation, evidence
coverage, fragment counts, chunk lengths, and semantic embedding cost. Once retrieval exists,
compare evidence Recall@k and assembled-context coverage at the same context budget. Keep held-out
questions out of threshold tuning.
