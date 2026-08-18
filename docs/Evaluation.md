# Evaluation

First basic version -- **RAGAS half only.** Implementation:
[`src/rag/evaluation/generate_ragas.py`](../src/rag/evaluation/generate_ragas.py).

## Overview

Goal: a synthetic QA dataset to evaluate retrieval + generation quality
against, since neither has been tested against real question/answer pairs
yet (see the open checklist items in [`Retrieval.md`](Retrieval.md)). Two
generation approaches were planned, combined per article (10 questions per
article -- 5 + 5):

1. **RAGAS's `TestsetGenerator`** -- this doc. An established library for
   synthetic test-set generation, varying question difficulty/type
   (single-hop, multi-hop, abstract vs. specific) automatically.
2. **A custom-prompted generator** -- not built yet. Would prompt the
   project's own generation model directly with a difficulty-tiered prompt
   template, more control but hand-rolled, consistent with how the rest of
   this project (chunking, retrieval) avoids frameworks.

`rag-eval-ragas` runs the generator **once per article**, not once for the
whole corpus, and within each article it generates from **this project's own
chunks** rather than letting RAGAS re-split the article text -- see Decisions
for both. Output is written to `config.evaluation.ragas_output_path`
(`data/eval/ragas_qa.json`); on top of RAGAS's own columns each row carries:

- `source_article` -- the article's `slug`, so a question can be traced to
  the document it should be answerable from (article-level retrieval
  scoring: did *a* chunk of the right article come back?).
- `reference_chunk_ids` -- the `{slug}::chunk-NN` ids of the exact chunks the
  question and reference answer were written from (chunk-level scoring: did
  *the* supporting chunk come back, and at what rank?). Multi-hop rows carry
  more than one.

## Decisions

- **Per-article generation loop, not one corpus-wide `TestsetGenerator`
  call.** RAGAS's typical usage pattern generates a testset across an
  entire document set at once, letting its knowledge-graph step relate
  chunks across different documents (useful for multi-hop questions that
  span multiple sources). That's the wrong shape here: every generated
  question needs to trace back to *one* source article for retrieval
  scoring, and "10 questions per article" was the explicit requirement --
  not a fixed total across the whole corpus. Running the generator once per
  `data/articles/*.json` file, `testset_size=config.evaluation.
  ragas_questions_per_article` each time, keeps every question's source
  unambiguous at the cost of losing genuine cross-article multi-hop
  questions (a document has enough internal chunks/sections for
  within-article multi-hop, just not across articles).

- **Questions are generated from this project's chunks, not RAGAS's own
  splits.** `TestsetGenerator.generate_with_langchain_docs()` hardcodes two
  things: it wraps each document as a `NodeType.DOCUMENT` node, and it
  builds `default_transforms()`, which includes RAGAS's `HeadlinesExtractor`
  + `HeadlineSplitter`. So passing a whole article always got it re-split by
  RAGAS, and the resulting `reference_contexts` corresponded to nothing in
  the Chroma index -- scoring "did retrieval return the supporting chunk?"
  would have needed fuzzy text-overlap matching against a tuned similarity
  threshold, with all the arbitrariness that implies.

  Instead the knowledge graph is built by hand: one `NodeType.CHUNK` node
  per chunk from `rag.embedding.chunking.chunk_article_file` (the *same*
  function `rag-embed` indexes with), and `default_transforms_for_prechunked`
  -- a first-class RAGAS entry point whose whole purpose is to skip the
  splitting step -- applied over it, then `generator.generate()` directly.
  Both synthesizers copy a node's `page_content` verbatim into
  `reference_contexts` (multi-hop prefixes each with `<N-hop>`), so mapping
  a generated row back to chunk ids is an exact dict lookup, not a
  similarity match. Any context that fails to resolve is counted and
  reported at the end of the run rather than silently dropped -- 0 is the
  expected number, and anything else means a RAGAS version changed that
  assumption.

- **Personas are regenerated per article.** `generate()` only infers
  personas when `persona_list is None`, and caches them on the generator
  afterwards -- so reusing one `TestsetGenerator` across a per-article loop
  silently applies the *first* article's personas to the entire corpus. The
  previous dataset shows the damage: 808 rows, exactly one persona
  ("Computer Vision Engineer", inferred from the first article, alphabetically
  a 3D computer vision post) writing questions about every other topic in the
  corpus. `generate_for_article` resets `persona_list` to `None` per article;
  `config.evaluation.ragas_personas_per_article` controls how many.

- **Embedding calls are serialized behind a lock.** Concurrent calls into the
  local sentence-transformers model segfault the interpreter (exit 139, no
  traceback) -- see
  [`Troubleshooting.md`](Troubleshooting.md#local-embedding-model----sigsegv-under-concurrent-ragas-extraction-resolved).
  A crash like this is invisible to any `try`/`except`, which is worth
  remembering when reading the skip-on-failure guard below: it protects
  against Python exceptions only.

- **RAGAS's `CustomNodeFilter` is dropped from the transform list.** For a
  `CHUNK` node it scores the chunk against its *parent* document's summary,
  which assumes the chunk was produced by RAGAS's own splitter and still has
  a `child` relationship up to a `DOCUMENT` node. Chunks built directly from
  `chunk_article_file` have no parent, so the filter reads an empty summary,
  logs `does not have a summary. Skipping filtering.` and keeps the node --
  once per chunk, roughly 680 warnings per run, while filtering nothing. It
  costs nothing (it returns before making its LLM call), but an inert
  LLM-backed quality gate is worse than a visibly absent one, so
  `build_transforms` filters it out.

  **Consequence: nothing currently drops low-signal chunks from the eval
  set.** Restoring the filter would mean giving each article's graph a
  summarized `DOCUMENT` parent node related to its chunks, at one extra LLM
  call per article -- worth doing if generated question quality turns out to
  be the bottleneck.

- **One un-generatable article skips instead of aborting the batch.**
  `default_query_distribution` raises outright when no synthesizer finds a
  usable node cluster, which a very short or single-chunk article can
  trigger. A full-corpus run is far too expensive (several LLM calls per
  chunk, ~680 chunks) to lose at article 100 over one such article, so
  `main()` catches per-article failures, logs the article name, and
  continues -- reporting the skip list at the end.

- **Generator embeddings are always the local Hugging Face model
  (`config.embedding.huggingface.model_id`), regardless of
  `config.embedding.provider`.** RAGAS only needs embeddings internally,
  for its own knowledge-graph node clustering/relationship-building -- this
  has nothing to do with which embedding model the RAG pipeline itself is
  configured to search with. Hardcoding it to the always-available local
  model means test-set generation never depends on Bedrock/Titan access at
  all, only the generator *LLM* does.

- **The generator LLM is configurable and deliberately not the pipeline's own
  generation model.** `config.evaluation.generator_llm_provider` picks
  `bedrock` (`config.evaluation.bedrock.model_id`, currently
  `qwen.qwen3-coder-30b-a3b-v1:0`) or `huggingface` (a small local model, no
  AWS dependency, but noticeably weaker at RAGAS's structured JSON prompts).
  Claude Haiku -- `config.generation.model_id`, what the RAG pipeline itself
  would answer with -- is deliberately *not* used here: it is still blocked
  on the Anthropic use-case form and an IAM deny (see
  [`Troubleshooting.md`](Troubleshooting.md#bedrock-claude-haiku-converse----accessdeniedexception-unresolved-different-bug)),
  and generating eval data with the same model under evaluation would be
  circular anyway.

- **New dependencies, and one had to be pinned:** `ragas`, `langchain-aws`
  (the `ChatBedrockConverse` wrapper), `langchain-huggingface` (the
  embeddings wrapper), and `rapidfuzz` (an undeclared runtime import inside
  `ragas.testset.transforms.relationship_builders.traditional` -- not
  listed in `ragas`'s own dependency metadata, only discovered by running
  it). `langchain-community` had to be pinned to `<0.4`: `ragas==0.4.3`
  unconditionally imports `langchain_community.chat_models.vertexai` at
  import time, which in `langchain-community==0.4.2` was split out to
  require the separate (unrelated, GCP-specific) `langchain-google-vertexai`
  package -- an upstream version-compatibility bug between the two
  libraries' latest releases, not something wrong in this project's code.

## Testing

**Verified, with the generator LLM stubbed out (no Bedrock calls):** config
loading and imports; `build_chunk_nodes` producing `NodeType.CHUNK` nodes
from `chunk_article_file` output; `resolve_reference_chunk_ids` round-tripping
both a verbatim single-hop context and a `<N-hop>`-prefixed multi-hop pair
back to the right chunk ids (and returning `None` for an unknown context);
and the **full transform chain over three real articles** -- every node
summarized, embedded (with real embeddings, which is where the segfault was),
themed, NER-extracted, and related, with 2-8 chunks per article. The RAGAS
API details this rewrite depends on were read out of the installed
`ragas==0.4.3` rather than assumed.

**Not verified:** `generate()` itself -- persona inference, scenario
selection and query/answer writing -- which needs live generator-LLM calls,
and the `reference_contexts` round trip on real generated output (as opposed
to synthetic contexts fed through `resolve_reference_chunk_ids` directly).
Do a single-article smoke test before any full batch: each article is now
several LLM calls *per chunk* (summary, themes, NER), not per article.

Because `_write_output` rewrites the file after every article, the cheapest
smoke test needs no code change at all -- start the run and interrupt it once
the first `[1/142]` line appears, then inspect `reference_chunk_ids` and the
unresolved-context count in the output.

A previous, pre-rewrite full batch did complete: 808 rows over 135 of 142
articles, preserved at `data/eval/ragas_qa_old.json`. It is still usable for
article-level scoring, but not chunk-level (its contexts are RAGAS's own
splits), and its questions all carry the single leaked persona described
above.

## Checklist / open edge cases

- [ ] Single-article smoke test of `generate()` before the full 142-article
      batch (the transform chain is already verified with a stubbed LLM)
- [ ] No cost/time estimate for the full batch under the new node count
      (~680 chunk nodes vs. 142 document nodes; RAGAS's own splitter had
      already been producing a comparable number of chunk nodes, so the real
      increase is `SummaryExtractor` now running per chunk rather than per
      document)
- [ ] Adjacent chunks share text: `chunk_blocks` prepends the title and
      carries an overlap paragraph from the previous chunk, so a question
      drawn from an overlap region is genuinely answerable from two chunks
      while `reference_chunk_ids` names only the one RAGAS sampled. Retrieval
      scoring should count an adjacent-chunk hit as its own category rather
      than a clean miss.
- [ ] No chunk quality filtering at all, now that `CustomNodeFilter` is out
      (see Decisions). Add a summarized `DOCUMENT` parent per article if
      generated-question quality needs it.
- [ ] Option 2 (custom-prompted generator, the other 5 questions/article)
      not started
- [ ] No dedup or quality filtering on RAGAS's output yet -- rows are
      written as-is from `testset.to_pandas()` (the old dataset had 2
      duplicate questions out of 808)
- [ ] `ragas_questions_per_article = 5` and `ragas_personas_per_article = 3`
      are starting numbers, not validated against actual output
      quality/diversity per article
- [ ] Nothing consumes `reference_chunk_ids` yet -- the eval/metrics script
      that scores retrieval against this dataset is the next piece
