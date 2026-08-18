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

`rag-eval-ragas` runs `TestsetGenerator` **once per article**, not once
for the whole corpus -- see Decisions for why. Output is written to
`config.evaluation.ragas_output_path` (`data/eval/ragas_qa.json`), each row
tagged with `source_article` (the article's `slug`) so generated questions
can be traced back to which document they should be answerable from --
needed to score retrieval (did the right chunk come back?), not just
generation quality.

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
- **Generator LLM is Claude Haiku via `ChatBedrockConverse`, reusing
  `config.generation.model_id`/`config.bedrock.region`** -- the same model
  already wired up (in principle -- see Testing below) for the RAG
  pipeline's own answer generation, rather than introducing a third model
  just for eval data generation.
- **Generator embeddings are always the local Hugging Face model
  (`config.embedding.huggingface.model_id`), regardless of
  `config.embedding.provider`.** RAGAS only needs embeddings internally,
  for its own knowledge-graph node clustering/relationship-building -- this
  has nothing to do with which embedding model the RAG pipeline itself is
  configured to search with. Hardcoding it to the always-available local
  model means test-set generation never depends on Bedrock/Titan access at
  all, only the generator *LLM* does.
- **Article text reuses `rag.embedding.chunking.chunk_text`** (blank line
  around headings, single newline elsewhere) to build the LangChain
  `Document` handed to RAGAS, rather than a second ad hoc text-joining
  function -- one canonical "block list -> string" formatter for the whole
  project.
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

**Not yet verified end to end.** Config loading, all imports (`ragas`,
`langchain_aws`, `langchain_huggingface`), and the local embeddings half all
work. The generator LLM call itself is currently blocked: Bedrock Converse
against `eu.anthropic.claude-haiku-4-5-20251001-v1:0` fails with
`AccessDeniedException` (an explicit IAM deny, unrelated to and distinct
from the Titan quota issue) -- see
[`Troubleshooting.md`](Troubleshooting.md#bedrock-claude-haiku-converse----accessdeniedexception-unresolved-different-bug).
Decided to wait for that to be fixed on the AWS side rather than swap in a
different generator LLM provider, so a real single-article
(`testset_size=2`) smoke test is still pending -- do that before ever
running the full 142-article batch, since each article run is several real
LLM calls (persona/summary/query synthesis steps), not one.

## Checklist / open edge cases

- [ ] Full IAM-unblocked smoke test on 1-2 articles, to confirm the actual
      generated question/answer shape before running all 142 articles
- [ ] Option 2 (custom-prompted generator, the other 5 questions/article)
      not started
- [ ] No dedup or quality filtering on RAGAS's output yet -- rows are
      written as-is from `testset.to_pandas()`
- [ ] `ragas_questions_per_article = 5` is the requested starting number,
      not validated against actual output quality/diversity per article
- [ ] No cost/time estimate yet for the full 142-article batch (each
      article is several LLM calls, not one) -- worth sizing before running
      it unattended
