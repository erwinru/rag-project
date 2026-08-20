# Retrieval

First basic version. Implementation: [`backend/src/rag/retrieval/`](../backend/src/rag/retrieval/).

## Overview

Answering a question is three steps, two different Bedrock models, one
local lookup:

1. **Embed** ([`search.py`](../backend/src/rag/retrieval/search.py)) -- the question
   is embedded with whichever backend `config.embedding.provider` selects
   (Titan Text Embeddings V2 or a local Hugging Face model -- see
   [`Embedding.md`](Embedding.md)), the *same* model used to build the
   index in `rag.embedding`. This isn't optional: a query vector is only
   comparable to vectors from the same model, so the embedding model here
   can't be swapped independently of the one that built the index.
2. **Retrieve** (`search.py`, same function) -- that vector is queried
   against the Chroma collection (`config.retrieval.top_k` results, 3 by
   default). This is a local lookup, not a Bedrock call.
3. **Generate** ([`generation.py`](../backend/src/rag/retrieval/generation.py)) --
   the retrieved chunks' text, labeled by source article title, are handed
   to Claude Haiku as context in a prompt, along with the original question.
   Claude never sees the vectors or does any searching itself; it just reads
   whatever text `search()` handed it.

[`ask.py`](../backend/src/rag/retrieval/ask.py) wires these together and is the
`rag-ask` CLI entry point.
Steps 1-2 are also exposed over HTTP -- see [`API.md`](API.md).

## Decisions

- **Two separate models/services, used for two unrelated jobs.** Embedding
  (`rag.embedding.embedding`, reused as-is here) is provider-configurable
  (Titan Text Embeddings V2 via Bedrock, or a local Hugging Face model --
  see [`Embedding.md`](Embedding.md)) and can't generate text at all.
  Claude Haiku (`rag.retrieval.generation`) always runs on Bedrock,
  independently of the embedding provider, called through a completely
  different API (Converse, not `invoke_model`).
- **Claude uses the Converse API (`client.converse`), not `invoke_model`.**
  `invoke_model` (what `rag.embedding.embedding` uses for Titan) is the raw
  per-model request format; Converse is Bedrock's unified interface for
  chat/generation models across providers, and is what Claude's Bedrock
  integration is built around.
- **The model id is a region gotcha, not the plain model name.**
  `eu-central-1` (this project's configured Bedrock region) has no
  in-region/direct endpoint for Claude Haiku 4.5 -- only the **EU geo
  cross-region inference profile**, whose id is prefixed `eu.`:
  `eu.anthropic.claude-haiku-4-5-20251001-v1:0`. Using the bare model id
  (`anthropic.claude-haiku-4-5-20251001-v1:0`) from this region would fail;
  confirmed against AWS's own Bedrock model-card documentation, not guessed.
  Model access for this profile still needs to be granted separately from
  Titan's, the same way Titan's own access needed granting earlier.
- **`generation.py` builds its own `boto3` client** (same adaptive-retry
  config as `embedding.py`'s Titan client, reusing `config.bedrock.
  max_attempts`/`retry_mode`) rather than sharing the Titan client, since
  the two talk to different model families and there's no real coupling
  between them beyond "same AWS account/region."
- **Context is chunks' stored `text`, labeled by source article title**,
  joined with a `---` separator -- not just raw concatenation, so Claude
  (and a human reading the prompt) can tell which article each piece of
  context came from.
- **`top_k` defaults to 3.** Enough context for most questions without
  overloading the prompt or diluting relevance; both this and
  `generation.max_tokens` (1024) are arbitrary starting points, not tuned
  against real question/answer quality yet.

## Testing

Not yet covered by automated tests -- `search()` and `generate_answer()`
both require live, billed Bedrock calls (embedding + generation), and
`search()` additionally requires a populated Chroma index. Verified
structurally instead: `build_context`/`PROMPT_TEMPLATE` (pure functions)
directly, and the Chroma query-result parsing logic against a local dummy
vector (no Bedrock call, since a fixed-value embedding is enough to
exercise Chroma's actual response shape).

## Checklist / open edge cases

- [ ] No automated tests -- `build_context`'s formatting is straightforward
      enough to unit test without any live calls; `search()`/`generate_answer()`
      would need mocking Bedrock or a recorded fixture to test without cost
- [ ] No handling for an empty Chroma collection (0 results) -- `search()`
      just returns `[]`, and `generate_answer()` would build an empty
      context block and ask Claude to answer from nothing
- [ ] No multi-turn/conversation history -- every call is a single,
      independent question
- [ ] No answer citations back to source articles/chunk ids in the actual
      generated text (only printed separately to stderr as "Sources") --
      worth revisiting once there's a UI that could render real links
- [ ] `top_k` and `generation.max_tokens` are unvalidated guesses -- same
      "need real data to decide" situation as the chunking window/ceiling
