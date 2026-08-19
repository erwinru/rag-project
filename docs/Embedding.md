# Embedding

Implementation: [`backend/src/rag/embedding/embedding.py`](../backend/src/rag/embedding/embedding.py)
(dispatcher) and [`backend/src/rag/embedding/providers/`](../backend/src/rag/embedding/providers/)
(the two backends).

## Overview

`embed_text(text: str) -> list[float]` is the single interface every caller
uses -- `rag.embedding.index` (building the Chroma collection) and
`rag.retrieval.search` (embedding a query) both import it from
`rag.embedding.embedding`, unaware of which backend actually runs. Which
backend that is comes entirely from `config.embedding.provider`:

- `"bedrock"` -- Amazon Titan Text Embeddings V2, via `invoke_model`. Hosted,
  billed per call, and subject to AWS's own region/quota/model-access
  management -- see [`Troubleshooting.md`](Troubleshooting.md) for what went
  wrong here.
- `"huggingface"` -- a local `sentence-transformers` model
  (`sentence-transformers/all-MiniLM-L6-v2` by default), downloaded once and
  run on-machine. No network call per embedding, no rate limits, no billing.

Provider-specific settings live under `[embedding.bedrock]` /
`[embedding.huggingface]` in `config.toml`; `provider` and
`collection_name` are the only fields shared across both.

## Decisions

- **One dispatcher module, two provider modules, picked at import time --
  not a runtime branch inside `embed_text`.** `embedding.py` imports only
  the active provider's module based on `config.embedding.provider`, so
  switching to `"huggingface"` never imports `boto3`, and switching to
  `"bedrock"` never imports `sentence_transformers`/`torch`. Neither
  provider's heavy dependency is paid for unless it's actually selected.
- **Both providers expose the exact same signature** (`embed_text(text:
  str) -> list[float]`), so `index.py`/`search.py` need no provider-aware
  code at all -- config is the only thing that changes.
- **Embeddings from the two providers are not interchangeable.** Titan V2
  produces 1024-dim vectors (configurable, but 1024 here); MiniLM-L6-v2
  produces 384-dim vectors. Chroma infers dimensionality from whatever's
  upserted first, so switching `provider` on an existing collection means
  re-running `rag-embed` against a fresh `collection_name` (or wiping the
  index dir) -- a query vector from one model is never comparable to
  index vectors built with the other, same constraint noted in
  [`Retrieval.md`](Retrieval.md).
- **`sentence-transformers/all-MiniLM-L6-v2` was picked as the default
  Hugging Face model deliberately small** (~90MB, 384-dim) -- fast to
  download, fine on CPU, good enough to validate the whole pipeline works
  end to end without Bedrock. Not evaluated against retrieval quality yet;
  see checklist.
- **The Hugging Face model loads once at import time** (module-level
  `_model = SentenceTransformer(...)` in `providers/huggingface.py`), same
  pattern as the Bedrock client being built once at module load in
  `providers/bedrock.py` -- avoids re-loading model weights on every
  `embed_text` call.
- **No Hugging Face auth wired up.** `all-MiniLM-L6-v2` (and the other
  candidates considered -- `BAAI/bge-*`, `intfloat/e5-*`) are public
  repos; an `HF_TOKEN` would only be needed for a gated/private model or to
  raise the anonymous download rate limit, neither of which applies here.

## Testing

Not covered by automated tests -- both providers make a real model call
(a live, billed Bedrock request, or a local model needing its weights
downloaded first), same reasoning as `rag.retrieval.search`/`generation`
in [`Retrieval.md`](Retrieval.md).

## Checklist / open edge cases

- [ ] No retrieval-quality comparison yet between Titan V2 and
      MiniLM-L6-v2 -- picked the Hugging Face default for being small and
      dependency-light, not for benchmarked accuracy
- [ ] No guard against querying a Chroma collection with a different
      `provider`/dimension than it was built with -- currently just fails
      inside Chroma (dimension mismatch) rather than a clear error from
      this project's own code
- [ ] Bedrock provider still blocked on the region/access issue in
      `Troubleshooting.md` -- untested end to end since the config split
