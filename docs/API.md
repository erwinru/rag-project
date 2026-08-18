# API

HTTP retrieval endpoint. Implementation: [`src/rag/api/`](../src/rag/api/).

## Overview

A FastAPI service that exposes step 1+2 of the pipeline
([`Retrieval.md`](Retrieval.md)) over HTTP: a question goes in, the most
similar chunks come back. There is deliberately **no generation step** --
this returns retrieved source material, not an answer.

Run it:

```bash
uv run rag-api
```

It binds to `config.api.host`/`config.api.port` (127.0.0.1:8000 by default)
and serves interactive docs at `/docs`.
To run it in a container instead, see [`Docker.md`](Docker.md).

### `GET /search` / `POST /search`

Two shapes of the same handler -- query string for curl/browser, JSON body
for clients that prefer POST.

```bash
curl "http://127.0.0.1:8000/search?question=What%20is%20MLOps%3F&top_k=5"
```

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is MLOps?", "top_k": 5}'
```

Response:

```json
{
  "question": "What is MLOps?",
  "top_k": 5,
  "results": [
    {
      "id": "the-mlops-maturity-model::chunk-03",
      "slug": "the-mlops-maturity-model",
      "title": "The MLOps maturity model",
      "author": "...",
      "distance": 0.41,
      "text": "..."
    }
  ]
}
```

`top_k` defaults to `config.retrieval.top_k` and is capped at
`config.api.max_top_k`; out-of-range or missing values are rejected by
FastAPI with a 422 before any embedding call happens.

### `GET /health`

Reports the resolved collection name, its chunk count and the active
embedding provider -- enough to tell "the service is up" apart from "the
service is up but pointed at an empty index."

## Decisions

- **`top_k` counts chunks, not articles.** The unit stored in Chroma is a
  chunk ([`Chunking.md`](Chunking.md)), so two results can come from the
  same article. Each result carries the article it belongs to (`slug`,
  `title`, `author`) rather than being silently deduplicated: collapsing to
  distinct articles would mean either returning fewer than `top_k` results
  or querying more than asked for and trimming, and which of those is right
  depends on how a caller uses this -- not decided yet.
- **The API is a shape, not a second implementation.** It calls
  `rag.retrieval.search.search()`, the same function `rag-ask` uses, so the
  HTTP path can't drift from the CLI path. Fixing that up meant two small
  changes on the way in: `search.py` was calling a `get_collection()` that
  doesn't exist in `rag.embedding.store` (the real name is
  `get_vector_db_collection()`), and `ask.py` carried a duplicate copy of
  `search()` that has now been deleted in favour of the import.
- **`slug` is derived from the chunk id, not read from metadata.** Chunk
  metadata is only `{title, author}` (see
  `rag.embedding.chunking.chunk_blocks`), but ids are `{slug}::chunk-NN`,
  so the slug is recoverable without re-indexing. The article **URL** is
  *not* -- it's written to `data/articles.csv` and the per-article JSON, but
  never into Chroma, so the API can't return a link yet.
- **`distance` is passed through raw.** Chroma's distance -- lower is more
  similar, not normalised to 0-1. It is not a similarity score and
  shouldn't be rendered as a percentage.
- **An embedding failure is a 502, not a 500 or a 400.** When
  `config.embedding.provider` is `bedrock`, the only outbound dependency in
  the request path is the Titan call; throttling or missing model access is
  an upstream failure, and 502 says so. Bad input never gets that far --
  FastAPI validates it first.
- **The index is touched at startup** (FastAPI lifespan), so a missing or
  empty Chroma collection shows up in the server log at boot rather than as
  a confusing empty result on someone's first request.

## Checklist / open edge cases

- [ ] No automated tests -- the handler is thin, but `run_search()`'s
      chunk-to-`SearchResult` mapping is worth a test with a stubbed
      `search()`, no embedding call needed
- [ ] No article URL in the response (not in Chroma metadata -- would need
      `chunk_blocks` to carry it and a re-index)
- [ ] No auth, no rate limiting, no CORS -- localhost-only by default
      (`config.api.host`); all three need deciding before this is exposed
      anywhere else
- [ ] Single worker, and `search()` is synchronous -- FastAPI runs it in a
      threadpool, but the local Hugging Face embedding model is loaded once
      per process, so concurrency is untested
- [ ] Empty collection returns `200` with `results: []` rather than
      signalling "nothing indexed" -- `/health` is the way to tell, for now
- [ ] No generation endpoint -- `generation.py` doesn't exist on this branch
      yet, so `/search` is retrieval only
