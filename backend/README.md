# Backend

Python side of the project: scrape the ML6 blog, chunk and embed it into
Chroma, and serve retrieval over HTTP. Design notes live in
[`../docs/`](../docs/).

## Run it

Every command resolves `config.toml` and its `data/` paths relative to the
working directory, so run them from this directory.

```bash
uv sync
```

The retrieval API ([`../docs/API.md`](../docs/API.md)):

```bash
uv run rag-api
```

## Pipeline

Only needed to build the index from scratch -- `data/index/` is already
populated in a working checkout.

```bash
uv run rag-scrape && uv run rag-process && uv run rag-embed
```

| Command | What |
| --- | --- |
| `rag-scrape` | Sitemap -> `data/raw_html/` + `data/pages.csv` |
| `rag-process` | Raw HTML -> `data/articles/*.json` + `data/articles.csv` |
| `rag-embed` | Chunk and embed into `data/index/` (Chroma) |
| `rag-ask` | CLI equivalent of the API's `/search` |
| `rag-api` | FastAPI retrieval service |
| `rag-viewer-data` | Build `chunk_viewer/data.json` for the chunk viewer |
| `rag-eval-ragas` | Synthetic QA pairs ([`../docs/Evaluation.md`](../docs/Evaluation.md)) |

## Chunk viewer

A static page for inspecting how articles were chunked. Pick a port that
isn't 8000 -- that's the API's.

```bash
uv run rag-viewer-data && python3 -m http.server 8001 --directory chunk_viewer
```

## Checks

```bash
uv run pytest && uv run ruff check && uv run mypy src
```
