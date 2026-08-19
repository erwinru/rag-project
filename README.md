# rag-project

RAG over the ML6 blog: scrape, index, retrieve.

## Layout

| Path | What |
| --- | --- |
| [`backend/`](backend/README.md) | Python: ingestion, chunking, embedding, the retrieval API. Run commands from here. |
| [`frontend/`](frontend/README.md) | React chat UI for the retrieval API. |
| [`docs/`](docs/) | Design notes and decisions, one file per subsystem. |
| [`infrastructure/`](infrastructure/) | Terraform (S3 bucket for the raw scrape). |

## Quickstart

Backend, from `backend/` -- `config.toml` and every path in it resolve
relative to the working directory, so the directory matters:

```bash
cd backend && uv sync && uv run rag-api
```

Frontend, in a second shell:

```bash
cd frontend && npm install && npm run dev
```

http://localhost:5173. The API is retrieval only -- it returns matching
source chunks, not a generated answer.
