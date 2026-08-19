"""FastAPI service exposing retrieval: a question in, its top-k chunks out.

Retrieval only -- no generation step. `rag.retrieval.search` does the actual
work (embed the question with whichever backend `config.embedding.provider`
selects, query Chroma); this module is a thin HTTP shape around it, so the
API can never drift from what `rag-ask` does.

The unit stored in Chroma is a *chunk*, not a whole article, so `top_k`
counts chunks: two results can come from the same article, each carrying
the article it belongs to (`title`, `author`, `slug`). See docs/API.md.

Usage:
    uv run rag-api
    curl "http://127.0.0.1:8000/search?question=What+is+MLOps?&top_k=5"
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rag.config import config
from rag.embedding.store import get_vector_db_collection
from rag.retrieval.search import search


class SearchResult(BaseModel):
    """One retrieved chunk, plus the article it came from."""

    id: str
    # Derived from `id` (`{slug}::chunk-NN`), not stored as chunk metadata --
    # see rag.embedding.chunking.chunk_blocks.
    slug: str
    title: str
    author: str
    # Chroma's distance: lower is more similar. Not a similarity score, and
    # not normalised to 0-1 -- don't present it as a percentage.
    distance: float
    text: str


class SearchResponse(BaseModel):
    question: str
    top_k: int
    results: list[SearchResult]


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    # Defaults to config.retrieval.top_k, capped at config.api.max_top_k so a
    # caller can't ask Chroma for the entire collection.
    top_k: int = Field(default=config.retrieval.top_k, ge=1, le=config.api.max_top_k)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Touch the Chroma collection at startup so a missing/empty index shows
    up in the server log immediately, not on the first user request."""
    collection = get_vector_db_collection()
    print(
        f"[api] collection '{config.embedding.resolved_collection_name}' "
        f"({config.data.index_dir}): {collection.count()} chunks",
        file=sys.stderr,
    )
    yield


app = FastAPI(
    title="RAG retrieval API",
    description="Retrieve the most relevant ML6 blog chunks for a question.",
    version="0.1.0",
    lifespan=lifespan,
)


def run_search(question: str, top_k: int) -> SearchResponse:
    try:
        chunks = search(question, top_k=top_k)
    except Exception as exc:
        # Embedding is the only outbound dependency here (Bedrock, when
        # config.embedding.provider is "bedrock") -- throttling or missing
        # model access is an upstream failure, not a bad request.
        raise HTTPException(
            status_code=502, detail=f"retrieval failed: {exc}"
        ) from exc

    return SearchResponse(
        question=question,
        top_k=top_k,
        results=[
            SearchResult(
                id=chunk["id"],
                slug=chunk["id"].split("::")[0],
                title=str(chunk["metadata"].get("title", "")),
                author=str(chunk["metadata"].get("author", "")),
                distance=chunk["distance"],
                text=chunk["text"],
            )
            for chunk in chunks
        ],
    )


@app.get("/health")
def health() -> dict[str, object]:
    collection = get_vector_db_collection()
    return {
        "status": "ok",
        "collection": config.embedding.resolved_collection_name,
        "chunks": collection.count(),
        "embedding_provider": config.embedding.provider,
    }


@app.get("/search", response_model=SearchResponse)
def search_get(
    question: str = Query(min_length=1),
    top_k: int = Query(default=config.retrieval.top_k, ge=1, le=config.api.max_top_k),
) -> SearchResponse:
    """Query-string form -- curl/browser friendly."""
    return run_search(question, top_k)


@app.post("/search", response_model=SearchResponse)
def search_post(request: SearchRequest) -> SearchResponse:
    """JSON-body form -- same handler, for clients that prefer POST."""
    return run_search(request.question, request.top_k)


def main() -> int:
    import uvicorn

    uvicorn.run(app, host=config.api.host, port=config.api.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
