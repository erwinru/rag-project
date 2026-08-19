"""Embed a question and find its most similar chunks in Chroma."""

from __future__ import annotations

from rag.config import config
from rag.embedding.embedding import embed_text
from rag.embedding.store import get_vector_db_collection


def search(question: str, top_k: int = config.retrieval.top_k) -> list[dict]:
    """Return the `top_k` chunks most similar to `question`.

    Embeds `question` with the same Titan Text Embeddings V2 model (and
    dimensions) the index was built with -- a query vector is only
    comparable to vectors from the same model, so this can't be swapped for
    a different embedding model without re-embedding the whole index.

    Each result is `{"id", "text", "metadata", "distance"}`, `text` being
    the chunk's stored document (title + heading/paragraph blocks, see
    rag.embedding.chunking) -- the actual context handed to the generation
    model, not just a snippet.
    """
    collection = get_vector_db_collection()
    query_embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[query_embedding],  # type: ignore[arg-type]  # list[float] satisfies Sequence[float]; mypy can't see it through chromadb's invariant List stub
        n_results=top_k,
    )

    # query() always populates these for a plain query (only an explicit
    # `include=` override could drop them) -- the `or []` just satisfies
    # their Optional type, not a real fallback path.
    ids = results["ids"] or [[]]
    documents = results["documents"] or [[]]
    metadatas = results["metadatas"] or [[]]
    distances = results["distances"] or [[]]

    return [
        {
            "id": ids[0][i],
            "text": documents[0][i],
            "metadata": metadatas[0][i],
            "distance": distances[0][i],
        }
        for i in range(len(ids[0]))
    ]
