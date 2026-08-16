"""Ask a question over the embedded corpus: embed -> retrieve -> generate.

Three steps, two different Bedrock models: `rag.retrieval.search` embeds the
question with Titan Text Embeddings V2 (same model the index was built
with) and finds the top-k most similar chunks in Chroma; `rag.retrieval.
generation` hands those chunks to Claude Haiku as context and asks it to
answer. Chroma itself is a local lookup, not a Bedrock call.

Usage:
    uv run rag-ask "How does 3D computer vision work?"
"""

from __future__ import annotations

import sys

from rag.config import config
from rag.embedding.store import get_vector_db_collection
from rag.embedding.embedding import embed_text


def search(question: str, top_k: int = 3) -> list[dict]:
    collection = get_vector_db_collection()
    query_embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

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


def main() -> int:
    question = sys.argv[1]

    chunks = search(question, top_k=config.retrieval.top_k)

    print("Sources:", )
    for chunk in chunks:
        print(f"  - {chunk['metadata'].get('title')} ({chunk['id']})", )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
