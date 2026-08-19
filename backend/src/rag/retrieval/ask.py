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
from rag.retrieval.search import search


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
