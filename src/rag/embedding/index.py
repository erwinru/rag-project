"""Chunk every processed article and embed it into the Chroma index.

Reads data/articles/*.json (written by rag-process), chunks each with
rag.embedding.chunking, embeds every chunk's text with Titan Text Embeddings
V2, and upserts into a persistent Chroma collection -- keyed by chunk id, so
re-running updates rather than duplicates.

Usage:
    uv run rag-embed
"""

from __future__ import annotations

import sys

from rag.config import config
from rag.embedding.chunking import chunk_article_file
from rag.embedding.embedding import embed_text
from rag.embedding.store import get_vector_db_collection, upsert_chunks


def main() -> int:
    INDEX_DIR = config.data.index_dir

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(config.data.articles_dir.glob("*.json"))
    print(f"Found {len(paths)} articles in {config.data.articles_dir}", file=sys.stderr)

    collection = get_vector_db_collection()
    total_chunks = 0

    for i, path in enumerate(paths, start=1):
        chunks = chunk_article_file(path)
        embeddings = [embed_text(chunk["text"]) for chunk in chunks]
        upsert_chunks(collection, chunks, embeddings)

        total_chunks += len(chunks)
        print(f"[{i}/{len(paths)}] {path.name}: {len(chunks)} chunks", file=sys.stderr)

    print(
        f"\nEmbedded {total_chunks} chunks from {len(paths)} articles "
        f"into '{config.embedding.resolved_collection_name}' ({INDEX_DIR}).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
