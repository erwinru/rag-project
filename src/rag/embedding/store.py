"""Persistent Chroma collection for embedded article chunks."""

from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection

from rag.config import config


def get_collection() -> Collection:
    client = chromadb.PersistentClient(path=str(config.data.index_dir))
    return client.get_or_create_collection(config.embedding.collection_name)


def upsert_chunks(
    collection: Collection, chunks: list[dict], embeddings: list[list[float]]
) -> None:
    """Upsert by chunk id, so re-running the pipeline updates rather than duplicates."""
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings,  # type: ignore[arg-type]  # list[float] satisfies Sequence[float]; mypy can't see it through chromadb's invariant List stub
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
