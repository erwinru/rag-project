"""Text embedding via a local Hugging Face sentence-transformers model."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from rag.config import config

_model = SentenceTransformer(config.embedding.huggingface.model_id)


def embed_text(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()
