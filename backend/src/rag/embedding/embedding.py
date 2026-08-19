"""Text embedding dispatcher -- picks a provider (Bedrock or local Hugging
Face) based on config.embedding.provider. See docs/Embedding.md.
"""

from __future__ import annotations

import sys

from rag.config import config

print(f"[embedding] using provider: {config.embedding.provider}", file=sys.stderr)

if config.embedding.provider == "bedrock":
    from rag.embedding.providers.bedrock import embed_text
elif config.embedding.provider == "huggingface":
    from rag.embedding.providers.huggingface import embed_text
else:
    raise ValueError(f"Unknown embedding provider: {config.embedding.provider!r}")

__all__ = ["embed_text"]
