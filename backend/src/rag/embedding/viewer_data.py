"""Generate the JSON data file for the local chunk_viewer webapp.

Reads every processed article (data/articles/*.json) plus its computed
chunks, and writes one combined JSON file that chunk_viewer/index.html loads
client-side over a plain fetch -- no backend server needed, just a static
file server.

Usage:
    uv run rag-viewer-data
    cd chunk_viewer && python3 -m http.server 8000
    # then open http://localhost:8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.config import config
from rag.embedding.chunking import chunk_article_file

VIEWER_DATA_PATH = Path("chunk_viewer/data.json")


def build_article_entry(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    metadata = doc["metadata"]
    return {
        "slug": metadata["slug"],
        "title": metadata["title"],
        "author": metadata["author"],
        "article_number": metadata["article_number"],
        "word_count": metadata["word_count"],
        "content": doc["content"],
        "chunks": chunk_article_file(path),
    }


def main() -> int:
    paths = sorted(config.data.articles_dir.glob("*.json"))
    print(f"Found {len(paths)} articles in {config.data.articles_dir}", file=sys.stderr)

    articles = [build_article_entry(path) for path in paths]

    VIEWER_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    VIEWER_DATA_PATH.write_text(
        json.dumps({"articles": articles}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(articles)} articles to {VIEWER_DATA_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
