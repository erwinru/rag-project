"""Where things live on disk.

Paths are relative to the working directory, not to the installed package, so
the same code works from a checkout and from an install. Point RAG_DATA_DIR
somewhere else to move the whole tree at once; individual paths are also
overridable per-command via CLI flags.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))

CACHE_DIR = DATA_DIR / "cache"
ARTICLES_DIR = DATA_DIR / "articles"
CSV_PATH = DATA_DIR / "articles.csv"
INDEX_DIR = DATA_DIR / "index"
