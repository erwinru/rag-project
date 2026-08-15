"""Typed config object, loaded from config.toml.

Paths are relative to the working directory, not to the installed package, so
the same code works from a checkout and from an install. Set RAG_CONFIG_PATH
to point at a different toml file (e.g. per environment); individual paths
are also overridable per-command via CLI flags.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel


class DataConfig(BaseModel):
    raw_html_dir: Path
    # Manifest written by the scrape step, read by the processing step.
    pages_csv: Path
    articles_dir: Path
    csv_path: Path
    index_dir: Path


class ScrapeConfig(BaseModel):
    sitemap_url: str
    blog_prefix: str
    user_agent: str


class ChunkingConfig(BaseModel):
    window_tokens: int
    heading_search_allowance_tokens: int
    min_chunk_tokens: int


class BedrockConfig(BaseModel):
    # Matches the region the rest of this project's infrastructure runs in
    # (infrastructure/environments/dev/variables.tfvars) -- Bedrock is
    # regional, and Titan Text Embeddings V2 must be invoked in a region that
    # offers it.
    region: str
    max_attempts: int
    retry_mode: str


class EmbeddingConfig(BaseModel):
    model_id: str
    dimensions: int
    collection_name: str


class Config(BaseModel):
    data: DataConfig
    scrape: ScrapeConfig
    chunking: ChunkingConfig
    bedrock: BedrockConfig
    embedding: EmbeddingConfig


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return Config.model_validate(raw)


CONFIG_PATH = Path(os.environ.get("RAG_CONFIG_PATH", "config.toml"))
config = load_config(CONFIG_PATH)
