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
from typing import Literal

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


class BedrockEmbeddingConfig(BaseModel):
    model_id: str
    dimensions: int


class HuggingFaceEmbeddingConfig(BaseModel):
    model_id: str


class EmbeddingConfig(BaseModel):
    # Which embedding backend rag.embedding.embedding.embed_text() uses.
    # See docs/Embedding.md.
    provider: Literal["bedrock", "huggingface"]
    collection_name: str
    bedrock: BedrockEmbeddingConfig
    huggingface: HuggingFaceEmbeddingConfig

    @property
    def resolved_collection_name(self) -> str:
        """Chroma collection actually used -- suffixed by provider, since
        each provider's model produces a different vector dimension (Titan
        V2: 1024, MiniLM-L6-v2: 384) and a single Chroma collection locks
        to whichever dimension it saw first. See docs/Embedding.md.
        """
        return f"{self.collection_name}_{self.provider}"


class RetrievalConfig(BaseModel):
    top_k: int


class ApiConfig(BaseModel):
    # Bind address for the FastAPI retrieval service (rag.api.app).
    host: str
    port: int
    # Ceiling on a request's `top_k` -- retrieval.top_k is the default, this
    # is how far a caller is allowed to raise it. See docs/API.md.
    max_top_k: int


class GenerationConfig(BaseModel):
    # eu-central-1 has no in-region Claude Haiku 4.5 -- only the EU
    # cross-region inference profile ("eu." prefix). See docs/Retrieval.md.
    model_id: str
    max_tokens: int


class EvaluationBedrockGeneratorConfig(BaseModel):
    # Deliberately separate from config.generation.model_id -- that's
    # Claude Haiku, for the RAG pipeline's own answer generation; this is
    # whichever Bedrock model RAGAS's generator uses, independently. See
    # docs/Evaluation.md.
    model_id: str


class EvaluationHuggingFaceGeneratorConfig(BaseModel):
    model_id: str
    max_new_tokens: int
    # "auto" lets accelerate pick the best available device (MPS on Apple
    # Silicon, CUDA if present, else CPU) -- see docs/Evaluation.md.
    device_map: str


class EvaluationConfig(BaseModel):
    # Synthetic QA generation for retrieval/generation eval. See
    # docs/Evaluation.md.
    ragas_questions_per_article: int
    ragas_output_path: Path
    # Which LLM RAGAS's TestsetGenerator uses as its generator model --
    # "bedrock" or "huggingface" (a local model). Independent of
    # config.embedding.provider.
    generator_llm_provider: Literal["bedrock", "huggingface"]
    bedrock: EvaluationBedrockGeneratorConfig
    huggingface: EvaluationHuggingFaceGeneratorConfig


class Config(BaseModel):
    data: DataConfig
    scrape: ScrapeConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    api: ApiConfig
    bedrock: BedrockConfig
    embedding: EmbeddingConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return Config.model_validate(raw)


CONFIG_PATH = Path(os.environ.get("RAG_CONFIG_PATH", "config.toml"))
config = load_config(CONFIG_PATH)
