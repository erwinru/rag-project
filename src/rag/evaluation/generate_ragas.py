"""Generate synthetic QA pairs per article via RAGAS's TestsetGenerator.

Run once per `data/articles/*.json` file (not once for the whole corpus),
so every generated question/answer stays scoped to a single source article
-- see docs/Evaluation.md for why. This is the RAGAS half of the eval
framework; a second, custom-prompted generator covers the other half.

Usage:
    uv run rag-eval-ragas
"""

from __future__ import annotations

import asyncio
import json
import sys
import typing as t

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

from rag.config import config
from rag.embedding.chunking import chunk_text

if t.TYPE_CHECKING:
    from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatResult


class AsyncCompatibleChatHuggingFace:
    """Mixin fixing a real gap in `langchain_huggingface`: `ChatHuggingFace`
    unconditionally raises `NotImplementedError` on `_agenerate` when
    wrapping a local `HuggingFacePipeline` (its async path is only wired up
    for remote endpoints/text-gen-inference servers). RAGAS's
    `TestsetGenerator` runs its entire pipeline through async calls, so
    that's a hard incompatibility, not a config problem -- there's no
    actual reason a local pipeline call can't be awaited by running the
    existing sync `_generate` in a thread, so that's what this does.
    """

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: t.Any,
    ) -> ChatResult:
        return await asyncio.to_thread(
            self._generate, messages, stop=stop, run_manager=None, **kwargs
        )


def load_article_as_document(path) -> Document:
    doc = json.loads(path.read_text(encoding="utf-8"))
    metadata = doc["metadata"]
    return Document(
        page_content=chunk_text(doc["content"]),
        metadata={"title": metadata["title"], "source_article": metadata["slug"]},
    )


def build_generator_llm() -> LangchainLLMWrapper:
    provider = config.evaluation.generator_llm_provider
    print(f"[evaluation] generator LLM provider: {provider}", file=sys.stderr)

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return LangchainLLMWrapper(
            ChatBedrockConverse(
                model_id=config.evaluation.bedrock.model_id,
                region_name=config.bedrock.region,
            )
        )
    elif provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        class _AsyncChatHuggingFace(AsyncCompatibleChatHuggingFace, ChatHuggingFace):
            pass

        pipeline = HuggingFacePipeline.from_model_id(
            model_id=config.evaluation.huggingface.model_id,
            task="text-generation",
            device_map=config.evaluation.huggingface.device_map,
            pipeline_kwargs={"max_new_tokens": config.evaluation.huggingface.max_new_tokens},
        )
        return LangchainLLMWrapper(_AsyncChatHuggingFace(llm=pipeline))
    else:
        raise ValueError(f"Unknown generator_llm_provider: {provider!r}")


def _write_output(rows: list[dict], output_path) -> None:
    # Written atomically (temp file + rename) so a crash mid-write can never
    # corrupt the previous, still-valid file -- called after every article,
    # not just once at the end, so a crash partway through the batch loses
    # only the in-flight article, not everything generated so far.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(rows, indent=2, default=str))
    tmp_path.replace(output_path)


def build_generator() -> TestsetGenerator:
    # Generator embeddings always reuse the local MiniLM model regardless of
    # config.embedding.provider -- RAGAS only needs embeddings for its own
    # knowledge-graph construction, unrelated to which embedding backend the
    # RAG pipeline itself is configured to search with.
    generator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.embedding.huggingface.model_id)
    )
    return TestsetGenerator(llm=build_generator_llm(), embedding_model=generator_embeddings)


def main() -> int:
    paths = sorted(config.data.articles_dir.glob("*.json"))
    print(f"Found {len(paths)} articles in {config.data.articles_dir}", file=sys.stderr)

    generator = build_generator()
    output_path = config.evaluation.ragas_output_path

    rows: list[dict] = []
    for i, path in enumerate(paths, start=1):
        document = load_article_as_document(path)
        testset = generator.generate_with_langchain_docs(
            [document], testset_size=config.evaluation.ragas_questions_per_article
        )
        records = testset.to_pandas().to_dict(orient="records")
        for record in records:
            record["source_article"] = document.metadata["source_article"]
        rows.extend(records)
        _write_output(rows, output_path)
        print(f"[{i}/{len(paths)}] {path.name}: {len(records)} questions", file=sys.stderr)

    print(
        f"\nGenerated {len(rows)} QA pairs from {len(paths)} articles into "
        f"{output_path}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
