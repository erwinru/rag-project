"""Generate synthetic QA pairs per article via RAGAS's TestsetGenerator.

Runs once per `data/articles/*.json` file (not once for the whole corpus),
so every generated question/answer stays scoped to a single source article
-- see docs/Evaluation.md for why. Within an article, questions are
generated from *this project's own chunks* (rag.embedding.chunking, the
same ones rag-embed indexes) rather than letting RAGAS re-split the article
with its own splitter, so each row records the exact `chunk_id`s its answer
was written from. That's what makes chunk-level retrieval scoring an exact
lookup instead of a fuzzy text match.

This is the RAGAS half of the eval framework; a second, custom-prompted
generator covers the other half.

Usage:
    uv run rag-eval-ragas
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import typing as t

from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    Transforms,
    apply_transforms,
    default_transforms_for_prechunked,
)

from rag.config import config
from rag.embedding.chunking import chunk_article_file

if t.TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatResult


# RAGAS's multi-hop synthesizer prefixes every context it hands the query
# writer with "<1-hop>", "<2-hop>", ... (see `make_contexts` in
# ragas.testset.synthesizers.multi_hop.base); the single-hop one passes the
# node's `page_content` through verbatim. Stripping that prefix is what lets
# the same exact-text lookup resolve chunk ids for both kinds of row.
HOP_PREFIX = re.compile(r"^<\d+-hop>\n\n")


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


def build_chunk_nodes(chunks: list[dict], slug: str) -> list[Node]:
    """Wrap this project's chunks as RAGAS knowledge-graph nodes.

    `NodeType.CHUNK` (not `DOCUMENT`) is the part that matters: it's what
    `default_transforms_for_prechunked` filters on, and what tells RAGAS
    these are already final units. A `DOCUMENT` node -- which is all
    `TestsetGenerator.generate_with_langchain_docs` can build -- would be
    run through RAGAS's own `HeadlineSplitter` first, producing contexts
    that don't correspond to anything in the Chroma index.

    `page_content` is the only property the synthesizers read;
    `document_metadata` is carried for traceability, since RAGAS doesn't
    propagate it into the generated rows (see `resolve_reference_chunk_ids`
    for how the row -> chunk id mapping is actually recovered).
    """
    return [
        Node(
            type=NodeType.CHUNK,
            properties={
                "page_content": chunk["text"],
                "document_metadata": {
                    "chunk_id": chunk["id"],
                    "source_article": slug,
                    **chunk["metadata"],
                },
            },
        )
        for chunk in chunks
    ]


def resolve_reference_chunk_ids(
    reference_contexts: t.Iterable[str], chunk_ids_by_text: dict[str, str]
) -> list[str | None]:
    """Map each of a row's `reference_contexts` back to the chunk id it came from.

    An exact dict lookup, not a similarity match: the synthesizers copy a
    node's `page_content` into `reference_contexts` verbatim (modulo the
    multi-hop `<N-hop>` prefix), and those nodes are this project's chunks.
    A `None` entry therefore means that assumption broke somewhere upstream
    -- worth counting and reporting rather than silently dropping, since
    every unresolved context is a row that can only be scored at article
    level.
    """
    return [
        chunk_ids_by_text.get(HOP_PREFIX.sub("", context))
        for context in reference_contexts
    ]


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
            pipeline_kwargs={
                "max_new_tokens": config.evaluation.huggingface.max_new_tokens
            },
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
    return TestsetGenerator(
        llm=build_generator_llm(), embedding_model=generator_embeddings
    )


def generate_for_article(
    generator: TestsetGenerator, transforms: Transforms, path: Path
) -> list[dict]:
    """Generate QA rows for one article, from that article's own chunks.

    Builds the knowledge graph by hand instead of calling
    `generate_with_langchain_docs`, which hardcodes both `NodeType.DOCUMENT`
    and the splitting `default_transforms`.
    """
    chunks = chunk_article_file(path)
    if not chunks:
        return []
    # chunk_blocks ids chunks as "{slug}::chunk-NN", so the slug is the id
    # prefix -- no need to re-read the article json just for its metadata.
    slug = chunks[0]["id"].split("::", 1)[0]

    knowledge_graph = KnowledgeGraph(nodes=build_chunk_nodes(chunks, slug))
    apply_transforms(knowledge_graph, transforms)
    generator.knowledge_graph = knowledge_graph
    # Personas are derived from the knowledge graph, but `generate()` only
    # builds them when `persona_list` is still None -- and it caches them on
    # the generator afterwards. Without this reset, the personas inferred
    # from the *first* article are reused for the whole corpus, so questions
    # about an MLOps article get written in the voice of a persona inferred
    # from a computer vision one (which is exactly what happened to the
    # previous dataset: 808 rows, one persona).
    generator.persona_list = None

    testset = generator.generate(
        testset_size=config.evaluation.ragas_questions_per_article,
        num_personas=config.evaluation.ragas_personas_per_article,
    )

    chunk_ids_by_text = {chunk["text"]: chunk["id"] for chunk in chunks}
    records = testset.to_pandas().to_dict(orient="records")
    for record in records:
        record["source_article"] = slug
        record["reference_chunk_ids"] = resolve_reference_chunk_ids(
            record["reference_contexts"], chunk_ids_by_text
        )
    return records


def main() -> int:
    paths = sorted(config.data.articles_dir.glob("*.json"))
    print(f"Found {len(paths)} articles in {config.data.articles_dir}", file=sys.stderr)

    generator = build_generator()
    # Transforms only hold references to the llm/embedding model, so one set
    # is reused across every article's graph.
    transforms = default_transforms_for_prechunked(
        llm=generator.llm, embedding_model=generator.embedding_model
    )
    output_path = config.evaluation.ragas_output_path

    rows: list[dict] = []
    unresolved_contexts = 0
    skipped: list[str] = []

    for i, path in enumerate(paths, start=1):
        try:
            records = generate_for_article(generator, transforms, path)
        except Exception as exc:  # keep going; report at the end
            # One article being un-generatable is expected, not exceptional:
            # RAGAS's LLM-backed `CustomNodeFilter` can reject every chunk of
            # a short or low-signal article, after which
            # `default_query_distribution` raises because no synthesizer has
            # a usable node cluster left. A full-corpus batch is far too
            # expensive to lose at article 100 over one such article, so skip
            # and report instead of aborting.
            skipped.append(path.name)
            print(f"[{i}/{len(paths)}] {path.name}: SKIPPED -- {exc}", file=sys.stderr)
            continue

        rows.extend(records)
        unresolved_contexts += sum(
            1
            for record in records
            for chunk_id in record["reference_chunk_ids"]
            if chunk_id is None
        )
        _write_output(rows, output_path)
        print(
            f"[{i}/{len(paths)}] {path.name}: {len(records)} questions", file=sys.stderr
        )

    print(
        f"\nGenerated {len(rows)} QA pairs from {len(paths) - len(skipped)} articles "
        f"into {output_path}.",
        file=sys.stderr,
    )
    if skipped:
        print(
            f"Skipped {len(skipped)} article(s) with no generatable chunks: "
            f"{', '.join(skipped)}",
            file=sys.stderr,
        )
    if unresolved_contexts:
        # Not fatal, but it means the verbatim-page_content assumption these
        # rows rely on no longer holds for some of them -- likely a RAGAS
        # version change. Those rows are still usable at article level.
        print(
            f"WARNING: {unresolved_contexts} reference_contexts could not be "
            "matched back to a chunk id (expected 0).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
