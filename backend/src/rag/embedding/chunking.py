"""Chunk a processed article JSON file into embeddable windows.

Sections first, merge second -- see docs/Chunking.md for the full write-up
of the approach, the decisions behind it, and what's still open.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

from rag.config import config

# The top-level split anchor: a section is an h2 heading plus everything up
# to the next one. An oversized section is recursively split on h3 instead
# of just flowing through it. See docs/Chunking.md.
SECTION_HEADING_TYPE = "h2"
SUBSECTION_HEADING_TYPE = "h3"

PARAGRAPH_TYPE = "p"

# Titan Text Embeddings V2 has no offline tokenizer -- the only exact count
# comes from `inputTextTokenCount` in a live (billed) invoke_model response.
# AWS documents ~4.7 chars/token for English as the estimation ratio; see
# docs/Chunking.md. TODO: swap to the real Bedrock count once we wire up the
# actual embedding call.
CHARS_PER_TOKEN = 4.7


def count_tokens(text: str) -> int:
    """Estimate token count via AWS's documented chars/token ratio for English."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def section_tokens(section: list[dict]) -> int:
    return sum(count_tokens(block["text"]) for block in section)


def is_heading(block_type: str) -> bool:
    """True for any heading level (h1-h6), not just the h2/h3 section
    anchors -- this is about text formatting, not chunk splitting."""
    return block_type[:1] == "h" and block_type[1:].isdigit()


def chunk_text(content: list[dict]) -> str:
    """Join a chunk's content blocks into the single string sent to the
    embedding model: a blank line around headings, a single newline
    elsewhere."""
    parts = [content[0]["text"]]
    for previous, block in pairwise(content):
        separator = (
            "\n\n"
            if is_heading(previous["type"]) or is_heading(block["type"])
            else "\n"
        )
        parts.append(separator)
        parts.append(block["text"])
    return "".join(parts)


def glue_heading_only_groups(groups: list[list[dict]]) -> list[list[dict]]:
    """A group made entirely of heading blocks (e.g. an h2 immediately
    followed by an h3, with nothing of its own in between) has no content to
    anchor a chunk boundary to -- glue it onto the group that follows,
    whatever that group turns out to be."""
    glued: list[list[dict]] = []
    for group in groups:
        if glued and all(is_heading(block["type"]) for block in glued[-1]):
            glued[-1].extend(group)
        else:
            glued.append(group)
    return glued


def group_by_heading(blocks: list[dict], heading_type: str) -> list[list[dict]]:
    """Split `blocks` into groups, each starting at a `heading_type` block and
    running until the next one. Content before the first one (or all of
    `blocks`, if `heading_type` never appears) still forms its own leading
    group -- nothing is ever dropped. See `glue_heading_only_groups` for how
    a heading-only group (no content of its own) is handled.
    """
    groups: list[list[dict]] = [[]]
    for block in blocks:
        if block["type"] == heading_type:
            groups.append([])
        groups[-1].append(block)
    return glue_heading_only_groups([group for group in groups if group])


def fallback_split(blocks: list[dict], window_tokens: int) -> list[list[dict]]:
    """Last-resort splitter for a section/subsection that's still too big
    even after h2/h3 grouping (no further heading to subdivide by, or a
    single block bigger than the window on its own).

    Pure token-windowed splitting, no heading-lookahead -- `group_by_heading`
    already guarantees nothing heading-shaped remains in `blocks` except
    possibly its own leading block. That leading heading is never flushed on
    its own, though: if it's followed by a single block big enough to blow
    the window by itself (e.g. one huge paragraph), the two stay together as
    one oversized fragment rather than orphaning the heading -- the same
    "a single block bigger than the window stays whole" acceptance already
    documented in docs/Chunking.md, just extended to a leading heading plus
    its first real block.
    """
    raw: list[list[dict]] = []
    chunk: list[dict] = []
    tokens = 0
    for block in blocks:
        block_tokens = count_tokens(block["text"])
        only_heading_so_far = chunk and all(is_heading(b["type"]) for b in chunk)
        if chunk and not only_heading_so_far and tokens + block_tokens > window_tokens:
            raw.append(chunk)
            chunk = []
            tokens = 0
        chunk.append(block)
        tokens += block_tokens
    if chunk:
        raw.append(chunk)
    return raw


def split_oversized_section(
    section: list[dict], ceiling_tokens: int, window_tokens: int
) -> list[list[dict]]:
    """Return `section` as a single atomic unit if it fits under
    `ceiling_tokens`; otherwise recurse into h3-delimited subsections, and
    fall back to plain token-windowed splitting for any subsection still too
    big (e.g. no h3 inside it at all)."""
    if section_tokens(section) <= ceiling_tokens:
        return [section]

    units: list[list[dict]] = []
    for subsection in group_by_heading(section, SUBSECTION_HEADING_TYPE):
        if section_tokens(subsection) <= ceiling_tokens:
            units.append(subsection)
        else:
            units.extend(fallback_split(subsection, window_tokens))
    return units


def peel_trailing_headings(units: list[list[dict]]) -> list[list[dict]]:
    """Split off a unit's trailing run of heading blocks into their own unit,
    if any -- e.g. a whole (non-oversized) section that legitimately ends on
    an h3 with nothing under it in the source content, right before the next
    section's own heading starts. That section was never split by h3 at all
    (it fit under the ceiling as one piece), so the heading-only tail is
    buried inside one atomic unit rather than being its own unit -- peeling
    it off here is what lets the cross-unit glue pass (see
    `glue_heading_only_groups`, used in `chunk_blocks`) attach it to
    whatever comes next instead of leaving it stuck as a bare trailing
    heading."""
    peeled: list[list[dict]] = []
    for unit in units:
        split_at = len(unit)
        while split_at > 0 and is_heading(unit[split_at - 1]["type"]):
            split_at -= 1
        if 0 < split_at < len(unit):
            peeled.append(unit[:split_at])
            peeled.append(unit[split_at:])
        else:
            peeled.append(unit)
    return peeled


def merge_units(units: list[list[dict]], ceiling_tokens: int) -> list[list[dict]]:
    """Greedily merge adjacent atomic units (whole sections/subsections, or
    fallback-split fragments of an oversized one) into chunks up to
    `ceiling_tokens`. A unit is never split apart here -- that already
    happened, if needed, before this point."""
    raw_chunks: list[list[dict]] = []
    chunk: list[dict] = []
    tokens = 0
    for unit in units:
        unit_tokens = section_tokens(unit)
        if chunk and tokens + unit_tokens > ceiling_tokens:
            raw_chunks.append(chunk)
            chunk = []
            tokens = 0
        chunk.extend(unit)
        tokens += unit_tokens
    if chunk:
        raw_chunks.append(chunk)
    return raw_chunks


def merge_tiny_chunks(
    raw_chunks: list[list[dict]], min_tokens: int
) -> list[list[dict]]:
    """Fold any chunk under `min_tokens` into the previous chunk.

    The first chunk is left as-is even if it's tiny -- there's no previous
    chunk to merge it into.
    """
    merged: list[list[dict]] = []
    for chunk in raw_chunks:
        chunk_tokens = section_tokens(chunk)
        if merged and chunk_tokens < min_tokens:
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)
    return merged


def chunk_blocks(
    blocks: list[dict],
    title: str,
    author: str,
    slug: str,
    window_tokens: int = config.chunking.window_tokens,
    heading_search_allowance_tokens: int = config.chunking.heading_search_allowance_tokens,
    min_chunk_tokens: int = config.chunking.min_chunk_tokens,
) -> list[dict]:
    """Split content blocks into chunks of roughly `window_tokens` each.

    Sections first, merge second: `blocks` is split into whole h2 sections
    (recursing into h3 only for an oversized section, and falling back to
    plain token-windowed splitting only if even that's still too big), then
    adjacent sections are greedily merged back together up to a ceiling of
    `window_tokens + heading_search_allowance_tokens`. A section is always
    atomic once formed, so a chunk can never end on a bare trailing heading
    -- a heading and its own content are never separated. Content before the
    first h2 (or all of it, if there's no h2 at all) still forms its own
    leading section rather than being dropped.

    Returns one `{"id": ..., "metadata": {...}, "content": [...], "text": ...}`
    dict per chunk. `id` is `{slug}::chunk-{NN}`, deterministic across reruns
    so an upsert into Chroma updates rather than duplicates. `metadata`
    (title, author) is never embedded -- it's for citation/filtering once a
    chunk is retrieved. `content` is the structured block list (still useful
    for citation by block `number`); `text` is `content`'s blocks joined into
    the single string actually sent to the embedding model, with a blank
    line around headings. `content` still gets the title as a leading block
    too: that copy helps the embedding itself (a short paragraph is often
    ambiguous on its own), which non-embedded metadata can't do.

    `heading_search_allowance_tokens` and `min_chunk_tokens` default from
    config but are real parameters (not just read off `config` internally) so
    tests can scale them down alongside a smaller `window_tokens` instead of
    being stuck with production-sized values.
    """
    ceiling_tokens = window_tokens + heading_search_allowance_tokens

    units: list[list[dict]] = []
    for section in group_by_heading(blocks, SECTION_HEADING_TYPE):
        units.extend(split_oversized_section(section, ceiling_tokens, window_tokens))

    # A whole (non-oversized) section can still legitimately end on a bare
    # heading -- peel that off into its own unit, then glue it forward onto
    # whatever comes next, across section boundaries if need be.
    units = peel_trailing_headings(units)
    units = glue_heading_only_groups(units)

    raw_chunks = merge_units(units, ceiling_tokens)
    raw_chunks = merge_tiny_chunks(raw_chunks, min_chunk_tokens)

    # TODO(Chunking.md): revisit adding `description` to every chunk too.
    chunks: list[dict] = []
    for idx, chunk in enumerate(raw_chunks):
        overlap = []
        if idx > 0:
            previous_paragraph = next(
                (
                    b
                    for b in reversed(raw_chunks[idx - 1])
                    if b["type"] == PARAGRAPH_TYPE
                ),
                None,
            )
            if previous_paragraph is not None:
                overlap = [previous_paragraph]
        content = [{"type": "title", "text": title}, *overlap, *chunk]
        chunks.append(
            {
                "id": f"{slug}::chunk-{idx:02d}",
                "metadata": {"title": title, "author": author},
                "content": content,
                "text": chunk_text(content),
            }
        )

    return chunks


def chunk_article_file(
    path: Path,
    window_tokens: int = config.chunking.window_tokens,
    heading_search_allowance_tokens: int = config.chunking.heading_search_allowance_tokens,
    min_chunk_tokens: int = config.chunking.min_chunk_tokens,
) -> list[dict]:
    """Chunk one `data/articles/*.json` file, as written by rag-process."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    metadata = doc["metadata"]
    return chunk_blocks(
        doc["content"],
        metadata["title"],
        metadata["author"],
        metadata["slug"],
        window_tokens,
        heading_search_allowance_tokens,
        min_chunk_tokens,
    )
