import json

from rag.embedding.chunking import (
    chunk_article_file,
    chunk_blocks,
    chunk_text,
    count_tokens,
    fallback_split,
    glue_heading_only_groups,
    group_by_heading,
    is_heading,
    merge_tiny_chunks,
    peel_trailing_headings,
)

# Scaled-down, proportional to the production config.toml values
# (500 / 200 / 50) so tests read close to real behavior without needing
# hundreds of words of fixture text per case.
WINDOW = 50
ALLOWANCE = 20
MIN_CHUNK = 5

TITLE = "Test Article"
AUTHOR = "Ada Lovelace"
SLUG = "test-article"


def para(number: int, tokens: int, tag: str = "") -> dict:
    """A `p` block with an exact token count (via the real `count_tokens`)."""
    text = _text_with_tokens(tokens, tag)
    return {"number": number, "type": "p", "text": text}


def heading(number: int, level: str, text: str) -> dict:
    return {"number": number, "type": level, "text": text}


def _text_with_tokens(n: int, tag: str) -> str:
    length = max(1, round(n * 4.7))
    text = f"{tag} " + "x" * length
    while count_tokens(text) != n:
        length += 1 if count_tokens(text) < n else -1
        text = f"{tag} " + "x" * length
    return text


def chunk(blocks: list[dict]) -> list[dict]:
    return chunk_blocks(
        blocks,
        TITLE,
        AUTHOR,
        SLUG,
        window_tokens=WINDOW,
        heading_search_allowance_tokens=ALLOWANCE,
        min_chunk_tokens=MIN_CHUNK,
    )


def content_texts(c: dict) -> list[str]:
    return [b["text"] for b in c["content"]]


def test_everything_fits_in_one_chunk():
    blocks = [heading(1, "h2", "Intro"), para(2, 20), para(3, 20)]
    result = chunk(blocks)

    assert len(result) == 1
    assert result[0]["metadata"] == {"title": TITLE, "author": AUTHOR}
    assert content_texts(result[0])[0] == TITLE
    assert content_texts(result[0])[1:] == [b["text"] for b in blocks]


def test_splits_at_the_next_h2_when_it_fits_in_the_allowance():
    # 30 + 30 tokens overflows the 50-token window; the h2 sits 5 tokens
    # later, well within the 20-token allowance, so it should anchor the cut.
    blocks = [
        heading(1, "h2", "First"),
        para(2, 30),
        para(3, 30),
        heading(4, "h2", "Second"),
        para(5, 10),
    ]
    result = chunk(blocks)

    assert len(result) == 2
    # first chunk absorbed the overflowing paragraph to reach the heading
    assert [b["number"] for b in result[0]["content"] if "number" in b] == [1, 2, 3]
    # second chunk starts right at the heading (plus its 1-paragraph overlap)
    second_numbers = [b["number"] for b in result[1]["content"] if "number" in b]
    assert second_numbers == [3, 4, 5]  # 3 is the carried-over overlap paragraph


def test_falls_back_to_h3_when_no_h2_is_within_the_allowance():
    # No h2 anywhere nearby, but an h3 sits within the allowance -- large
    # sections should subdivide along their own subheadings.
    blocks = [
        heading(1, "h2", "Big section"),
        para(2, 30),
        para(3, 30),
        heading(4, "h3", "Subsection"),
        para(5, 10),
    ]
    result = chunk(blocks)

    assert len(result) == 2
    # second chunk = [overlap paragraph, the h3 itself, its trailing paragraph]
    second_types = [b["type"] for b in result[1]["content"] if b["type"] != "title"]
    assert second_types == ["p", "h3", "p"]


def test_falls_back_to_last_paragraph_when_no_heading_fits():
    # No h3 inside this h2 section at all, so it falls all the way through to
    # plain token-windowed splitting -- fallback_split cuts every ~50 tokens,
    # but merge_units then re-merges adjacent fragments back up to the
    # 70-token ceiling, so chunk 0 ends up with 3 blocks, not 2.
    blocks = [
        heading(1, "h2", "Section"),
        para(2, 30),
        para(3, 30),
        para(4, 30),
        para(5, 30),
    ]
    result = chunk(blocks)

    assert len(result) >= 2
    first_content = [b["text"] for b in result[0]["content"] if b["type"] != "title"]
    assert first_content == [blocks[0]["text"], blocks[1]["text"], blocks[2]["text"]]


def test_no_heading_at_all_still_chunks_from_the_start():
    blocks = [para(1, 30), para(2, 30), para(3, 30)]
    result = chunk(blocks)

    assert len(result) >= 2
    assert result[0]["content"][1]["text"] == blocks[0]["text"]


def test_intro_paragraph_before_first_h2_is_kept():
    blocks = [para(1, 10), heading(2, "h2", "Intro"), para(3, 10)]
    result = chunk(blocks)

    numbers = [b["number"] for b in result[0]["content"] if "number" in b]
    assert numbers == [1, 2, 3]


def test_overlap_repeats_the_previous_chunks_last_paragraph():
    blocks = [
        heading(1, "h2", "First"),
        para(2, 45),
        heading(3, "h2", "Second"),
        para(4, 45),
        heading(5, "h2", "Third"),
        para(6, 10),
    ]
    result = chunk(blocks)

    assert len(result) >= 2
    for i in range(len(result) - 1):
        this_chunk_paragraphs = [b for b in result[i]["content"] if b["type"] == "p"]
        next_chunk_first_real_block = result[i + 1]["content"][1]
        assert next_chunk_first_real_block["text"] == this_chunk_paragraphs[-1]["text"]

    # the very first chunk gets no overlap -- just title + its own content
    first_types = [b["type"] for b in result[0]["content"]]
    assert first_types[0] == "title"


def test_tiny_trailing_chunk_is_merged_into_the_previous_one():
    blocks = [
        heading(1, "h2", "Section"),
        para(2, 45),
        heading(3, "h2", "Tiny tail"),
        para(4, 2),  # well under MIN_CHUNK (5)
    ]
    result = chunk(blocks)

    assert len(result) == 1
    numbers = [b["number"] for b in result[0]["content"] if "number" in b]
    assert numbers == [1, 2, 3, 4]


def test_tiny_first_chunk_is_kept_when_there_is_nothing_to_merge_into():
    # merge_tiny_chunks in isolation: first chunk tiny, nothing before it.
    raw = [[{"type": "p", "text": "x"}], [{"type": "p", "text": "y" * 500}]]
    result = merge_tiny_chunks(raw, min_tokens=MIN_CHUNK)

    assert len(result) == 2
    assert result[0] == raw[0]


def test_single_block_larger_than_the_window_is_kept_whole():
    huge = para(1, 200)
    blocks = [heading(2, "h2", "Small"), para(3, 10), huge]
    result = chunk(blocks)

    all_texts = {t for c in result for t in content_texts(c)}
    assert huge["text"] in all_texts


def test_block_number_passes_through_unchanged():
    blocks = [heading(1, "h2", "Section"), para(2, 10)]
    result = chunk(blocks)

    numbered = [b for b in result[0]["content"] if b["type"] != "title"]
    assert numbered[0]["number"] == 1
    assert numbered[1]["number"] == 2


def test_chunk_article_file_reads_metadata_and_content(tmp_path):
    doc = {
        "metadata": {
            "title": "From Disk",
            "author": "Grace Hopper",
            "slug": "from-disk",
        },
        "content": [heading(1, "h2", "Section"), para(2, 10)],
    }
    path = tmp_path / "001_from-disk.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    result = chunk_article_file(
        path,
        window_tokens=WINDOW,
        heading_search_allowance_tokens=ALLOWANCE,
        min_chunk_tokens=MIN_CHUNK,
    )

    assert len(result) == 1
    assert result[0]["id"] == "from-disk::chunk-00"
    assert result[0]["metadata"] == {"title": "From Disk", "author": "Grace Hopper"}
    assert result[0]["content"][0] == {"type": "title", "text": "From Disk"}


def test_chunk_ids_are_slug_scoped_and_zero_indexed():
    blocks = [
        heading(1, "h2", "First"),
        para(2, 45),
        heading(3, "h2", "Second"),
        para(4, 45),
        heading(5, "h2", "Third"),
        para(6, 10),
    ]
    result = chunk(blocks)

    assert len(result) >= 2
    assert [c["id"] for c in result] == [
        f"{SLUG}::chunk-{i:02d}" for i in range(len(result))
    ]


def test_chunk_ids_are_deterministic_across_reruns():
    blocks = [heading(1, "h2", "Section"), para(2, 10)]

    assert [c["id"] for c in chunk(blocks)] == [c["id"] for c in chunk(blocks)]


def test_chunk_text_puts_a_blank_line_around_headings():
    content = [
        {"type": "title", "text": "T"},
        {"type": "h2", "text": "H2"},
        {"type": "p", "text": "P1"},
        {"type": "p", "text": "P2"},
        {"type": "h3", "text": "H3"},
        {"type": "p", "text": "P3"},
    ]

    # blank line (\n\n) whenever either side of the join is a heading;
    # a single \n between two consecutive paragraphs
    assert chunk_text(content) == "T\n\nH2\n\nP1\nP2\n\nH3\n\nP3"


def test_every_chunks_text_field_matches_its_own_content():
    # Each section is under the ceiling alone, but the two together aren't,
    # so they stay as separate chunks rather than merging.
    blocks = [
        heading(1, "h2", "First"),
        para(2, 60),
        heading(3, "h2", "Second"),
        para(4, 60),
    ]
    result = chunk(blocks)

    assert len(result) >= 2
    for c in result:
        assert c["text"] == chunk_text(c["content"])


def test_group_by_heading_glues_a_heading_immediately_followed_by_another():
    # h2 with nothing under it before an h3 starts -- the h2 alone would
    # otherwise become its own heading-only group.
    blocks = [heading(1, "h2", "Big section"), heading(2, "h3", "Sub"), para(3, 10)]

    groups = group_by_heading(blocks, "h3")

    assert len(groups) == 1
    assert [b["number"] for b in groups[0]] == [1, 2, 3]


def test_group_by_heading_keeps_normal_sections_separate():
    blocks = [
        heading(1, "h2", "First"),
        para(2, 10),
        heading(3, "h2", "Second"),
        para(4, 10),
    ]

    groups = group_by_heading(blocks, "h2")

    assert [[b["number"] for b in g] for g in groups] == [[1, 2], [3, 4]]


def test_group_by_heading_keeps_leading_content_and_no_heading_at_all():
    assert [b["number"] for b in group_by_heading([para(1, 10)], "h2")[0]] == [1]

    blocks = [para(1, 10), heading(2, "h2", "First"), para(3, 10)]
    groups = group_by_heading(blocks, "h2")
    assert [[b["number"] for b in g] for g in groups] == [[1], [2, 3]]


def test_glue_heading_only_groups_merges_forward():
    groups = [
        [heading(1, "h2", "A")],
        [heading(2, "h2", "B"), para(3, 10)],
    ]

    result = glue_heading_only_groups(groups)

    assert len(result) == 1
    assert [b["number"] for b in result[0]] == [1, 2, 3]


def test_glue_heading_only_groups_leaves_a_trailing_one_alone():
    # nothing follows the heading-only group, so there's nothing to glue it to
    groups = [[heading(1, "h2", "A"), para(2, 10)], [heading(3, "h2", "B")]]

    result = glue_heading_only_groups(groups)

    assert result == groups


def test_peel_trailing_headings_splits_off_the_trailing_run():
    unit = [para(1, 10), heading(2, "h3", "Resources")]

    result = peel_trailing_headings([unit])

    assert [[b["number"] for b in u] for u in result] == [[1], [2]]


def test_peel_trailing_headings_leaves_units_without_a_bare_tail_alone():
    unit = [heading(1, "h2", "A"), para(2, 10)]

    assert peel_trailing_headings([unit]) == [unit]


def test_peel_trailing_headings_leaves_an_all_heading_unit_whole():
    unit = [heading(1, "h2", "A"), heading(2, "h3", "B")]

    assert peel_trailing_headings([unit]) == [unit]


def test_fallback_split_keeps_a_leading_heading_with_its_first_block_even_if_oversized():
    # No further heading inside this span, and the paragraph alone already
    # blows the window -- the heading must not be flushed on its own.
    blocks = [heading(1, "h2", "Section"), para(2, 200)]

    result = fallback_split(blocks, window_tokens=WINDOW)

    assert len(result) == 1
    assert [b["number"] for b in result[0]] == [1, 2]


def test_no_chunk_ever_ends_on_a_bare_heading():
    # Stress fixture exercising all three fixed cases at once: an oversized
    # section whose h2 is immediately followed by h3 (bug #1), a whole small
    # section that legitimately ends on a bare h3 right before the next
    # section's h2 (bug #2), and a heading immediately followed by one huge
    # paragraph inside a fallback split (bug #3).
    blocks = [
        heading(1, "h2", "Oversized, h2 then h3 with nothing between"),
        heading(2, "h3", "Sub"),
        para(3, 30),
        para(4, 30),
        heading(5, "h2", "Small section ending on a bare h3"),
        para(6, 10),
        heading(7, "h3", "Resources"),
        heading(8, "h2", "Section with a huge paragraph"),
        para(9, 200),
    ]
    result = chunk(blocks)

    for c in result:
        assert not is_heading(c["content"][-1]["type"]), c

    # nothing got dropped: every block number shows up somewhere
    seen = {b["number"] for c in result for b in c["content"] if "number" in b}
    assert seen == {b["number"] for b in blocks}
