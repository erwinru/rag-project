# Chunking

First basic version. Implementation: [`src/rag/embedding/chunking.py`](../src/rag/embedding/chunking.py).

## Input / output

- **Input:** one `data/articles/*.json` file, as written by `rag-process`
  (`{"metadata": {...}, "content": [blocks]}`). Each content block also
  carries a stable 1-based `number` (assigned once, in `rag-process`), so a
  block can later be cited back to the user as an exact reference (article
  title + number).
- **Output:** `list[dict]`, one
  `{"id": ..., "metadata": {...}, "content": [...], "text": ...}` per chunk:
  - `id`: `{slug}::chunk-{NN}`, zero-indexed and deterministic across
    reruns, for upserting into Chroma (see Decisions below).
  - `metadata`: `{"title": ..., "author": ...}`. Never embedded -- for
    citation/filtering once a chunk is retrieved, not for search.
  - `content`: the structured block list, in document order -- kept around
    for citation by block `number`, not what's sent to the embedding model.
    The first block is always a synthetic `{"type": "title", "text": ...}`
    block (see Decisions below for why title appears in both places).
  - `text`: `content`'s blocks joined into the single string actually sent
    to the embedding model (see Decisions below).

## Algorithm (v2 -- sections first, merge second)

The original v1 walked blocks token-by-token and only looked for a heading to
align to *after* already going over the window. That meant a heading cheap
enough to fit alongside the preceding paragraphs got silently absorbed into
the *previous* section's chunk, orphaned from its own content the moment the
next (real) paragraph overflowed the window one block later. v2 restructures
around a simple principle instead: **a heading and its content are grouped
into one atomic section before any token budget is considered, so a chunk
can never end on a bare heading.**

1. **Group into sections.** Split `content` into whole `h2` sections (a
   heading plus everything up to the next `h2`) via `group_by_heading`.
   Content before the first `h2` (or all of it, if there's no `h2` at all)
   still forms its own leading section -- nothing is dropped.
2. **Split any oversized section.** A section over `ceiling_tokens`
   (`window_tokens + heading_search_allowance_tokens`, 700 by default)
   recurses into `h3`-delimited subsections (`split_oversized_section`). A
   subsection still too big (no `h3` inside it, or one block bigger than the
   window on its own) falls back to plain token-windowed splitting
   (`fallback_split`) -- the *only* place paragraph-level cutting still
   happens, and only as a last resort.
3. **Normalize heading-only fragments.** Two ways a "unit" can end up with
   no content of its own, and how each is caught:
   - `group_by_heading` itself glues a group made entirely of heading blocks
     onto the group that follows (`glue_heading_only_groups`) -- e.g. an
     `h2` immediately followed by an `h3`, nothing directly under the `h2`.
   - A whole section can still legitimately *end* on a bare heading with
     nothing under it before the next section starts (it was never split by
     `h3` at all, since it fit under the ceiling as one piece). `peel_trailing_headings`
     splits that bare tail off into its own unit; a second `glue_heading_only_groups`
     pass then attaches it to whatever comes next, across section boundaries
     if need be.
4. **Merge adjacent units up to the ceiling** (`merge_units`) -- the same
   greedy-fill idea as v1, just retargeted at whole sections/subsections
   instead of individual blocks. A unit is never split apart here; that
   already happened (if needed) in step 2.
5. **Fold tiny leftovers** (`merge_tiny_chunks`, unchanged from v1): any
   chunk under `min_chunk_tokens` (50) gets merged into the previous one, so
   a small leftover at the end of a walk doesn't stand alone. The first
   chunk is left as-is even if it's tiny -- nothing precedes it.
6. **Overlap, title, metadata, id** (unchanged from v1): a 1-paragraph
   overlap from the previous chunk's last `p` block, the article title
   prepended to `content` as a block, `{"title", "author"}` as non-embedded
   `metadata`, and a deterministic `id`.

## Decisions

- **Token counting is approximate** -- we're using AWS Bedrock's Titan Text
  Embeddings V2 for the actual embedding step, but Titan has no offline
  tokenizer (unlike e.g. OpenAI's `tiktoken`); the only exact count is the
  `inputTextTokenCount` field returned by a live, billed `invoke_model` call.
  Calling that per unit would inject live AWS calls into what's otherwise a
  pure local function, so for now we estimate via AWS's documented ~4.7
  chars/token ratio for English (`CHARS_PER_TOKEN` in `chunking.py`). TODO:
  swap to the real Bedrock count once we wire up the actual embedding call.
- **`h2` sections, `h3` subsections; `h4`-`h6` don't get their own recursion
  level.** `h2` is the primary split; oversized sections recurse into `h3`
  specifically so a large section subdivides along its own subheadings
  rather than falling straight to paragraph-level fallback splitting.
  Deeper levels weren't common or large enough in the corpus to warrant a
  third recursion level yet -- see the checklist below.
- **The merge ceiling is `window_tokens + heading_search_allowance_tokens`
  (700 by default), reused from v1's lookahead-allowance concept.** It now
  means something slightly different: not "how far to look ahead for the
  next heading" but "how big a section/subsection can be before it's forced
  through `h3`-recursion or fallback splitting, and how much adjacent units
  can merge up to." Same field, same value, repurposed role -- didn't seem
  worth introducing a second config value for what's conceptually the same
  slack budget.
- **A chunk can never end on a bare heading -- enforced in three places,**
  because there turned out to be three distinct ways one could sneak in
  (found by chunking a real article and checking every chunk boundary by
  hand, then confirmed by scanning all 142 articles):
  1. An `h2` immediately followed by an `h3` with nothing directly under the
     `h2` -- `group_by_heading`'s own `glue_heading_only_groups` pass
     catches this while forming groups.
  2. A whole (non-oversized) section that itself legitimately ends on a
     bare subheading right before the *next* section's heading starts (never
     split by `h3` at all, since it fit under the ceiling as one piece) --
     `peel_trailing_headings` splits that bare tail off into its own unit
     first, so the cross-unit `glue_heading_only_groups` pass (run again,
     across section boundaries this time) can catch it.
  3. A leading heading inside `fallback_split`, followed by a single block
     big enough to blow the window by itself (e.g. one huge paragraph) --
     `fallback_split` now refuses to flush a fragment that's heading-only,
     so the two stay together as one oversized fragment instead.
  Confirmed via a full sweep of the real corpus: 0 chunks end on a heading
  across all 142 articles / 680 chunks (was 15, across several different
  articles, before these three fixes).
- **Title lives in both `content` (embedded) and `metadata` (not embedded),
  deliberately.** A short paragraph is often ambiguous on its own, so
  embedding the title alongside it gives the embedding real semantic signal
  about which article/topic it belongs to (the standard "contextual
  retrieval" trick) -- metadata can't do that, since it's never part of the
  vector. `metadata.title` is there for citation/filtering after retrieval.
- **Author lives only in `metadata`, never embedded.** Unlike title, author
  has no semantic value for retrieval -- a query doesn't get closer to the
  right chunk because `"Author: ML6"` is baked into the vector. Since
  metadata isn't embedded, there's no cost to attaching it to every chunk
  (not just the first, as the original chunking-strategy discussion first
  proposed) -- a chunk retrieved from the middle of an article still needs
  an author to cite.
- **References get no special handling at all.** They're not detected,
  tagged, or treated differently -- they just flow through as ordinary
  `p`/heading content in whichever chunk they land in (typically the last,
  since they're at the end of the doc). Deliberately simple: nothing to
  build here.
- **Description is deliberately not added yet** -- see TODO below.
- **Overlap is exactly 1 paragraph, taken from the previous chunk's own
  content** (its last `p` block, before that chunk got its own overlap/title
  prepended) -- so overlap never compounds across more than one chunk
  boundary. If a chunk has no `p` block at all (degenerate case), it's
  skipped rather than forcing something non-paragraph into the overlap slot.
  The overlapped paragraph's tokens aren't counted against the *next*
  chunk's window budget, since the window size is decided before overlap is
  added -- so actual chunk size can run a bit over `window_tokens` once
  overlap is included. That's an accepted, minor overshoot for now.
- **Tiny-chunk merging folds backward into the previous chunk, never
  forward**, and runs *before* overlap/title/metadata are added (so it
  merges on genuine new-content size, not padded by overlap). `50` tokens
  was picked as roughly "a sentence or two" -- small enough to only catch
  genuinely degenerate leftovers, not legitimate short sections. Confirmed
  against the real corpus: 5 of 142 articles have a chunk merged this way.
- **`heading_search_allowance_tokens` and `min_chunk_tokens` are real
  parameters of `chunk_blocks`/`chunk_article_file`, not just internal reads
  of `config.chunking`** (their defaults still come from there). Without
  this, a test using a smaller `window_tokens` would still inherit the
  production-sized allowance/threshold and the two would be badly out of
  proportion -- e.g. a 50-token test window paired with the real 200-token
  allowance means the lookahead search reaches nearly any heading in a small
  fixture, regardless of what the test is trying to isolate.
- **Chunk id is `{slug}::chunk-{NN}`**, `NN` being the chunk's position
  (zero-indexed) *after* tiny-chunk merging, so ids stay contiguous and
  stable for a given article as long as its content doesn't change. `slug`
  is threaded through from the article's metadata rather than derived from
  anything chunk-specific, since it's the one thing in the article JSON
  that's already guaranteed unique and filename-safe. Confirmed 0 collisions
  across all 711 chunks in the real corpus, and calling `chunk_blocks` twice
  on the same input yields identical ids.
- **`text` joins `content`'s block texts with a blank line (`"\n\n"`) around
  any heading (`h1`-`h6`, via `is_heading()`) and a single `"\n"` between two
  consecutive non-heading blocks** (mostly paragraph-to-paragraph). `is_heading`
  checks any heading level, not just the `h2`/`h3` chunk-boundary anchors --
  formatting and chunk-splitting are separate concerns, and a chunk's
  `content` can still carry `h4`-`h6` blocks that were never boundary
  candidates. No other formatting yet (no markdown heading prefixes, no
  special marker on the overlap paragraph) -- see "Later experiments" for
  what a richer format could still look like.

## Testing

[`tests/test_chunking.py`](../tests/test_chunking.py) uses `window_tokens=50`
(`heading_search_allowance_tokens=20`, `min_chunk_tokens=5` -- the same
500:200:50 ratio as production, just scaled down) so fixtures can stay a
handful of blocks instead of hundreds of words. A `para()`/`heading()` helper
builds blocks with an exact token count (verified against the real
`count_tokens`, not hand-derived from the chars/token ratio), so tests assert
on precise boundary behavior instead of approximate sizes.

Covered at the `chunk_blocks` level: no split needed, h2-anchored split, h3
fallback when a section is oversized with no h2 nearby, plain fallback
splitting when a section has no heading to subdivide by at all,
no-heading-at-all and intro-before-first-h2 (nothing gets dropped), overlap
correctness across every boundary, tiny-chunk merging (including the
un-mergeable first-chunk case), a single block bigger than the window, block
`number` pass-through, chunk id format/determinism, and the
blank-line-around-headings join.

Each of the three heading-orphaning fixes also has a direct unit test on its
own function (`group_by_heading`, `glue_heading_only_groups`,
`peel_trailing_headings`, `fallback_split`'s leading-heading protection),
plus one stress test (`test_no_chunk_ever_ends_on_a_bare_heading`) that
exercises all three at once in a single fixture and asserts both invariants:
no chunk's `content` ends on a heading, and no block number gets dropped.
`chunk_article_file`'s file-reading path is covered end to end too.

## TODO

- [ ] Revisit adding `description` to every chunk (title is in; description
      was part of the original design but deferred here to keep v1 minimal).

## Checklist / open edge cases

- [ ] Real Titan token counts (via a live Bedrock `invoke_model` call reading
      `inputTextTokenCount`) instead of the chars/token estimate -- likely
      makes sense once the actual embedding call is wired up, so the count
      comes for free alongside the embedding rather than as an extra call
- [ ] Evaluate the window size (500 tokens) and merge ceiling (700 tokens)
      against actual retrieval quality once we can test end-to-end
- [ ] A third recursion level (`h4`) for a subsection that's still oversized
      after splitting on `h3` -- currently falls straight to paragraph-level
      fallback splitting instead

## Later experiments (need real data/retrieval to decide, not guesswork)

- **A single block bigger than the window.** Currently always kept whole
  (never split), so it can overshoot the window on its own -- a test locks
  in that this is what happens today, but *whether that's actually the right
  behavior* is unresolved. The alternative (splitting an oversized paragraph,
  presumably with its own overlap) trades a cleaner window size against
  chopping a paragraph mid-thought, and there's no way to know which hurts
  or helps retrieval without testing both against real queries. Deliberately
  not deciding this now -- revisit once we can run actual retrieval
  experiments and compare.
- **A richer `text` format than the current blank-line-around-headings
  join.** Headings now get visual separation, but the *type* information
  (this line was a heading, that one was a paragraph) still isn't in the
  text itself -- worth testing markdown-style heading prefixes (`## `/`### `)
  on top of the blank line, or marking the overlap paragraph so it doesn't
  read as accidentally duplicated content. Same reasoning as the
  oversized-block case above: plausible in either direction, not worth
  guessing at without a retrieval eval to check against.
