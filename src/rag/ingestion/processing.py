"""Parse cached raw HTML (from rag.ingestion.scrape) into structured articles.

Metadata comes from the JSON-LD `Article` block the CMS embeds. The body is
pulled from trafilatura's XML tree (not its markdown output) since that XML
already tags each element -- headings carry a `rend` level, paragraphs are
`p` -- so blocks are read off directly instead of re-derived with regex over
markdown. Lists, quotes, and tables are folded into paragraph blocks.

Reads data/pages.csv (written by rag-scrape) plus the data/raw_html/ files it
points at. Everything lands under data/:
    data/articles/NNN_<slug>.json  {"metadata": {...}, "content": [blocks]}
    data/articles.csv              one row per article with the remaining metadata

Usage:
    uv run rag-process
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import trafilatura
from lxml import etree
from selectolax.parser import HTMLParser

from rag.config import config

ARTICLES_CSV_COLUMNS = [
    "article_number",
    "url",
    "slug",
    "author",
    "date_published",
    "date_modified",
    "image",
    "lastmod",
    "word_count",
    "text_file",
]

# Heading levels trafilatura tags in its XML output; a `<head>` without one of
# these `rend` values is a non-heading widget (e.g. a collapsible glossary
# blurb) and is folded into a paragraph block like everything else below.
HEADING_RENDS = {"h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass
class Page:
    article_number: int
    url: str
    slug: str
    lastmod: str | None
    html_file: Path


@dataclass
class Article:
    url: str
    slug: str
    title: str | None
    description: str | None
    author: str | None
    date_published: str | None
    date_modified: str | None
    image: str | None
    lastmod: str | None
    word_count: int
    blocks: list[dict]
    scraped_at: str


def load_pages(pages_csv: Path, raw_html_dir: Path) -> list[Page]:
    with pages_csv.open(encoding="utf-8") as f:
        return [
            Page(
                article_number=int(row["article_number"]),
                url=row["url"],
                slug=row["slug"],
                lastmod=row["lastmod"] or None,
                html_file=raw_html_dir / row["html_file"],
            )
            for row in csv.DictReader(f)
        ]


def element_text(element: etree._Element) -> str:
    """Flatten an element's text (including nested tags like <lb/>) to one string."""
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def blocks_from_html(html: str) -> list[dict]:
    """Read typed content blocks off trafilatura's XML tree, in document order.

    Headings become `{"type": "hN", "text": ...}`; everything else -- plain
    paragraphs, and lists/quotes/tables flattened to one paragraph each --
    becomes `{"type": "p", "text": ...}`.
    """
    xml = trafilatura.extract(
        html,
        output_format="xml",
        include_comments=False,
        include_tables=True,
        include_images=False,
    )
    if not xml:
        return []

    main = etree.fromstring(xml.encode("utf-8")).find("main")
    if main is None:
        return []

    blocks = []
    for element in main:
        if element.tag == "head" and element.get("rend") in HEADING_RENDS:
            text = element_text(element)
            if text:
                blocks.append({"type": element.get("rend"), "text": text})
        elif element.tag == "list":
            text = "; ".join(t for t in (element_text(item) for item in element) if t)
            if text:
                blocks.append({"type": "p", "text": text})
        elif element.tag in ("p", "head", "quote", "table"):
            text = element_text(element)
            if text:
                blocks.append({"type": "p", "text": text})
    return blocks


def extract_article_ld(html: str) -> dict:
    """Pull the JSON-LD `Article` block out of the page.

    The page carries several ld+json blocks (site navigation, breadcrumbs, and
    one that isn't valid JSON at all), so each is parsed defensively and we keep
    the one that looks like the article.
    """
    tree = HTMLParser(html)
    for node in tree.css('script[type="application/ld+json"]'):
        raw = (node.text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        candidates = data if isinstance(data, list) else [data]
        for candidate in candidates:
            if isinstance(candidate, dict) and "headline" in candidate:
                return candidate
    return {}


def meta_content(
    tree: HTMLParser, *, prop: str | None = None, name: str | None = None
) -> str | None:
    selector = f'meta[property="{prop}"]' if prop else f'meta[name="{name}"]'
    node = tree.css_first(selector)
    if node is None:
        return None
    content = node.attributes.get("content")
    return content.strip() if content else None


def flatten_name(value: object) -> str | None:
    """JSON-LD `author` may be a string, an object, or a list of either."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        return flatten_name(value.get("name"))
    if isinstance(value, list):
        names = [n for n in (flatten_name(v) for v in value) if n]
        return ", ".join(names) or None
    return None


def flatten_image(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        return flatten_image(value.get("url"))
    if isinstance(value, list):
        for item in value:
            found = flatten_image(item)
            if found:
                return found
    return None


def parse_article(page: Page) -> Article:
    html = page.html_file.read_text(encoding="utf-8")
    ld = extract_article_ld(html)
    tree = HTMLParser(html)
    blocks = blocks_from_html(html)

    title = flatten_name(ld.get("headline")) or meta_content(tree, prop="og:title")
    if not title:
        heading = tree.css_first("h1")
        title = heading.text(strip=True) if heading else None

    description = flatten_name(ld.get("description")) or meta_content(
        tree, name="description"
    )
    image = flatten_image(ld.get("image")) or meta_content(tree, prop="og:image")

    return Article(
        url=page.url,
        slug=page.slug,
        title=title,
        description=description,
        author=flatten_name(ld.get("author")),
        date_published=flatten_name(ld.get("datePublished")),
        date_modified=flatten_name(ld.get("dateModified")),
        image=image,
        lastmod=page.lastmod,
        word_count=sum(len(block["text"].split()) for block in blocks),
        blocks=blocks,
        scraped_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_article(article: Article, number: int, articles_dir: Path) -> str:
    """Write one article as `{"metadata": {...}, "content": [blocks]}`.

    Everything that describes the article lives under "metadata"; "content"
    is just the ordered list of typed body blocks.
    """
    filename = f"{number:03d}_{article.slug.strip('/').replace('/', '__')}.json"
    doc = {
        "metadata": {
            "article_number": number,
            "title": article.title or "",
            "author": article.author or "",
            "description": article.description or "",
            "url": article.url,
            "slug": article.slug,
            "date_published": article.date_published or "",
            "date_modified": article.date_modified or "",
            "image": article.image or "",
            "lastmod": article.lastmod or "",
            "word_count": article.word_count,
            "scraped_at": article.scraped_at,
        },
        "content": article.blocks,
    }
    (articles_dir / filename).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return filename


def csv_row(article: Article, number: int, text_file: str) -> dict:
    return {
        "article_number": number,
        "url": article.url,
        "slug": article.slug,
        "author": article.author or "",
        "date_published": article.date_published or "",
        "date_modified": article.date_modified or "",
        "image": article.image or "",
        "lastmod": article.lastmod or "",
        "word_count": article.word_count,
        "text_file": text_file,
    }


def main() -> int:
    ARTICLES_DIR = config.data.articles_dir
    ARTICLES_CSV = config.data.csv_path
    PAGES_CSV = config.data.pages_csv

    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLES_CSV.parent.mkdir(parents=True, exist_ok=True)

    pages = load_pages(PAGES_CSV, config.data.raw_html_dir)
    print(f"Loaded {len(pages)} pages from {PAGES_CSV}", file=sys.stderr)

    failures: list[tuple[str, str]] = []
    written = 0

    with ARTICLES_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ARTICLES_CSV_COLUMNS)
        writer.writeheader()

        for i, page in enumerate(pages, start=1):
            try:
                article = parse_article(page)
            except Exception as exc:  # keep going; report at the end
                failures.append((page.url, str(exc)))
                print(f"[{i}/{len(pages)}] FAILED {page.slug}: {exc}", file=sys.stderr)
                continue

            text_file = write_article(
                article, page.article_number, ARTICLES_DIR
            )
            writer.writerow(csv_row(article, page.article_number, text_file))
            written += 1
            print(
                f"[{i}/{len(pages)}] {text_file} ({article.word_count} words)",
                file=sys.stderr,
            )

    print(
        f"\nWrote {written} articles to {ARTICLES_DIR}/ and {ARTICLES_CSV}",
        file=sys.stderr,
    )
    if failures:
        print(f"{len(failures)} failed:", file=sys.stderr)
        for url, error in failures:
            print(f"  {url}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
