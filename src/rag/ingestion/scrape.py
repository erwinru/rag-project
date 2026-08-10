"""One-off dump of every ML6 blog post: metadata + full body text.

Discovery goes through https://www.ml6.eu/sitemap.xml rather than the paginated
listing, which hands us every post URL in a single request.

Each post page is server-rendered (HubSpot CMS), so plain HTTP is enough -- no
browser automation. Metadata comes from the JSON-LD `Article` block the CMS
embeds; the body is extracted with trafilatura, which strips the surrounding
nav/CTA/footer boilerplate.

Everything lands under data/:
    data/articles/NNN_<slug>.txt   title, author, description, then the body
    data/articles.csv              one row per article with the remaining metadata
    data/cache/                    raw HTML, so re-runs cost no extra requests

The leading number in each filename is the article number in the CSV, so the
two line up.

Usage:
    uv run rag-scrape                 # scrape everything
    uv run rag-scrape --limit 5       # smoke test on the first 5 posts
    uv run rag-scrape --no-cache      # ignore cached HTML and refetch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import trafilatura
from lxml import etree
from selectolax.parser import HTMLParser

from rag.config import ARTICLES_DIR, CACHE_DIR, CSV_PATH

SITEMAP_URL = "https://www.ml6.eu/sitemap.xml"
BLOG_PREFIX = "https://www.ml6.eu/en/blog/"
USER_AGENT = "ml6-blog-scraper/0.1 (one-off research dump; contact: erwin.rudi@enaovision.com)"

CSV_COLUMNS = [
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

REQUEST_DELAY = 0.5  # seconds between live requests
TIMEOUT = 30.0
MAX_RETRIES = 3

# Listing/feed URLs that live under the blog prefix but are not posts. The
# sitemap is clean today; this guards against it growing new shapes.
NON_POST_SEGMENTS = ("page/", "author/", "tag/", "topic/", "rss")


@dataclass
class Post:
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
    body: str | None
    scraped_at: str


def fetch(client: httpx.Client, url: str) -> str:
    """GET with retries and a polite delay. Raises on persistent failure."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.text
        except (httpx.HTTPError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_DELAY * 2**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def cache_name(slug: str) -> str:
    """Flatten a slug into a single safe filename.

    A handful of slugs are multi-segment -- HubSpot builds them from the post
    title, so "...(Joint committee vote 11/5/23)" yields a slug containing real
    path separators. Left as-is those nest into subdirectories, and a site with
    both `/foo` and `/foo/bar` would collide (file vs directory, same path).
    """
    return slug.strip("/").replace("/", "__")


def fetch_cached(client: httpx.Client, url: str, slug: str, use_cache: bool) -> str:
    """Fetch a page, backed by an on-disk HTML cache.

    The cache means re-running while iterating on the parsing logic costs no
    extra requests against ml6.eu.
    """
    cache_path = CACHE_DIR / f"{cache_name(slug)}.html"
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    html = fetch(client, url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    time.sleep(REQUEST_DELAY)
    return html


def discover_posts(client: httpx.Client) -> list[tuple[str, str | None]]:
    """Return (url, lastmod) for every blog post listed in the sitemap."""
    xml = fetch(client, SITEMAP_URL)
    root = etree.fromstring(xml.encode("utf-8"))
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    posts: list[tuple[str, str | None]] = []
    for entry in root.findall("sm:url", ns):
        loc_el = entry.find("sm:loc", ns)
        if loc_el is None or not loc_el.text:
            continue
        url = loc_el.text.strip()
        if not url.startswith(BLOG_PREFIX):
            continue
        tail = url[len(BLOG_PREFIX) :]
        if not tail or tail.startswith(NON_POST_SEGMENTS):
            continue

        lastmod_el = entry.find("sm:lastmod", ns)
        lastmod = lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None
        posts.append((url, lastmod))

    # The sitemap is not sorted; dedupe and give the dump a stable order.
    return sorted(dict(posts).items())


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


def meta_content(tree: HTMLParser, *, prop: str | None = None, name: str | None = None) -> str | None:
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


def parse_post(url: str, lastmod: str | None, html: str) -> Post:
    ld = extract_article_ld(html)
    tree = HTMLParser(html)

    # Markdown keeps headings and lists intact, which makes the dump far more
    # useful downstream than a flat wall of text.
    body = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_images=False,
    )

    title = flatten_name(ld.get("headline")) or meta_content(tree, prop="og:title")
    if not title:
        heading = tree.css_first("h1")
        title = heading.text(strip=True) if heading else None

    description = flatten_name(ld.get("description")) or meta_content(tree, name="description")
    image = flatten_image(ld.get("image")) or meta_content(tree, prop="og:image")

    return Post(
        url=url,
        slug=url[len(BLOG_PREFIX) :].strip("/"),
        title=title,
        description=description,
        author=flatten_name(ld.get("author")),
        date_published=flatten_name(ld.get("datePublished")),
        date_modified=flatten_name(ld.get("dateModified")),
        image=image,
        lastmod=lastmod,
        word_count=len(body.split()) if body else 0,
        body=body,
        scraped_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def write_article(post: Post, number: int, articles_dir: Path) -> str:
    """Write one article as plain text: title, author, description, body.

    Fields are labelled so the header stays unambiguous when a title or
    description happens to run over several lines.
    """
    filename = f"{number:03d}_{cache_name(post.slug)}.txt"
    lines = [
        f"Title: {post.title or ''}",
        f"Author: {post.author or ''}",
        f"Description: {post.description or ''}",
        "",
        post.body or "",
    ]
    (articles_dir / filename).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return filename


def csv_row(post: Post, number: int, text_file: str) -> dict:
    return {
        "article_number": number,
        "url": post.url,
        "slug": post.slug,
        "author": post.author or "",
        "date_published": post.date_published or "",
        "date_modified": post.date_modified or "",
        "image": post.image or "",
        "lastmod": post.lastmod or "",
        "word_count": post.word_count,
        "text_file": text_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only scrape the first N posts")
    parser.add_argument("--no-cache", action="store_true", help="refetch instead of reading cache/")
    parser.add_argument(
        "--articles-dir", type=Path, default=ARTICLES_DIR, help="directory for the .txt articles"
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="metadata CSV output path")
    args = parser.parse_args()

    args.articles_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    with httpx.Client(
        headers=headers, timeout=TIMEOUT, follow_redirects=True, http2=False
    ) as client:
        print(f"Reading {SITEMAP_URL} ...", file=sys.stderr)
        posts = discover_posts(client)
        if args.limit:
            posts = posts[: args.limit]
        print(f"Found {len(posts)} blog posts.", file=sys.stderr)

        failures: list[tuple[str, str]] = []
        empty_bodies: list[str] = []
        written = 0

        with args.csv.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            # Numbering follows sitemap position, so a failed post leaves a gap
            # rather than shifting every article number after it.
            for index, (url, lastmod) in enumerate(posts, start=1):
                slug = url[len(BLOG_PREFIX) :].strip("/")
                try:
                    html = fetch_cached(client, url, slug, use_cache=not args.no_cache)
                    post = parse_post(url, lastmod, html)
                except Exception as exc:  # keep going; report at the end
                    failures.append((url, str(exc)))
                    print(f"[{index}/{len(posts)}] FAILED {slug}: {exc}", file=sys.stderr)
                    continue

                if not post.body:
                    empty_bodies.append(slug)

                text_file = write_article(post, index, args.articles_dir)
                writer.writerow(csv_row(post, index, text_file))
                written += 1
                print(
                    f"[{index}/{len(posts)}] {text_file} ({post.word_count} words)",
                    file=sys.stderr,
                )

    print(f"\nWrote {written} articles to {args.articles_dir}/ and {args.csv}", file=sys.stderr)
    if empty_bodies:
        print(f"No body text extracted for {len(empty_bodies)}: {', '.join(empty_bodies)}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} failed:", file=sys.stderr)
        for url, error in failures:
            print(f"  {url}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
