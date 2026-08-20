"""Discover every ML6 blog post and cache its raw HTML.

Discovery goes through https://www.ml6.eu/sitemap.xml rather than the paginated
listing, which hands us every post URL in a single request.

Each post page is server-rendered (HubSpot CMS), so plain HTTP is enough -- no
browser automation. This step does no parsing beyond the sitemap itself: it
only fetches and caches pages, then records what it found so the processing
step (rag.ingestion.processing) can pick it up from disk.

Everything lands under data/:
    data/raw_html/<slug>.html   raw page HTML, so re-runs cost no extra requests
    data/pages.csv              url/slug/lastmod + which raw_html file, one row per page

The leading number in each pages.csv row is the article number that carries
through to the processed articles, so a failed fetch leaves a gap rather than
shifting every article number after it.

Usage:
    uv run rag-scrape
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import httpx
from lxml import etree

from rag.config import config

PAGES_CSV_COLUMNS = ["article_number", "url", "slug", "lastmod", "html_file"]

REQUEST_DELAY = 0.5  # seconds between live requests
TIMEOUT = 30.0
MAX_RETRIES = 3

# Listing/feed URLs that live under the blog prefix but are not posts. The
# sitemap is clean today; this guards against it growing new shapes.
NON_POST_SEGMENTS = ("page/", "author/", "tag/", "topic/", "rss")


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


def html_filename(slug: str) -> str:
    """Flatten a slug into a single safe filename for the raw HTML cache.

    A handful of slugs are multi-segment -- HubSpot builds them from the post
    title, so "...(Joint committee vote 11/5/23)" yields a slug containing real
    path separators. Left as-is those nest into subdirectories, and a site with
    both `/foo` and `/foo/bar` would collide (file vs directory, same path).
    """
    return slug.strip("/").replace("/", "__") + ".html"


def fetch_and_cache_html(
    client: httpx.Client, url: str, slug: str, use_cache: bool
) -> Path:
    """Fetch a page's HTML into data/raw_html/, reusing a cached copy if present.

    The cache means re-running while iterating on the processing step costs no
    extra requests against ml6.eu.
    """
    path = config.data.raw_html_dir / html_filename(slug)
    if use_cache and path.exists():
        return path

    html = fetch(client, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    time.sleep(REQUEST_DELAY)
    return path


def discover_posts(client: httpx.Client) -> list[tuple[str, str | None]]:
    """Return (url, lastmod) for every blog post listed in the sitemap."""
    xml = fetch(client, config.scrape.sitemap_url)
    root = etree.fromstring(xml.encode("utf-8"))
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    posts: list[tuple[str, str | None]] = []
    for entry in root.findall("sm:url", ns):
        loc_el = entry.find("sm:loc", ns)
        if loc_el is None or not loc_el.text:
            continue
        url = loc_el.text.strip()
        if not url.startswith(config.scrape.blog_prefix):
            continue
        tail = url[len(config.scrape.blog_prefix) :]
        if not tail or tail.startswith(NON_POST_SEGMENTS):
            continue

        lastmod_el = entry.find("sm:lastmod", ns)
        lastmod = (
            lastmod_el.text.strip()
            if lastmod_el is not None and lastmod_el.text
            else None
        )
        posts.append((url, lastmod))

    # The sitemap is not sorted; dedupe and give the dump a stable order.
    return sorted(dict(posts).items())


def main() -> int:
    RAW_HTML_DIR = config.data.raw_html_dir
    PAGES_CSV = config.data.pages_csv

    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_CSV.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": config.scrape.user_agent, "Accept-Language": "en"}
    with httpx.Client(
        headers=headers, timeout=TIMEOUT, follow_redirects=True, http2=False
    ) as client:
        print(f"Reading {config.scrape.sitemap_url} ...", file=sys.stderr)
        posts = discover_posts(client)
        print(f"Found {len(posts)} blog posts.", file=sys.stderr)

        failures: list[tuple[str, str]] = []
        written = 0

        with PAGES_CSV.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PAGES_CSV_COLUMNS)
            writer.writeheader()

            # Numbering follows sitemap position, so a failed fetch leaves a gap
            # rather than shifting every article number after it.
            for index, (url, lastmod) in enumerate(posts, start=1):
                slug = url[len(config.scrape.blog_prefix) :].strip("/")
                try:
                    path = fetch_and_cache_html(client, url, slug, use_cache=True)
                except Exception as exc:  # keep going; report at the end
                    failures.append((url, str(exc)))
                    print(
                        f"[{index}/{len(posts)}] FAILED {slug}: {exc}", file=sys.stderr
                    )
                    continue

                writer.writerow(
                    {
                        "article_number": index,
                        "url": url,
                        "slug": slug,
                        "lastmod": lastmod or "",
                        "html_file": path.name,
                    }
                )
                written += 1
                print(f"[{index}/{len(posts)}] {path.name}", file=sys.stderr)

    print(
        f"\nCached {written} pages to {RAW_HTML_DIR}/ and {PAGES_CSV}",
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
