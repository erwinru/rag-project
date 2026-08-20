from rag.ingestion.scrape import html_filename


def test_html_filename_flattens_multi_segment_slugs():
    assert html_filename("/foo/bar/") == "foo__bar.html"
