from rag.ingestion.processing import flatten_image, flatten_name


def test_flatten_name_handles_the_json_ld_shapes():
    assert flatten_name("Ada") == "Ada"
    assert flatten_name({"name": "Ada"}) == "Ada"
    assert flatten_name([{"name": "Ada"}, "Grace"]) == "Ada, Grace"
    assert flatten_name({"name": "  "}) is None


def test_flatten_image_takes_the_first_usable_url():
    assert flatten_image([{"url": ""}, {"url": "https://x/y.png"}]) == "https://x/y.png"
