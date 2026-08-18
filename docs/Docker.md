# Docker

Container image for the retrieval API ([`API.md`](API.md)).
Files: [`Dockerfile`](../Dockerfile), [`compose.yaml`](../compose.yaml),
[`.dockerignore`](../.dockerignore).

## Overview

The image bundles the app and its dependencies (including the local
embedding model); the **Chroma index is mounted at runtime**, not baked in.
Build it, then run it with `data/index` mounted:

```bash
docker compose up --build
```

or, without compose:

```bash
docker build -t rag-api .
docker run --rm -p 8000:8000 -v "$(pwd)/data/index:/app/data/index" rag-api
```

Then the endpoint is exactly the same as the local one:

```bash
curl "http://127.0.0.1:8000/search?question=What%20is%20MLOps%3F&top_k=5"
```

The lockfile must be current before building -- the image installs with
`uv sync --locked`, which refuses to resolve around a stale `uv.lock`. If a
dependency was just added to `pyproject.toml`, run `uv sync` (or `uv lock`)
locally first and rebuild.

## Decisions

- **Two stages, both on `python:3.13-slim-bookworm`, with `uv` copied in as
  a binary.** The obvious alternative -- building on uv's own
  `ghcr.io/astral-sh/uv:python3.13` image -- puts uv's *managed* interpreter
  in the venv, and a venv hardcodes the interpreter path it was created
  from. Copying that venv into a stage that doesn't have that interpreter
  gives a container that fails at startup. Same base on both sides avoids
  the whole class of problem.
- **Dependencies install in their own layer, before `src/` is copied.**
  `uv sync --no-install-project` against just `pyproject.toml`/`uv.lock`
  means the slow layer (torch and friends) is rebuilt only when
  dependencies actually change, not on every code edit.
- **The Chroma index is mounted, not copied in.** `data/` is gitignored and
  rebuilt by `uv run rag-embed`; baking an 18MB snapshot into the image
  would mean rebuilding it to re-index, and would put a build's worth of
  staleness between the image and the data. Mount it **read-write**: Chroma
  writes sqlite journal files even for a pure read query, so a `:ro` mount
  fails at query time, not at startup.
- **The embedding model is baked in at build time.** `rag.embedding.
  providers.huggingface` constructs its `SentenceTransformer` at *import*
  time, so without a pre-warmed cache the ~87MB download happens while
  uvicorn is starting -- the container would sit there not listening, and
  every restart would re-download it. A build-time `SentenceTransformer(...)`
  into `HF_HOME=/opt/huggingface` makes startup local and offline.
- **`CMD` runs uvicorn directly rather than the `rag-api` entry point.**
  `rag-api` binds `config.api.host`, which is `127.0.0.1` -- correct for a
  laptop, unreachable from outside a container. The container overrides the
  host to `0.0.0.0` deliberately; the port stays 8000 and is remapped from
  the outside (`-p 9000:8000`), not by editing config.
- **Non-root (`uid 10001`) with `/app/data` chowned to it.** On macOS,
  Docker Desktop maps bind-mount ownership to the container user, so the
  mounted index just works. On Linux the host `data/index` is owned by the
  host uid and the container user can't write sqlite's journal -- run with
  `--user "$(id -u):$(id -g)"` there.
- **`.dockerignore` excludes `data/` (~75MB), `.venv`, and `.git`.** The
  host `.venv` in particular holds macOS wheels that would be useless and
  slow to ship into a Linux build context.
- **AWS credentials are not in the image.** They're only needed if
  `config.embedding.provider` is switched to `bedrock`; `compose.yaml` has
  the env passthrough commented out for that case.

## Checklist / open edge cases

- [ ] **Image size.** The image installs the project's whole dependency set,
      including `ragas`, `langchain-*`, `trafilatura` and the scraper's
      parsers, none of which `/search` touches. Splitting into
      `[project.optional-dependencies]` (an `api` extra) is the real fix;
      not done yet because it changes how every other command installs too.
- [ ] **`torch` is CPU-only here by accident, not by declaration.** Built on
      Apple Silicon the image is `linux/arm64`, and PyTorch's aarch64 wheels
      carry no CUDA. Building for `linux/amd64` (e.g. to deploy on an x86
      host) instead pulls the CUDA build and several GB of unused NVIDIA
      libraries. Pinning a CPU index in `pyproject.toml` would make it
      explicit:

      ```toml
      [[tool.uv.index]]
      name = "pytorch-cpu"
      url = "https://download.pytorch.org/whl/cpu"
      explicit = true

      [tool.uv.sources]
      torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
      ```

      This needs a re-lock (`uv lock`) and hasn't been applied.
- [ ] Single uvicorn worker, no `--workers` and no process manager -- the
      embedding model is loaded once per process, so worker count is a real
      memory tradeoff, untested
- [ ] No image tagging/versioning scheme, no registry push, no CI build --
      `rag-api:latest` built locally is all there is
- [ ] `HEALTHCHECK` reports the process is serving, but `/health` returns
      `200` with `"chunks": 0` for an unmounted or empty index -- healthy and
      useless are indistinguishable to Docker
- [ ] No graceful handling of a missing `/app/data/index` mount: Chroma
      creates an empty collection instead of failing, so a forgotten `-v`
      looks like "the index has no results"
