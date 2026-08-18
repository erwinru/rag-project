# Chat frontend

React + Tailwind + shadcn/ui front end for the retrieval API
([`docs/API.md`](../docs/API.md)). You type a question, it comes back with the
chunks that best match it -- the top one expanded, the runners-up collapsed.

There is no generated answer, because the backend doesn't generate one: the
service is retrieval only.

## Run it

The API first (it must be running -- the frontend has nothing else to talk to):

```bash
uv run rag-api
```

Then, in `frontend/`:

```bash
npm install
```

```bash
npm run dev
```

http://localhost:5173. `npm run build` writes a static bundle to `dist/`.

## How it reaches the API

`POST /search`, via `ask()` in [`src/lib/api.ts`](src/lib/api.ts) -- **not**
`/ask`. `ask` is the name of the CLI (`rag-ask`); over HTTP the endpoint is
`/search`. The path lives in one constant (`ASK_PATH`) so pointing this at a
real `/ask` endpoint later is a one-line change.

Calls go to `/api/search` and the dev server proxies `/api` to
`http://127.0.0.1:8000` ([`vite.config.ts`](vite.config.ts)). That keeps them
same-origin, which matters: the API has no CORS middleware, so a direct
cross-origin `fetch` from `localhost:5173` would be blocked by the browser.

Serving the build from an origin that isn't in front of the API needs both
`VITE_API_BASE_URL` (see [`.env.example`](.env.example)) and CORS added to the
API.

## Notes

- **`distance` is shown raw.** Chroma's distance -- lower is more similar. It
  isn't a 0-1 similarity, so it's never rendered as a percentage or a bar.
- **The chunk count in the header comes from `/health`.** An empty collection
  answers `/search` with `200` and no results, indistinguishable from a
  question that matched nothing; the badge turns red at zero chunks.
- **`top_k` counts chunks, not articles**, so two results can come from the
  same article. The "chunks" toggle above the input leaves `top_k` unset by
  default, letting `config.retrieval.top_k` decide rather than hardcoding a
  second default here.
- **shadcn components are vendored** in `src/components/ui/` in the usual way,
  with `components.json` present, so `npx shadcn@latest add <component>` works
  for anything else you need.
- **The header badge is a snapshot from page load.** If the API goes down (or
  the index is rebuilt) afterwards, the badge keeps showing what it saw; an
  actual question will report the failure. Reload to refresh it.
- **Nothing is persisted.** Reloading clears the thread -- there's no history,
  no local storage, no share links.

`npm run typecheck` runs `tsc` without emitting, if you want the check without
the build.
