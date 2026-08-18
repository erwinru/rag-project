/**
 * Client for the retrieval API (`src/rag/api/app.py`, docs/API.md).
 *
 * The endpoint is `POST /search`, not `/ask`: `ask` exists only as the CLI
 * `rag-ask`, and the service deliberately has no generation step -- a question
 * goes in, the top-k most similar *chunks* come back. Change ASK_PATH if a
 * real `/ask` (retrieve + generate) endpoint ever lands; nothing else here
 * assumes the path.
 */

const ASK_PATH = "/search"

/**
 * Prefix for API calls. Empty by default, which makes requests same-origin
 * (`/api/search`) so the dev proxy in vite.config.ts can forward them and the
 * API's missing CORS middleware never comes up. Set VITE_API_BASE_URL only
 * when the frontend is served from a different origin than the API.
 */
const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "")

const API_PREFIX = BASE_URL === "" ? "/api" : BASE_URL

/** One retrieved chunk. Mirrors `SearchResult` in src/rag/api/app.py. */
export interface Chunk {
  /** `{slug}::chunk-NN` */
  id: string
  slug: string
  title: string
  author: string
  /**
   * Chroma's distance: lower is more similar. Not a similarity score and not
   * normalised to 0-1 -- never render it as a percentage.
   */
  distance: number
  text: string
}

/** Mirrors `SearchResponse` in src/rag/api/app.py. */
export interface AskResponse {
  question: string
  top_k: number
  results: Chunk[]
}

export interface HealthResponse {
  status: string
  collection: string
  chunks: number
  embedding_provider: string
}

/** An API call that came back with a non-2xx status. */
export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

/**
 * FastAPI reports errors as `{detail: ...}` -- a string for HTTPException, a
 * list of per-field objects for a 422 validation error. Flatten either into one
 * line; return null if the body isn't that shape, meaning the response didn't
 * come from the application at all.
 */
async function fastApiDetail(response: Response): Promise<string | null> {
  let detail: unknown
  try {
    detail = ((await response.json()) as { detail?: unknown }).detail
  } catch {
    return null
  }

  if (typeof detail === "string") return detail
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((item) =>
        typeof item === "object" && item !== null && "msg" in item
          ? String((item as { msg: unknown }).msg)
          : JSON.stringify(item),
      )
      .join("; ")
  }
  return null
}

async function errorMessage(response: Response): Promise<string> {
  const detail = await fastApiDetail(response)
  if (detail !== null) return detail

  // No FastAPI `detail` in the body, so this isn't the application answering.
  // The dev proxy returns a bodyless 500 when it can't reach the API at all,
  // which is much the most common way of getting here -- say so, rather than
  // passing on a status line that points at the wrong component.
  const status = `${response.status} ${response.statusText}`
  return response.status >= 500
    ? `${status} -- no response from the retrieval API. Is it running? \`uv run rag-api\``
    : status
}

/**
 * Ask a question and get back the chunks that best match it, most similar
 * first. `signal` lets the caller abort an in-flight request.
 *
 * Throws ApiError when the API answers with a non-2xx (422 for a bad `top_k`,
 * 502 when the embedding call upstream fails), and a plain Error when the
 * request never got an answer at all -- API not running, proxy target down.
 */
export async function ask(
  question: string,
  options: { topK?: number; signal?: AbortSignal } = {},
): Promise<AskResponse> {
  const body: { question: string; top_k?: number } = { question }
  // Omit top_k rather than guessing a default: unset means the API uses
  // config.retrieval.top_k, which is the one source of truth for it.
  if (options.topK !== undefined) body.top_k = options.topK

  let response: Response
  try {
    response = await fetch(`${API_PREFIX}${ASK_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: options.signal,
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause
    throw new Error(
      "Could not reach the retrieval API. Is it running? `uv run rag-api`",
      { cause },
    )
  }

  if (!response.ok) throw new ApiError(response.status, await errorMessage(response))

  return (await response.json()) as AskResponse
}

/**
 * Collection name, chunk count and embedding provider. Used to show whether
 * the API is up *and* pointed at a non-empty index -- an empty collection
 * answers `/search` with `200` and no results, which is otherwise
 * indistinguishable from a question that matched nothing.
 */
export async function health(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_PREFIX}/health`, { signal })
  if (!response.ok) throw new ApiError(response.status, await errorMessage(response))
  return (await response.json()) as HealthResponse
}
