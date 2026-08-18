import { useEffect, useRef, useState, type Ref } from "react"
import { AlertCircle, ArrowUp, Search, SearchX } from "lucide-react"

import { ask, type AskResponse } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { ChunkCard } from "@/components/ChunkCard"

/** One question and whatever came back for it. */
interface Turn {
  id: string
  question: string
  status: "pending" | "done" | "error"
  response?: AskResponse
  error?: string
}

/** `undefined` means "don't send top_k", i.e. let the API use its own default. */
const TOP_K_OPTIONS: { label: string; value: number | undefined }[] = [
  { label: "default", value: undefined },
  { label: "5", value: 5 },
  { label: "10", value: 10 },
]

const EXAMPLES = [
  "What is MLOps?",
  "How does 3D computer vision work?",
  "How do you evaluate a RAG pipeline?",
]

export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState("")
  const [topK, setTopK] = useState<number | undefined>(undefined)
  const [pending, setPending] = useState(false)

  // One controller for the in-flight request, so unmounting (or a hot reload)
  // doesn't leave a fetch writing into state that's gone.
  const abortRef = useRef<AbortController | null>(null)
  const newestTurnRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  // Pin the newest question to the top of the thread rather than scrolling to
  // the very bottom -- the top chunk is often tall enough that the end of it
  // would push the question itself off-screen. Runs twice per turn: once when
  // the question appears (when there may not yet be enough content below it to
  // scroll all the way) and again when its chunks land and there is. Older
  // turns resolving can't trigger it, so nothing yanks the view out from under
  // someone reading back through the thread. Instant, not smooth: there's
  // nothing to follow between here and a question that was just added.
  const newestStatus = turns.at(-1)?.status
  useEffect(() => {
    newestTurnRef.current?.scrollIntoView({ block: "start" })
  }, [turns.length, newestStatus])

  async function submit(question: string) {
    const trimmed = question.trim()
    if (trimmed === "" || pending) return

    const id = crypto.randomUUID()
    setTurns((current) => [...current, { id, question: trimmed, status: "pending" }])
    setDraft("")
    setPending(true)

    const controller = new AbortController()
    abortRef.current = controller

    const update = (patch: Partial<Turn>) =>
      setTurns((current) =>
        current.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)),
      )

    try {
      const response = await ask(trimmed, { topK, signal: controller.signal })
      update({ status: "done", response })
    } catch (error) {
      // An abort is this component tearing down, not a failure to report.
      if (error instanceof DOMException && error.name === "AbortError") return
      update({
        status: "error",
        error: error instanceof Error ? error.message : String(error),
      })
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setPending(false)
      }
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-8">
          {turns.length === 0 ? (
            <EmptyState onPick={(question) => void submit(question)} />
          ) : (
            turns.map((turn, index) => (
              <TurnView
                key={turn.id}
                turn={turn}
                ref={index === turns.length - 1 ? newestTurnRef : null}
              />
            ))
          )}
        </div>
      </div>

      <div className="bg-background/80 border-t backdrop-blur">
        <form
          className="mx-auto w-full max-w-3xl px-4 py-4"
          onSubmit={(event) => {
            event.preventDefault()
            void submit(draft)
          }}
        >
          <div className="flex items-center gap-2">
            <Input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                // Submit on Enter explicitly rather than relying on the
                // browser's implicit form submission, which doesn't fire for
                // every synthetic key event. preventDefault keeps the two
                // paths from both submitting where implicit *does* fire.
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault()
                  void submit(draft)
                }
              }}
              placeholder="Ask something about the ML6 blog..."
              autoFocus
              aria-label="Question"
              className="h-11"
            />
            <Button
              type="submit"
              size="icon"
              className="size-11 shrink-0"
              disabled={pending || draft.trim() === ""}
              aria-label="Ask"
            >
              <ArrowUp className="size-5" />
            </Button>
          </div>
          <div className="text-muted-foreground mt-2 flex items-center gap-2 text-xs">
            <span>chunks</span>
            <div className="flex items-center gap-1">
              {TOP_K_OPTIONS.map((option) => (
                <Button
                  key={option.label}
                  type="button"
                  variant={option.value === topK ? "secondary" : "ghost"}
                  size="sm"
                  className="h-6 px-2 text-xs"
                  onClick={() => setTopK(option.value)}
                >
                  {option.label}
                </Button>
              ))}
            </div>
            <span className="ml-auto hidden sm:inline">
              Retrieval only &mdash; no generated answer.
            </span>
          </div>
        </form>
      </div>
    </div>
  )
}

function TurnView({ turn, ref }: { turn: Turn; ref: Ref<HTMLDivElement> }) {
  const results = turn.response?.results ?? []
  const [top, ...rest] = results

  return (
    <div ref={ref} className="flex scroll-mt-8 flex-col gap-3">
      <div className="flex justify-end">
        <p className="bg-primary text-primary-foreground max-w-[85%] rounded-2xl rounded-br-sm px-4 py-2 text-sm">
          {turn.question}
        </p>
      </div>

      {turn.status === "pending" && <PendingChunk />}

      {turn.status === "error" && (
        <div className="border-destructive/40 bg-destructive/5 text-destructive flex items-start gap-2 rounded-xl border p-4 text-sm">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <span>{turn.error}</span>
        </div>
      )}

      {turn.status === "done" && top === undefined && (
        <div className="text-muted-foreground flex items-center gap-2 rounded-xl border border-dashed p-4 text-sm">
          <SearchX className="size-4 shrink-0" />
          <span>
            No chunks came back. Either nothing matched or the index is empty &mdash;
            check the collection count in the header.
          </span>
        </div>
      )}

      {top !== undefined && (
        <div className="flex flex-col gap-2">
          <ChunkCard chunk={top} top />
          {rest.length > 0 && (
            <>
              <p className="text-muted-foreground px-1 pt-2 text-xs">
                {rest.length} more {rest.length === 1 ? "chunk" : "chunks"} retrieved
              </p>
              {rest.map((chunk) => (
                <ChunkCard key={chunk.id} chunk={chunk} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}

function PendingChunk() {
  return (
    <div className="flex flex-col gap-3 rounded-xl border p-4">
      <Skeleton className="h-4 w-2/5" />
      <Skeleton className="h-3 w-1/4" />
      <div className="flex flex-col gap-2 pt-1">
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-4/5" />
      </div>
    </div>
  )
}

function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  return (
    <div className="flex flex-col items-center gap-6 py-16 text-center">
      <div className="bg-muted text-muted-foreground flex size-12 items-center justify-center rounded-full">
        <Search className="size-5" />
      </div>
      <div className="flex flex-col gap-1.5">
        <h2 className="text-lg font-semibold">Ask the ML6 blog</h2>
        <p className="text-muted-foreground max-w-md text-sm">
          Your question is embedded and matched against the indexed chunks. What comes
          back is the source material itself, most similar first &mdash; not an answer
          written by a model.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((example) => (
          <Button
            key={example}
            variant="outline"
            size="sm"
            className="text-xs"
            onClick={() => onPick(example)}
          >
            {example}
          </Button>
        ))}
      </div>
    </div>
  )
}
