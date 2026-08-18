import { useEffect, useState } from "react"
import { Database } from "lucide-react"

import { health, type HealthResponse } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Chat } from "@/components/Chat"

export default function App() {
  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b">
        <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex shrink-0 flex-col">
            <h1 className="text-sm font-semibold whitespace-nowrap">ML6 blog retrieval</h1>
            <p className="text-muted-foreground hidden text-xs sm:block">
              question in, most similar chunks out
            </p>
          </div>
          <IndexStatus />
        </div>
      </header>
      <Chat />
    </div>
  )
}

/**
 * `/health` in a badge: collection, chunk count, embedding provider. Worth the
 * extra call on load -- an empty index answers questions with `200` and zero
 * results, which otherwise looks exactly like a question that matched nothing.
 */
function IndexStatus() {
  const [status, setStatus] = useState<HealthResponse | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    health(controller.signal)
      .then(setStatus)
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return
        setFailed(true)
      })
    return () => controller.abort()
  }, [])

  if (failed) {
    return (
      <Badge variant="destructive" className="gap-1.5">
        <Database />
        API unreachable
      </Badge>
    )
  }

  if (status === null) return null

  return (
    <Badge
      variant={status.chunks === 0 ? "destructive" : "secondary"}
      className="min-w-0 shrink gap-1.5 font-mono"
      title={`embedding provider: ${status.embedding_provider}`}
    >
      <Database className="shrink-0" />
      <span className="truncate">{status.collection}</span>
      <span className="shrink-0">
        &middot; {status.chunks.toLocaleString()} chunks
      </span>
    </Badge>
  )
}
