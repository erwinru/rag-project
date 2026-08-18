import { useState } from "react"
import { ChevronDown, FileText, User } from "lucide-react"

import type { Chunk } from "@/lib/api"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

/** `{slug}::chunk-NN` -> `chunk-NN`, the part that isn't already in the title. */
function chunkLabel(id: string): string {
  const [, chunk] = id.split("::")
  return chunk ?? id
}

/**
 * A retrieved chunk. `top` gets the full text and a highlighted border; the
 * runners-up collapse to their heading and expand on click, so a top_k of 10
 * doesn't bury the answer.
 *
 * `distance` is Chroma's raw distance -- lower is more similar, not a 0-1
 * similarity -- so it's shown as-is and labelled, never as a percentage.
 */
export function ChunkCard({ chunk, top = false }: { chunk: Chunk; top?: boolean }) {
  const [expanded, setExpanded] = useState(top)

  return (
    <Card
      className={cn(
        "gap-0 overflow-hidden py-0",
        top && "border-primary/30 ring-primary/10 ring-1",
      )}
    >
      <CardHeader className="gap-2 px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <CardTitle className="text-sm leading-snug">{chunk.title || chunk.slug}</CardTitle>
          {top ? (
            <Badge className="shrink-0">top chunk</Badge>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              className="-my-1 shrink-0 text-xs"
              onClick={() => setExpanded((value) => !value)}
              aria-expanded={expanded}
            >
              {expanded ? "Hide" : "Show"}
              <ChevronDown
                className={cn("transition-transform", expanded && "rotate-180")}
              />
            </Button>
          )}
        </div>
        <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
          {chunk.author && (
            <span className="inline-flex items-center gap-1">
              <User className="size-3" />
              {chunk.author}
            </span>
          )}
          <span className="inline-flex items-center gap-1 font-mono">
            <FileText className="size-3" />
            {chunkLabel(chunk.id)}
          </span>
          <span title="Chroma distance -- lower is more similar, not a similarity score">
            distance {chunk.distance.toFixed(3)}
          </span>
        </div>
      </CardHeader>
      {expanded && (
        <CardContent className="px-4 pt-1 pb-4">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{chunk.text}</p>
        </CardContent>
      )}
    </Card>
  )
}
