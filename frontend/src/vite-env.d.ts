/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL for the retrieval API. Empty in dev -- see src/lib/api.ts. */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
