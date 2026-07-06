# LingYa Web UI

Browser-based chat interface for LingYa — the primary (and only) user interaction entry point since v0.9.4.

## Tech Stack

- **Vite 6** — build tool and dev server
- **React 19** — UI framework
- **TypeScript** — type safety
- **Tailwind CSS 4** — utility-first styling (dark-only design)
- **TanStack Query 5** — server state management and caching
- **React Router 7** — SPA client-side routing

## Development

```bash
cd web
npm install
npm run dev        # Dev server at http://localhost:5173, proxies /api → localhost:8765
```

The dev server proxies API requests to the Python daemon — make sure `python main.py` is running on port 8765.

## Production Build

```bash
npm run build      # Outputs to web/dist/
```

The Python daemon serves `web/dist/` via FastAPI `StaticFiles` at `/` (with SPA fallback: `html=True`). If `web/dist/` does not exist at startup, the daemon prints a note and runs without Web UI.

## Testing

```bash
npm test           # Vitest + @testing-library/react in happy-dom
```

## Architecture

```text
src/
├── App.tsx                  React Router SPA shell
├── main.tsx                 Entry point (mount to #root)
├── types.ts                 Shared TypeScript types
├── lib/
│   └── api.ts               TanStack Query hooks (fetch + Bearer auth)
├── hooks/
│   ├── useSSE.ts            SSE consumer (fetch + ReadableStream, POST /chat)
│   └── useApi.ts            REST fetch wrapper (Bearer auth)
└── components/
    ├── chat/
    │   ├── ChatWindow.tsx        Main chat interface (route "/")
    │   ├── MessageList.tsx       Message list with auto-scroll
    │   ├── MessageBubble.tsx     Markdown rendering + streaming cursor + ContentBlock parsing
    │   ├── ChatInput.tsx         Text input with send
    │   └── PhaseIndicator.tsx    Process phase animation (recalling/thinking/generating)
    ├── settings/
    │   ├── SettingsPanel.tsx     Settings dashboard
    │   ├── OCEANSliders.tsx      Five-dimension personality sliders
    │   ├── IdentityEditor.tsx    Identity name + personality description
    │   └── TonePresetPicker.tsx  Tone preset selector (5 presets)
    └── sessions/
        ├── SessionDrawer.tsx     Session list drawer
        └── SessionItem.tsx       Individual session entry
```

### SSE Streaming Flow

```text
User types message → POST /chat (fetch)
  → ReadableStream consumes SSE frames:
    {"type":"event","event":"process.phase","payload":{"phase":"thinking"}}
    {"type":"event","event":"memory.recall","payload":{"count":3,"top_match":"..."}}
    {"type":"event","event":"chat.delta","payload":{"content":"..."}}
    {"type":"event","event":"mind.transition","payload":{"pad":{...},"occ_emotion":"..."}}
    {"type":"chat_response","payload":{"text":"...","meta":{...}}}
  → PhaseIndicator shows current stage
  → MessageBubble renders streaming markdown with blinking cursor
```

### Settings Persistence

Settings changes (OCEAN, identity, tone) go through `PUT /settings/*` endpoints. The Python daemon applies them to `MindEngine` immediately via `reload_config()` and persists to SQLite. No YAML file write-back — daemon restores from SQLite on restart.
