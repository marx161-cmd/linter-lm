# LinteR-LM Proxy

Small OpenAI-compatible proxy for local LLM stacks with two independent features:

**Context injection (new)** — silent embedding-based knowledge retrieval that
splices relevant documents into the prompt before the model sees them.

**Output linting (V1)** — detect visible-output degradation, apply sampler
patches, and repair malformed tool-call JSON.

## Context Injection

The proxy maintains one or more SQLite + vector databases of ingested files.
On every `/v1/chat/completions` request, it:

1. Extracts the last user message
2. Embeds the query via a dedicated embedding server (llama-server `--embedding`)
3. Runs cosine similarity against all indexed chunks
4. Injects the best-matching files as `<context>` blocks into the prompt

The model receives the context transparently — no MCP, no tool call, no
model-visible step. The `upgrade/` module handles ingestion, chunking,
embedding, storage, and retrieval.

### Ingest documents

```bash
cd linter-lm
python3 -m upgrade.cli add /path/to/file --tags homelab --name "display-name"
python3 -m upgrade.cli list
python3 -m upgrade.cli rm <file_id>
```

### Multi-DB support

Configure via env var — a JSON map of store name → db path:

```bash
export CONTEXTSTORE_DBS='{"homelab":"/path/homelab.db","notes":"/path/notes.db"}'
```

All stores are queried in parallel on every message and results merged.
If `CONTEXTSTORE_DBS` is unset, falls back to a single default DB at
`CONTEXTSTORE_DB_PATH` (defaults to `linter-lm/upgrade/contextstore.db`).

### Embedding backend

Requires a running embedding server at `CONTEXTSTORE_EMBEDDING_ENDPOINT`
(defaults to `http://127.0.0.1:18084/v1/embeddings`). Any
OpenAI-compatible embeddings endpoint works.

## Output Linting (V1)

- Bypass `<think>...</think>` text unchanged
- Detect visible-output degradation
- Apply a toned-down sampler patch to the next request
- Extract/repair simple malformed tool-call JSON in non-streaming responses

This is intentionally V1: no rewind, no tail deletion, no stop/retry loop.

## Project Status

This repository is published as a public experiment. Use it, fork it, adapt it,
or strip it for parts. Do not expect maintenance, support, compatibility
guarantees, or a stable roadmap from the original author.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.local.example .env.local
# edit .env.local if needed, then:
source .env.local && python server.py
```

Point clients at `http://127.0.0.1:8099/v1`. The proxy forwards all `/v1/*`
traffic to `${LINTR_BACKEND_URL}`.

## Multi-Instance Setup (Local + Remote APIs)

Run two independent instances — one for a local Ollama/llama.cpp stack and one
for a remote API such as DeepSeek — on separate ports, each with its own
feature flags.

### 1. Create env files

```bash
cp .env.local.example    .env.local      # Ollama, port 8099, full features
cp .env.deepseek.example .env.deepseek   # DeepSeek, port 8098, context off
# edit .env.deepseek → set LINTR_BACKEND_API_KEY=sk-...
```

### 2. Install systemd user services

```bash
bash systemd/install.sh
systemctl --user enable --now linter-lm-local
systemctl --user enable --now linter-lm-deepseek
```

### 3. Expose via Tailscale serve (tailnet-only HTTPS)

```bash
bash bin/tailscale-setup.sh
```

This registers:

```
https://comrade.taile6163a.ts.net:8453  →  local   (Ollama,   port 8099)
https://comrade.taile6163a.ts.net:8452  →  deepseek (DeepSeek, port 8098)
```
Note: `:8451` is reserved for `homelab-mcp-hub`.

Point SillyTavern, Open WebUI, or any OpenAI-compatible client at either URL.

### 4. Control scripts

All bin scripts take an optional instance argument (`local` default, `deepseek`
or any raw URL). `LINTR_CTL_URL` overrides the argument entirely.

```bash
# from comrade (i3 status bar, keybindings)
lintr-status               # CTX:on LINT:mild REPAIR:on  (local)
lintr-status deepseek      # CTX:off LINT:mild REPAIR:on (deepseek)
lintr-toggle-linting       # cycle local instance: off→mild→medium→high→off
lintr-toggle-linting ds    # cycle deepseek instance
lintr-toggle-context       # toggle context inject on local
lintr-toggle-repair ds     # toggle JSON repair on deepseek
```

### 5. Termux shortcuts (Pixel)

Scripts in `bin/termux-shortcuts/` hit the Tailscale serve HTTPS URLs directly
— no SSH needed. Copy them to `~/.shortcuts/` on the Pixel and add via
Termux:Widget.

```bash
# on comrade — push to phone over ADB or Syncthing
adb push bin/termux-shortcuts/lintr-local-linting \
         /data/data/com.termux/files/home/.shortcuts/
```

Available shortcuts: `lintr-local-status`, `lintr-ds-status`,
`lintr-local-linting`, `lintr-ds-linting`, `lintr-local-context`,
`lintr-ds-context`.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `LINTR_BACKEND_URL` | `http://127.0.0.1:11434` | LLM backend root |
| `LINTR_BACKEND_API_KEY` | — | Bearer token injected into backend requests |
| `LINTR_HOST` | `127.0.0.1` | Listen host |
| `LINTR_PORT` | `8099` | Listen port |
| `LINTR_INTENSITY` | `mild` | Linting intensity (`off`/`mild`/`medium`/`high`) |
| `LINTR_CONTEXT_INJECT` | `1` | Set to `0` to disable context injection at startup |
| `LINTR_TOOL_REPAIR` | `1` | Set to `0` to disable tool JSON repair at startup |
| `LINTR_DEBUG` | `0` | Set to `1` for verbose logging |
| `CONTEXTSTORE_DB_PATH` | `./upgrade/contextstore.db` | Single-DB path |
| `CONTEXTSTORE_DBS` | — | JSON map for multi-DB (`{"name":"path",...}`) |
| `CONTEXTSTORE_EMBEDDING_ENDPOINT` | `http://127.0.0.1:18084/v1/embeddings` | Embedding server |
| `CONTEXTSTORE_SIMILARITY_THRESHOLD` | `0.5` | Min cosine similarity |
| `CONTEXTSTORE_TOP_K` | `3` | Max hits per query |
| `CONTEXTSTORE_MAX_CHARS_PER_EMBED` | `3000` | Chunk size for embedding |

## Debug

```bash
curl http://127.0.0.1:8099/health
curl http://127.0.0.1:8099/lintr/state
curl http://127.0.0.1:8099/lintr/state/default

# Feature flags (read + live toggle)
curl http://127.0.0.1:8099/lintr/features
curl http://127.0.0.1:8099/lintr/features/oneline
curl -X POST http://127.0.0.1:8099/lintr/features/context   # toggle context inject
curl -X POST http://127.0.0.1:8099/lintr/features/linting   # cycle intensity
curl -X POST http://127.0.0.1:8099/lintr/features/repair    # toggle JSON repair
```

## Architecture

```
Client → linter-lm (8099)
           ├─ context injection: embed query → cosine search SQLite → splice hits
           └─ output linting: monitor stream → detect degradation → patch sampler
              ↓
         LLM backend (e.g. llama-control 18080)
```

## Credits

The linting idea was developed from local experiments with small models,
tool-calling failure modes, and a SillyTavern patch by DavidAU. The context
injection module (`upgrade/`) is original to this repo.

Thanks to the Ollama, llama.cpp, FastAPI, httpx, and SillyTavern communities.

## License

MIT. See [LICENSE](LICENSE).
