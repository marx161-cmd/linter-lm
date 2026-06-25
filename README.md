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
pip install -r requirements.txt httpx pydantic onnx

LINTR_BACKEND_URL=http://127.0.0.1:11434 \
LINTR_HOST=127.0.0.1 \
LINTR_PORT=8099 \
python server.py
```

Point clients at:

```text
http://127.0.0.1:8099/v1
```

The proxy forwards to:

```text
${LINTR_BACKEND_URL}/v1/chat/completions
${LINTR_BACKEND_URL}/v1/models
${LINTR_BACKEND_URL}/v1/*
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `LINTR_BACKEND_URL` | `http://127.0.0.1:11434` | LLM backend root |
| `LINTR_HOST` | `127.0.0.1` | Listen host |
| `LINTR_PORT` | `8099` | Listen port |
| `LINTR_INTENSITY` | `mild` | Linting intensity (`off`/`mild`/`medium`/`high`) |
| `LINTR_DEBUG` | `0` | Set to `1` for verbose logging |
| `CONTEXTSTORE_DB_PATH` | `./upgrade/contextstore.db` | Single-DB path |
| `CONTEXTSTORE_DBS` | — | JSON map for multi-DB (`{"name":"path",...}`) |
| `CONTEXTSTORE_EMBEDDING_ENDPOINT` | `http://127.0.0.1:18084/v1/embeddings` | Embedding server |
| `CONTEXTSTORE_SIMILARITY_THRESHOLD` | `0.5` | Min cosine similarity |
| `CONTEXTSTORE_TOP_K` | `3` | Max hits per query |
| `CONTEXTSTORE_MAX_CHARS_PER_EMBED` | `3000` | Chunk size for embedding |

## Debug

```bash
curl http://127.0.0.1:8099/lintr/state
curl http://127.0.0.1:8099/lintr/state/default
curl http://127.0.0.1:8099/health   # includes active store names
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
