# LinteR-LM Proxy

Small OpenAI-compatible proxy for testing the LinteR-LM V1 idea:

- bypass `<think>...</think>` text unchanged,
- detect visible-output degradation,
- apply a toned-down sampler patch to the next request,
- extract/repair simple malformed tool-call JSON in non-streaming responses,
- proxy `/v1/chat/completions` and `/v1/models` to an OpenAI-compatible backend.

This is intentionally V1: no rewind, no tail deletion, no stop/retry loop.

## Project Status

This repository is published as a public experiment. Use it, fork it, adapt it,
or strip it for parts. Do not expect maintenance, support, compatibility
guarantees, or a stable roadmap from the original author.

The code is meant to be readable and hackable before it is meant to be polished.
Treat it as a prototype for local LLM stacks, not production middleware.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

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
```

## Debug

```bash
curl http://127.0.0.1:8099/lintr/state
curl http://127.0.0.1:8099/lintr/state/default
```

## Environment

- `LINTR_BACKEND_URL`: backend root, default `http://127.0.0.1:11434`
- `LINTR_HOST`: listen host, default `127.0.0.1`
- `LINTR_PORT`: listen port, default `8099`
- `LINTR_INTENSITY`: `off`, `mild`, `medium`, or `high`, default `mild`
- `LINTR_DEBUG`: `1` to print interventions, default `0`

## V1 Behavior

Sampler patches are conversation-scoped and one-shot. The conversation id is
derived from `user`, `metadata.conversation_id`, `conversation_id`, or
`default`.

Streaming responses are passed through as SSE while the proxy monitors content
deltas. Non-streaming responses are inspected after completion and may have
recoverable tool-call JSON normalized in the assistant message content.

## Credits

The idea was developed from local experiments with small models, tool-calling
failure modes, and a SillyTavern patch by DavidAU that explored live stream
correction, reconsideration, and sampler perturbation. This V1 proxy does not
copy SillyTavern code; it keeps the lighter concept of observing output,
leaving hidden reasoning spans alone, and applying a small one-shot sampler
change after visible degradation is detected.

Thanks also to the broader Ollama, FastAPI, httpx, and SillyTavern communities
for the tools and patterns this prototype builds around.

## License

MIT. See [LICENSE](LICENSE).

Additional attribution and project expectations are in [NOTICE.md](NOTICE.md).
