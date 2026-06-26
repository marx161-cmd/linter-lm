from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator

from dotenv import load_dotenv
load_dotenv()  # must be before any os.environ.get() calls below

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
import httpx
import uvicorn

from lintr_core import LintrEngine, conversation_id
from upgrade.contextstore import ContextStore
from upgrade.chroma_backend import ChromaContextStore


BACKEND_URL = os.environ.get("LINTR_BACKEND_URL", "http://127.0.0.1:11434").rstrip("/")
BACKEND_API_KEY = os.environ.get("LINTR_BACKEND_API_KEY", "")
INTENSITY = os.environ.get("LINTR_INTENSITY", "mild")
DEBUG = os.environ.get("LINTR_DEBUG", "0") == "1"

_INTENSITY_CYCLE = ["off", "mild", "medium", "high"]


@dataclass
class FeatureFlags:
    context_inject: bool
    tool_repair: bool
    intensity: str


flags = FeatureFlags(
    context_inject=os.environ.get("LINTR_CONTEXT_INJECT", "1") != "0",
    tool_repair=os.environ.get("LINTR_TOOL_REPAIR", "1") != "0",
    intensity=INTENSITY,
)

# Context store backend selection:
#   CONTEXTSTORE_BACKEND=chroma  → query mcp-hub's ChromaDB (recommended, no ingestion needed)
#   CONTEXTSTORE_BACKEND=sqlite  → local SQLite stores (requires manual ingestion via upgrade.cli)
_backend = os.environ.get("CONTEXTSTORE_BACKEND", "chroma").lower()
if _backend == "chroma":
    context_stores: dict[str, Any] = {"chroma": ChromaContextStore()}
else:
    # Legacy SQLite multi-DB: CONTEXTSTORE_DBS = '{"name":"/path/to.db",...}'
    _raw_dbs = os.environ.get("CONTEXTSTORE_DBS", "")
    if _raw_dbs:
        _db_map = json.loads(_raw_dbs)
        context_stores = {name: ContextStore(db_path=path) for name, path in _db_map.items()}
    else:
        _default_path = os.environ.get(
            "CONTEXTSTORE_DB_PATH",
            os.path.join(os.path.dirname(__file__), "upgrade", "contextstore.db"),
        )
        context_stores = {"default": ContextStore(db_path=_default_path)}

app = FastAPI(title="LinteR-LM Proxy", version="0.3.0")
engine = LintrEngine(intensity=INTENSITY)
engine.tool_repair_enabled = flags.tool_repair

if DEBUG:
    for name, store in context_stores.items():
        files = store.list_files()
        print(f"CONTEXT store '{name}': {len(files)} files loaded")


# -- feature flag helpers ----------------------------------------------------

def _features_dict() -> dict[str, Any]:
    return {
        "context_inject": flags.context_inject,
        "tool_repair": flags.tool_repair,
        "intensity": flags.intensity,
    }


def _features_oneline() -> str:
    ctx = "CTX:on" if flags.context_inject else "CTX:off"
    lint = f"LINT:{flags.intensity}"
    repair = "REPAIR:on" if flags.tool_repair else "REPAIR:off"
    return f"{ctx} {lint} {repair}"


# -- endpoints ---------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "backend_url": BACKEND_URL,
        "intensity": flags.intensity,
        "stores": list(context_stores.keys()),
    }


@app.get("/lintr/state")
async def state_all() -> dict[str, Any]:
    return engine.debug_state()


@app.get("/lintr/state/{session_id}")
async def state_one(session_id: str) -> dict[str, Any]:
    return engine.debug_state(session_id)


@app.get("/lintr/features")
async def get_features() -> dict[str, Any]:
    return _features_dict()


@app.get("/lintr/features/oneline", response_class=PlainTextResponse)
async def get_features_oneline() -> str:
    return _features_oneline()


@app.post("/lintr/features/context")
async def toggle_context() -> dict[str, Any]:
    flags.context_inject = not flags.context_inject
    return _features_dict()


@app.post("/lintr/features/repair")
async def toggle_repair() -> dict[str, Any]:
    flags.tool_repair = not flags.tool_repair
    engine.tool_repair_enabled = flags.tool_repair
    return _features_dict()


@app.post("/lintr/features/linting")
async def cycle_linting() -> dict[str, Any]:
    try:
        idx = _INTENSITY_CYCLE.index(flags.intensity)
    except ValueError:
        idx = 0
    flags.intensity = _INTENSITY_CYCLE[(idx + 1) % len(_INTENSITY_CYCLE)]
    engine.scrambler.intensity = flags.intensity
    return _features_dict()


@app.get("/v1/models")
async def models(request: Request) -> Response:
    return await proxy_raw(request, "/v1/models")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    session_id = conversation_id(payload)
    patched_payload = engine.begin_request(session_id, payload)

    if flags.context_inject:
        patched_payload = await inject_context(patched_payload)

    stream = bool(patched_payload.get("stream"))
    url = f"{BACKEND_URL}/v1/chat/completions"
    headers = outbound_headers(request)

    if stream:
        return StreamingResponse(
            stream_chat(url, headers, patched_payload, session_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, headers=headers, json=patched_payload)
    if resp.headers.get("content-type", "").startswith("application/json"):
        data = resp.json()
        lint_non_streaming_response(session_id, data)
        return JSONResponse(content=data, status_code=resp.status_code)
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_other(path: str, request: Request) -> Response:
    return await proxy_raw(request, f"/v1/{path}")


# -- pipeline helpers --------------------------------------------------------

async def inject_context(payload: dict[str, Any]) -> dict[str, Any]:
    if DEBUG:
        print(f"CONTEXT inject_context called, stores={list(context_stores.keys())}")
    if not context_stores:
        return payload

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return payload

    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return payload

    user_msg = messages[last_user_idx]
    query_text = user_msg.get("content", "")
    if not isinstance(query_text, str) or not query_text.strip():
        return payload

    if DEBUG:
        print(f"CONTEXT querying '{query_text[:80]}...' across {len(context_stores)} stores")

    all_hits = []
    for name, store in context_stores.items():
        result = await store.retrieve(query_text)
        if result.hits:
            all_hits.append((name, result))

    if not all_hits:
        return payload

    context_blocks = []
    for store_name, result in all_hits:
        for hit in result.hits:
            context_blocks.append(
                f"[{store_name}: {hit.name} (score={hit.best_similarity:.3f})]\n{hit.content}"
            )
    context_text = "\n\n---\n\n".join(context_blocks)
    augmented = (
        f"<context>\nThe following relevant information is available:\n\n"
        f"{context_text}\n</context>\n\n{query_text}"
    )

    patched = dict(payload)
    patched["messages"] = list(messages)
    patched["messages"][last_user_idx] = {**user_msg, "content": augmented}
    if DEBUG:
        print(f"CONTEXT injected {sum(len(r.hits) for _, r in all_hits)} hits from {len(all_hits)} stores")
    return patched


async def proxy_raw(request: Request, path: str) -> Response:
    url = f"{BACKEND_URL}{path}"
    body = await request.body()
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.request(
            request.method,
            url,
            headers=outbound_headers(request),
            content=body,
            params=request.query_params,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


async def stream_chat(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    session_id: str,
) -> AsyncIterator[bytes]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    yield b"\n"
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() != "[DONE]":
                        observe_stream_event(session_id, data)
                yield (line + "\n").encode("utf-8")


def observe_stream_event(session_id: str, data_text: str) -> None:
    try:
        data = json.loads(data_text)
    except json.JSONDecodeError:
        return
    for choice in data.get("choices", []) or []:
        delta = choice.get("delta") or {}
        text = delta_text(delta)
        if text:
            interventions = engine.observe_delta(session_id, text)
            if DEBUG:
                for item in interventions:
                    print("LINTR", item)


def lint_non_streaming_response(session_id: str, data: dict[str, Any]) -> None:
    for choice in data.get("choices", []) or []:
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            message["content"] = engine.lint_final_text(session_id, content)


def delta_text(delta: dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def outbound_headers(request: Request) -> dict[str, str]:
    blocked = {"host", "content-length", "connection", "accept-encoding"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in blocked}
    if BACKEND_API_KEY:
        headers["authorization"] = f"Bearer {BACKEND_API_KEY}"
    return headers


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.environ.get("LINTR_HOST", "127.0.0.1"),
        port=int(os.environ.get("LINTR_PORT", "8099")),
        reload=False,
    )
