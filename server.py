from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
import uvicorn

from lintr_core import LintrEngine, conversation_id


BACKEND_URL = os.environ.get("LINTR_BACKEND_URL", "http://127.0.0.1:11434").rstrip("/")
INTENSITY = os.environ.get("LINTR_INTENSITY", "mild")
DEBUG = os.environ.get("LINTR_DEBUG", "0") == "1"

app = FastAPI(title="LinteR-LM Proxy", version="0.1.0")
engine = LintrEngine(intensity=INTENSITY)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "backend_url": BACKEND_URL, "intensity": INTENSITY}


@app.get("/lintr/state")
async def state_all() -> dict[str, Any]:
    return engine.debug_state()


@app.get("/lintr/state/{session_id}")
async def state_one(session_id: str) -> dict[str, Any]:
    return engine.debug_state(session_id)


@app.get("/v1/models")
async def models(request: Request) -> Response:
    return await proxy_raw(request, "/v1/models")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_other(path: str, request: Request) -> Response:
    return await proxy_raw(request, f"/v1/{path}")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    session_id = conversation_id(payload)
    patched_payload = engine.begin_request(session_id, payload)
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
    return {k: v for k, v in request.headers.items() if k.lower() not in blocked}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.environ.get("LINTR_HOST", "127.0.0.1"),
        port=int(os.environ.get("LINTR_PORT", "8099")),
        reload=False,
    )
