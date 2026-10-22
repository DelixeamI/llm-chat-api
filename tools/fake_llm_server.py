"""A stand-in for the model: an OpenAI-compatible server that needs no GPU.

For trying the whole stack without Ollama and for the smoke test in CI. It is deterministic:
the reply is built from the request, token counts come from the text length. It speaks the part
of the API the service uses: chat completions, plain and streamed, with usage, and JSON-schema
structured output. A few magic words in the last user message trigger failures on demand:

    __503__       answer 503 (a transient error: the service should retry, then give up)
    __429__       answer 429
    __slow__      wait 3 seconds before answering

Run: python tools/fake_llm_server.py   (port 9000, or FAKE_LLM_PORT)
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="fake-llm")


def _last_user(messages: list[dict[str, Any]]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def _tokens(text: str) -> int:
    return max(1, len(text.split()))


def _fill(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """A value that satisfies a JSON Schema: enough for the schemas this service sends."""
    if "$ref" in schema:
        return _fill(defs[schema["$ref"].split("/")[-1]], defs)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {k: _fill(v, defs) for k, v in schema.get("properties", {}).items()}
    if kind == "number" or kind == "integer":
        return schema.get("minimum", 0.5) if kind == "number" else 1
    if kind == "string":
        return "fake summary"
    if kind == "array":
        return []
    if kind == "boolean":
        return True
    return None


def _answer(body: dict[str, Any]) -> str:
    response_format = body.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        schema = response_format["json_schema"]["schema"]
        value = _fill(schema, schema.get("$defs", {}))
        # A plausible confidence rather than the schema's lower bound
        if isinstance(value, dict) and "confidence" in value:
            value["confidence"] = 0.9
        return json.dumps(value, ensure_ascii=False)
    return f"echo: {_last_user(body['messages'])}"


async def _misbehave(prompt: str) -> None:
    if "__503__" in prompt:
        raise HTTPException(503, "fake overload")
    if "__429__" in prompt:
        raise HTTPException(429, "fake rate limit")
    if "__slow__" in prompt:
        await asyncio.sleep(3)


def _usage(body: dict[str, Any], answer: str) -> dict[str, int]:
    prompt = sum(_tokens(m["content"]) for m in body["messages"])
    completion = _tokens(answer)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _chunk(content: str | None, finish: str | None, **extra: Any) -> str:
    choices = (
        [
            {
                "index": 0,
                "delta": {"content": content} if content is not None else {},
                "finish_reason": finish,
            }
        ]
        if "usage" not in extra
        else []
    )
    payload = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "fake",
        "choices": choices,
    } | extra
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream(body: dict[str, Any], answer: str) -> AsyncIterator[str]:
    for word in answer.split(" "):
        yield _chunk(word + " ", None)
        await asyncio.sleep(0.01)
    yield _chunk(None, "stop")
    if (body.get("stream_options") or {}).get("include_usage"):
        yield _chunk(None, None, usage=_usage(body, answer))
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": "fake", "object": "model"}]}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(body: dict[str, Any]) -> Any:
    await _misbehave(_last_user(body["messages"]))
    answer = _answer(body)
    if body.get("stream"):
        return StreamingResponse(_stream(body, answer), media_type="text/event-stream")
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "fake"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(body, answer),
    }


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.environ.get("FAKE_LLM_PORT", "9000")), log_level="warning"
    )
