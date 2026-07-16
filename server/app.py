from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .backend import ChatBackend, NumpyBackend, format_prompt


MODEL_NAME = "tiny-gpt"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=64, ge=1, le=4096)
    stream: bool = False


def _get_backend(request: Request) -> ChatBackend:
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        try:
            backend = NumpyBackend(request.app.state.export_dir)
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"model unavailable: {error}") from error
        request.app.state.backend = backend
    return backend


def _usage(prompt: str, completion: str) -> dict[str, int]:
    # The bundled tokenizer is character-level, so character counts equal token counts.
    prompt_tokens = len(prompt)
    completion_tokens = len(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse_chunks(completion_id: str, created: int, model: str, text: str) -> Iterator[str]:
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
    for character in text:
        chunk = {**base, "choices": [{"index": 0, "delta": {"content": character}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(backend: ChatBackend | None = None, *, export_dir: str = "exports") -> FastAPI:
    application = FastAPI(title="TinyGPT API", version="1.0.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )
    application.state.backend = backend
    application.state.export_dir = export_dir

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/v1/chat/completions")
    def chat_completions(
        body: ChatCompletionRequest,
        inference: Annotated[ChatBackend, Depends(_get_backend)],
    ):
        prompt = format_prompt([(message.role, message.content) for message in body.messages])
        try:
            text = inference.generate(
                prompt,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if body.stream:
            return StreamingResponse(
                _sse_chunks(completion_id, created, body.model, text),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
            ],
            "usage": _usage(prompt, text),
        }

    return application


app = create_app()


__all__ = ["app", "create_app"]
