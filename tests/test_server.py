import json
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import uvicorn

from server.app import create_app


class FakeBackend:
    def __init__(self, result: str = "好的") -> None:
        self.result = result
        self.calls = []

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.calls.append((prompt, temperature, max_tokens))
        return self.result


@contextmanager
def running_server(backend):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(backend), host="127.0.0.1", port=port, log_level="critical")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("test server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def post(base_url, payload, *, accept="application/json"):
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": accept},
        method="POST",
    )
    return urllib.request.urlopen(request)


def test_non_streaming_chat_completion() -> None:
    backend = FakeBackend()
    with running_server(backend) as base_url:
        response = post(base_url, {
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！"},
                {"role": "user", "content": "继续"},
            ],
            "temperature": 0.4,
            "max_tokens": 7,
        })
        body = json.load(response)

    assert response.status == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "好的"}
    assert body["usage"] == {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}
    assert backend.calls == [("你好\n你好！\n继续", 0.4, 7)]


def test_streaming_chat_completion_is_sse() -> None:
    with running_server(FakeBackend("ab")) as base_url:
        response = post(
            base_url,
            {"messages": [{"role": "user", "content": "x"}], "stream": True},
            accept="text/event-stream",
        )
        events = [line.decode().strip() for line in response if line.startswith(b"data: ")]

    assert response.headers["Content-Type"].startswith("text/event-stream")
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "ab"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_request_validation() -> None:
    with running_server(FakeBackend()) as base_url:
        for payload in (
            {"messages": []},
            {"messages": [{"role": "user", "content": "x"}], "temperature": -1},
        ):
            try:
                post(base_url, payload)
            except urllib.error.HTTPError as error:
                assert error.code == 422
            else:
                raise AssertionError("invalid request was accepted")
