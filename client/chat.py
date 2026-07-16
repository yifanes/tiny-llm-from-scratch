from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def stream_completion(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    temperature: float,
    max_tokens: int,
):
    payload = json.dumps(
        {
            "model": "tiny-gpt",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            content = event["choices"][0]["delta"].get("content")
            if content:
                yield content


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with the local TinyGPT server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    messages: list[dict[str, str]] = []
    print("TinyGPT chat. Enter /quit to exit.")
    while True:
        try:
            prompt = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.strip() in {"/quit", "/exit"}:
            return 0
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        print("TinyGPT> ", end="", flush=True)
        answer = ""
        try:
            for fragment in stream_completion(
                messages,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ):
                answer += fragment
                print(fragment, end="", flush=True)
        except urllib.error.URLError as error:
            print(f"\nRequest failed: {error}", file=sys.stderr)
            messages.pop()
            continue
        print()
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    raise SystemExit(main())
