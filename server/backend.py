from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class ChatBackend(Protocol):
    """Small interface used by the HTTP layer and its tests."""

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str: ...


class NumpyBackend:
    """Adapter from the exported NumPy model to the chat service."""

    def __init__(self, export_dir: str | Path = "exports") -> None:
        from infer import load_model

        self.model = load_model(export_dir)

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        tokenizer = self.model.tokenizer
        if tokenizer is None:
            raise RuntimeError("the exported model does not include a tokenizer")
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("the prompt must contain at least one character")
        generated = self.model.generate(
            np.asarray(token_ids, dtype=np.int64),
            max_tokens,
            temperature=temperature,
            rng=np.random.default_rng(),
            use_kv_cache=True,
        )
        return tokenizer.decode(generated[len(token_ids) :])


def format_prompt(messages: list[tuple[str, str]]) -> str:
    """Flatten chat history for the character model without adding unknown labels."""
    return "\n".join(content for _, content in messages)


__all__ = ["ChatBackend", "NumpyBackend", "format_prompt"]
