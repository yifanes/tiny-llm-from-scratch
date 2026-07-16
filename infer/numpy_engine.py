from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .kv_cache import HeadKVCache, KVCache
from .sampling import sample_token


MAGIC = b"TLLMWGT1"
FORMAT_NAME = "tiny-llm-model"
FORMAT_VERSION = 1
HEADER_LENGTH = struct.Struct("<Q")
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.integer[Any]]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    head_size: int

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model_type") != "tiny_gpt":
            raise ValueError(f"unsupported model type: {data.get('model_type')!r}")
        config = cls(
            vocab_size=int(data["vocab_size"]),
            block_size=int(data["block_size"]),
            n_embd=int(data["n_embd"]),
            n_head=int(data["n_head"]),
            n_layer=int(data["n_layer"]),
            head_size=int(data["head_size"]),
        )
        if config.n_embd != config.n_head * config.head_size:
            raise ValueError("n_embd must equal n_head * head_size")
        return config


class CharTokenizer:
    """The character tokenizer saved alongside the exported model."""

    def __init__(self, data: dict[str, Any]) -> None:
        if data.get("type") != "char":
            raise ValueError(f"unsupported tokenizer type: {data.get('type')!r}")
        self.stoi = {str(char): int(index) for char, index in data["stoi"].items()}
        self.itos = {int(index): str(char) for index, char in data["itos"].items()}
        self.vocab_size = int(data["vocab_size"])
        expected_ids = set(range(self.vocab_size))
        if len(self.stoi) != self.vocab_size or set(self.itos) != expected_ids:
            raise ValueError("tokenizer vocabulary does not match vocab_size")

    @classmethod
    def from_json(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"character is not in the vocabulary: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int] | IntArray) -> str:
        try:
            return "".join(self.itos[int(token_id)] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"token id is not in the vocabulary: {error.args[0]}") from error


def load_weights(path: Path) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Read and validate the self-describing model.bin format without PyTorch."""
    with path.open("rb") as file:
        if file.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} has an invalid model magic header")
        length_bytes = file.read(HEADER_LENGTH.size)
        if len(length_bytes) != HEADER_LENGTH.size:
            raise ValueError(f"{path} is missing its header length")
        (header_length,) = HEADER_LENGTH.unpack(length_bytes)
        header_bytes = file.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{path} contains a truncated JSON header")
        header = json.loads(header_bytes)
        tensor_data = file.read()

    if header.get("format") != FORMAT_NAME or header.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path} uses an unsupported weight format")
    if header.get("byte_order") != "little":
        raise ValueError(f"{path} uses an unsupported byte order")
    if len(tensor_data) != int(header["data_nbytes"]):
        raise ValueError(f"{path} tensor data size does not match its header")

    weights: dict[str, FloatArray] = {}
    for entry in header["tensors"]:
        if entry["dtype"] != "float32":
            raise ValueError(f"unsupported dtype for {entry['name']}: {entry['dtype']}")
        start = int(entry["offset"])
        end = start + int(entry["nbytes"])
        payload = tensor_data[start:end]
        if len(payload) != int(entry["nbytes"]):
            raise ValueError(f"tensor {entry['name']} is truncated")
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"tensor {entry['name']} failed checksum validation")
        weights[entry["name"]] = np.frombuffer(payload, dtype="<f4").reshape(entry["shape"]).copy()

    if len(weights) != int(header["tensor_count"]):
        raise ValueError("tensor count does not match the model header")
    return header, weights


def linear(x: FloatArray, weight: FloatArray, bias: FloatArray | None = None) -> FloatArray:
    """PyTorch Linear convention: y = x @ weight.T + bias."""
    output = x @ weight.T
    if bias is not None:
        output = output + bias
    return np.asarray(output, dtype=np.float32)


def layer_norm(
    x: FloatArray,
    weight: FloatArray,
    bias: FloatArray,
    eps: float = 1e-5,
) -> FloatArray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    variance = np.mean(np.square(x - mean), axis=-1, keepdims=True, dtype=np.float32)
    normalized = (x - mean) / np.sqrt(variance + np.float32(eps))
    return np.asarray(normalized * weight + bias, dtype=np.float32)


def softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    # Subtracting the maximum leaves the result unchanged and prevents exp overflow.
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / np.sum(exponentials, axis=axis, keepdims=True), dtype=np.float32)


class TinyGPT:
    """Inference-only TinyGPT whose numerical operations are implemented in NumPy."""

    def __init__(
        self,
        config: ModelConfig,
        weights: dict[str, FloatArray],
        tokenizer: CharTokenizer | None = None,
    ) -> None:
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self._validate_weights()

    @classmethod
    def from_export(cls, export_dir: str | Path) -> "TinyGPT":
        directory = Path(export_dir)
        config = ModelConfig.from_json(directory / "config.json")
        tokenizer = CharTokenizer.from_json(directory / "tokenizer.json")
        _, weights = load_weights(directory / "model.bin")
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError("model and tokenizer vocabulary sizes differ")
        return cls(config, weights, tokenizer)

    def _validate_weights(self) -> None:
        required = {
            "token_embedding_table.weight": (self.config.vocab_size, self.config.n_embd),
            "position_embedding_table.weight": (self.config.block_size, self.config.n_embd),
            "ln_f.weight": (self.config.n_embd,),
            "ln_f.bias": (self.config.n_embd,),
            "lm_head.weight": (self.config.vocab_size, self.config.n_embd),
            "lm_head.bias": (self.config.vocab_size,),
        }
        for block in range(self.config.n_layer):
            prefix = f"blocks.{block}"
            required.update(
                {
                    f"{prefix}.sa.proj.weight": (self.config.n_embd, self.config.n_embd),
                    f"{prefix}.sa.proj.bias": (self.config.n_embd,),
                    f"{prefix}.ffwd.net.0.weight": (4 * self.config.n_embd, self.config.n_embd),
                    f"{prefix}.ffwd.net.0.bias": (4 * self.config.n_embd,),
                    f"{prefix}.ffwd.net.2.weight": (self.config.n_embd, 4 * self.config.n_embd),
                    f"{prefix}.ffwd.net.2.bias": (self.config.n_embd,),
                    f"{prefix}.ln1.weight": (self.config.n_embd,),
                    f"{prefix}.ln1.bias": (self.config.n_embd,),
                    f"{prefix}.ln2.weight": (self.config.n_embd,),
                    f"{prefix}.ln2.bias": (self.config.n_embd,),
                }
            )
            for head in range(self.config.n_head):
                for role in ("key", "query", "value"):
                    required[f"{prefix}.sa.heads.{head}.{role}.weight"] = (
                        self.config.head_size,
                        self.config.n_embd,
                    )
        for name, shape in required.items():
            if name not in self.weights:
                raise ValueError(f"model is missing required tensor: {name}")
            if self.weights[name].shape != shape:
                raise ValueError(
                    f"tensor {name} has shape {self.weights[name].shape}, expected {shape}"
                )

    def _head(self, x: FloatArray, block: int, head: int) -> FloatArray:
        prefix = f"blocks.{block}.sa.heads.{head}"
        key = linear(x, self.weights[f"{prefix}.key.weight"])
        query = linear(x, self.weights[f"{prefix}.query.weight"])
        value = linear(x, self.weights[f"{prefix}.value.weight"])

        scores = (query @ np.swapaxes(key, -2, -1)) * np.float32(
            self.config.head_size**-0.5
        )
        time_steps = x.shape[1]
        future_positions = np.triu(np.ones((time_steps, time_steps), dtype=bool), k=1)
        scores = np.where(future_positions, -np.inf, scores)
        attention = softmax(np.asarray(scores, dtype=np.float32))
        return np.asarray(attention @ value, dtype=np.float32)

    def _head_cached(
        self,
        x: FloatArray,
        block: int,
        head: int,
        cached: HeadKVCache | None,
        past_length: int,
    ) -> tuple[FloatArray, HeadKVCache]:
        prefix = f"blocks.{block}.sa.heads.{head}"
        key = linear(x, self.weights[f"{prefix}.key.weight"])
        query = linear(x, self.weights[f"{prefix}.query.weight"])
        value = linear(x, self.weights[f"{prefix}.value.weight"])

        if cached is not None:
            key = np.concatenate((cached.key, key), axis=1)
            value = np.concatenate((cached.value, value), axis=1)

        scores = (query @ np.swapaxes(key, -2, -1)) * np.float32(
            self.config.head_size**-0.5
        )
        new_steps = x.shape[1]
        query_positions = past_length + np.arange(new_steps)[:, None]
        key_positions = np.arange(key.shape[1])[None, :]
        scores = np.where(key_positions > query_positions, -np.inf, scores)
        attention = softmax(np.asarray(scores, dtype=np.float32))
        output = np.asarray(attention @ value, dtype=np.float32)
        return output, HeadKVCache(key=key, value=value)

    def _block(self, x: FloatArray, block: int) -> FloatArray:
        prefix = f"blocks.{block}"
        normalized = layer_norm(
            x,
            self.weights[f"{prefix}.ln1.weight"],
            self.weights[f"{prefix}.ln1.bias"],
        )
        heads = [self._head(normalized, block, head) for head in range(self.config.n_head)]
        attention_output = linear(
            np.concatenate(heads, axis=-1),
            self.weights[f"{prefix}.sa.proj.weight"],
            self.weights[f"{prefix}.sa.proj.bias"],
        )
        x = np.asarray(x + attention_output, dtype=np.float32)

        normalized = layer_norm(
            x,
            self.weights[f"{prefix}.ln2.weight"],
            self.weights[f"{prefix}.ln2.bias"],
        )
        hidden = linear(
            normalized,
            self.weights[f"{prefix}.ffwd.net.0.weight"],
            self.weights[f"{prefix}.ffwd.net.0.bias"],
        )
        hidden = np.maximum(hidden, np.float32(0.0))
        feed_forward = linear(
            hidden,
            self.weights[f"{prefix}.ffwd.net.2.weight"],
            self.weights[f"{prefix}.ffwd.net.2.bias"],
        )
        return np.asarray(x + feed_forward, dtype=np.float32)

    def _block_cached(
        self,
        x: FloatArray,
        block: int,
        cached: tuple[HeadKVCache, ...] | None,
        past_length: int,
    ) -> tuple[FloatArray, tuple[HeadKVCache, ...]]:
        prefix = f"blocks.{block}"
        normalized = layer_norm(
            x,
            self.weights[f"{prefix}.ln1.weight"],
            self.weights[f"{prefix}.ln1.bias"],
        )
        head_results = [
            self._head_cached(
                normalized,
                block,
                head,
                None if cached is None else cached[head],
                past_length,
            )
            for head in range(self.config.n_head)
        ]
        attention_output = linear(
            np.concatenate([result[0] for result in head_results], axis=-1),
            self.weights[f"{prefix}.sa.proj.weight"],
            self.weights[f"{prefix}.sa.proj.bias"],
        )
        x = np.asarray(x + attention_output, dtype=np.float32)

        normalized = layer_norm(
            x,
            self.weights[f"{prefix}.ln2.weight"],
            self.weights[f"{prefix}.ln2.bias"],
        )
        hidden = linear(
            normalized,
            self.weights[f"{prefix}.ffwd.net.0.weight"],
            self.weights[f"{prefix}.ffwd.net.0.bias"],
        )
        hidden = np.maximum(hidden, np.float32(0.0))
        feed_forward = linear(
            hidden,
            self.weights[f"{prefix}.ffwd.net.2.weight"],
            self.weights[f"{prefix}.ffwd.net.2.bias"],
        )
        return (
            np.asarray(x + feed_forward, dtype=np.float32),
            tuple(result[1] for result in head_results),
        )

    def _validate_token_ids(self, token_ids: IntArray) -> NDArray[np.integer[Any]]:
        ids = np.asarray(token_ids)
        if ids.ndim != 2:
            raise ValueError(f"token_ids must have shape (batch, time), got {ids.shape}")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("token_ids must contain integers")
        if ids.shape[1] < 1:
            raise ValueError("time dimension must contain at least one token")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("token id is outside the model vocabulary")
        return ids

    def _validate_cache(self, cache: KVCache, batch_size: int) -> int:
        if not isinstance(cache, KVCache):
            raise TypeError("cache must be a KVCache or None")
        if len(cache.layers) != self.config.n_layer:
            raise ValueError("cache layer count does not match the model")

        expected_length: int | None = None
        expected_shape_tail = (self.config.head_size,)
        for layer in cache.layers:
            if len(layer) != self.config.n_head:
                raise ValueError("cache head count does not match the model")
            for head in layer:
                if head.key.shape != head.value.shape:
                    raise ValueError("cached key and value shapes differ")
                if head.key.ndim != 3 or head.key.shape[0] != batch_size:
                    raise ValueError("cache batch size or rank does not match the input")
                if head.key.shape[2:] != expected_shape_tail:
                    raise ValueError("cache head size does not match the model")
                if expected_length is None:
                    expected_length = int(head.key.shape[1])
                elif head.key.shape[1] != expected_length:
                    raise ValueError("all cached heads must have the same length")
        if expected_length is None or expected_length < 1:
            raise ValueError("cache must contain at least one token")
        return expected_length

    def forward(self, token_ids: IntArray) -> FloatArray:
        """Return logits shaped (batch, time, vocab) for a 2-D token id array."""
        ids = self._validate_token_ids(token_ids)
        _, time_steps = ids.shape
        if time_steps > self.config.block_size:
            raise ValueError(f"time dimension must be between 1 and {self.config.block_size}")

        token_embedding = self.weights["token_embedding_table.weight"][ids]
        position_embedding = self.weights["position_embedding_table.weight"][:time_steps]
        x = np.asarray(token_embedding + position_embedding, dtype=np.float32)
        for block in range(self.config.n_layer):
            x = self._block(x, block)
        x = layer_norm(x, self.weights["ln_f.weight"], self.weights["ln_f.bias"])
        return linear(x, self.weights["lm_head.weight"], self.weights["lm_head.bias"])

    def forward_cached(
        self,
        token_ids: IntArray,
        cache: KVCache | None = None,
    ) -> tuple[FloatArray, KVCache]:
        """Return logits for new tokens and append their per-head K/V to ``cache``."""
        ids = self._validate_token_ids(token_ids)
        past_length = 0 if cache is None else self._validate_cache(cache, ids.shape[0])
        total_length = past_length + ids.shape[1]
        if total_length > self.config.block_size:
            raise ValueError(
                "cached context would exceed block_size; rebuild it from the cropped window"
            )

        token_embedding = self.weights["token_embedding_table.weight"][ids]
        position_embedding = self.weights["position_embedding_table.weight"][
            past_length:total_length
        ]
        x = np.asarray(token_embedding + position_embedding, dtype=np.float32)
        layers: list[tuple[HeadKVCache, ...]] = []
        for block in range(self.config.n_layer):
            x, layer_cache = self._block_cached(
                x,
                block,
                None if cache is None else cache.layers[block],
                past_length,
            )
            layers.append(layer_cache)
        x = layer_norm(x, self.weights["ln_f.weight"], self.weights["ln_f.bias"])
        logits = linear(x, self.weights["lm_head.weight"], self.weights["lm_head.bias"])
        return logits, KVCache(layers=tuple(layers))

    def generate(
        self,
        token_ids: IntArray,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        rng: np.random.Generator | None = None,
        use_kv_cache: bool = True,
    ) -> NDArray[np.int64]:
        """Append tokens with greedy or sampled decoding and an optional KV cache."""
        ids = np.asarray(token_ids)
        original_rank = ids.ndim
        if original_rank == 1:
            ids = ids[None, :]
        elif original_rank != 2:
            raise ValueError("token_ids must be a one- or two-dimensional array")
        if ids.shape[1] == 0:
            raise ValueError("generation needs at least one prompt token")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral):
            raise ValueError("max_new_tokens must be a non-negative integer")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if isinstance(temperature, bool) or not isinstance(temperature, Real):
            raise ValueError("temperature must be a finite number greater than or equal to zero")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be a finite number greater than or equal to zero")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator or None")
        if not isinstance(use_kv_cache, bool):
            raise TypeError("use_kv_cache must be a boolean")

        output = np.asarray(ids, dtype=np.int64)
        cache: KVCache | None = None
        generator = rng if rng is not None else np.random.default_rng()
        for _ in range(max_new_tokens):
            if use_kv_cache:
                new_input = (
                    output[:, -self.config.block_size :]
                    if cache is None
                    else output[:, -1:]
                )
                logits, cache = self.forward_cached(new_input, cache)
            else:
                context = output[:, -self.config.block_size :]
                logits = self.forward(context)

            next_logits = logits[:, -1, :]
            if temperature == 0.0:
                next_token = np.argmax(next_logits, axis=-1).astype(np.int64)
            else:
                next_token = np.asarray(
                    [
                        sample_token(
                            row,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            rng=generator,
                        )
                        for row in next_logits
                    ],
                    dtype=np.int64,
                )
            next_token = next_token[:, None]
            output = np.concatenate((output, next_token), axis=1)
            if use_kv_cache and cache is not None and cache.length == self.config.block_size:
                # Sliding the learned absolute positions changes every retained token.
                # Rebuilding on the next step preserves exact uncached semantics.
                cache = None
        return output[0] if original_rank == 1 else output

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        rng: np.random.Generator | None = None,
        use_kv_cache: bool = True,
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("generate_text requires a tokenizer")
        prompt_ids = np.asarray(self.tokenizer.encode(prompt), dtype=np.int64)
        generated = self.generate(
            prompt_ids,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
            use_kv_cache=use_kv_cache,
        )
        return self.tokenizer.decode(generated)


def load_model(export_dir: str | Path) -> TinyGPT:
    return TinyGPT.from_export(export_dir)
