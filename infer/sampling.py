"""Sampling utilities for autoregressive text generation."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_logits(logits: ArrayLike) -> NDArray[np.float64]:
    """Return logits as a floating-point copy with a non-empty vocabulary axis."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("logits must not contain NaN or positive infinity")
    return values.copy()


def stable_softmax(logits: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Compute softmax without overflowing for large finite logits.

    Negative infinity is supported for masked entries. Every slice along ``axis``
    must contain at least one finite value.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("logits must be a non-empty array")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("logits must not contain NaN or positive infinity")

    try:
        maximum = np.max(values, axis=axis, keepdims=True)
    except (np.exceptions.AxisError, TypeError) as exc:
        raise ValueError(f"invalid softmax axis: {axis}") from exc
    if np.isneginf(maximum).any():
        raise ValueError("each softmax slice must contain a finite logit")

    exponentials = np.exp(values - maximum)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def scale_temperature(logits: ArrayLike, temperature: float = 1.0) -> NDArray[np.float64]:
    """Scale logits by a finite, strictly positive temperature."""
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise ValueError("temperature must be a finite number greater than zero")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite number greater than zero")
    return _as_logits(logits) / temperature


def top_k_filter(logits: ArrayLike, top_k: int | None = None) -> NDArray[np.float64]:
    """Keep exactly the ``top_k`` largest logits along the last axis."""
    values = _as_logits(logits)
    if top_k is None:
        return values
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
        raise ValueError("top_k must be a positive integer or None")

    vocab_size = values.shape[-1]
    if top_k >= vocab_size:
        return values

    # Stable sorting makes tied logits deterministic by preferring lower ids.
    keep = np.argsort(-values, axis=-1, kind="stable")[..., : int(top_k)]
    filtered = np.full_like(values, -np.inf)
    np.put_along_axis(filtered, keep, np.take_along_axis(values, keep, axis=-1), axis=-1)
    return filtered


def top_p_filter(logits: ArrayLike, top_p: float | None = None) -> NDArray[np.float64]:
    """Keep the smallest high-probability token set whose mass reaches ``top_p``."""
    values = _as_logits(logits)
    if top_p is None:
        return values
    if isinstance(top_p, bool) or not isinstance(top_p, Real):
        raise ValueError("top_p must be in the interval (0, 1] or None")
    top_p = float(top_p)
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1] or None")
    if top_p == 1.0:
        return values

    order = np.argsort(-values, axis=-1, kind="stable")
    sorted_logits = np.take_along_axis(values, order, axis=-1)
    cumulative = np.cumsum(stable_softmax(sorted_logits, axis=-1), axis=-1)

    # The token that crosses the threshold belongs to the nucleus.
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1]
    remove[..., 0] = False

    sorted_filtered = np.where(remove, -np.inf, sorted_logits)
    filtered = np.empty_like(values)
    np.put_along_axis(filtered, order, sorted_filtered, axis=-1)
    return filtered


def sample_token(
    logits: ArrayLike,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token id from a one-dimensional logits vector."""
    values = np.asarray(logits)
    if values.ndim != 1:
        raise ValueError("sample_token expects a one-dimensional logits vector")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")

    filtered = scale_temperature(values, temperature)
    filtered = top_k_filter(filtered, top_k)
    filtered = top_p_filter(filtered, top_p)
    probabilities = stable_softmax(filtered)
    generator = rng if rng is not None else np.random.default_rng()
    return int(generator.choice(probabilities.size, p=probabilities))


__all__ = [
    "sample_token",
    "scale_temperature",
    "stable_softmax",
    "top_k_filter",
    "top_p_filter",
]
