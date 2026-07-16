"""Data structures for TinyGPT's per-layer, per-head KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class HeadKVCache:
    """Key and value history for one attention head."""

    key: FloatArray
    value: FloatArray


@dataclass(frozen=True)
class KVCache:
    """KV history indexed as ``layers[layer][head]``."""

    layers: tuple[tuple[HeadKVCache, ...], ...]

    @property
    def batch_size(self) -> int:
        return int(self.layers[0][0].key.shape[0])

    @property
    def length(self) -> int:
        return int(self.layers[0][0].key.shape[1])


__all__ = ["HeadKVCache", "KVCache"]
