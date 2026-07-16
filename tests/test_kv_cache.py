from __future__ import annotations

import numpy as np
import pytest

from infer.kv_cache import KVCache
from infer.numpy_engine import CharTokenizer, ModelConfig, TinyGPT


def _tiny_model() -> TinyGPT:
    config = ModelConfig(
        vocab_size=7,
        block_size=4,
        n_embd=4,
        n_head=2,
        n_layer=2,
        head_size=2,
    )
    rng = np.random.default_rng(42)

    def weight(shape: tuple[int, ...]) -> np.ndarray:
        return rng.normal(0.0, 0.2, shape).astype(np.float32)

    weights = {
        "token_embedding_table.weight": weight((7, 4)),
        "position_embedding_table.weight": weight((4, 4)),
        "ln_f.weight": weight((4,)),
        "ln_f.bias": weight((4,)),
        "lm_head.weight": weight((7, 4)),
        "lm_head.bias": weight((7,)),
    }
    for block in range(config.n_layer):
        prefix = f"blocks.{block}"
        weights.update(
            {
                f"{prefix}.sa.proj.weight": weight((4, 4)),
                f"{prefix}.sa.proj.bias": weight((4,)),
                f"{prefix}.ffwd.net.0.weight": weight((16, 4)),
                f"{prefix}.ffwd.net.0.bias": weight((16,)),
                f"{prefix}.ffwd.net.2.weight": weight((4, 16)),
                f"{prefix}.ffwd.net.2.bias": weight((4,)),
                f"{prefix}.ln1.weight": weight((4,)),
                f"{prefix}.ln1.bias": weight((4,)),
                f"{prefix}.ln2.weight": weight((4,)),
                f"{prefix}.ln2.bias": weight((4,)),
            }
        )
        for head in range(config.n_head):
            for role in ("key", "query", "value"):
                weights[f"{prefix}.sa.heads.{head}.{role}.weight"] = weight((2, 4))

    tokenizer = CharTokenizer(
        {
            "type": "char",
            "vocab_size": 7,
            "stoi": {char: index for index, char in enumerate("abcdefg")},
            "itos": {str(index): char for index, char in enumerate("abcdefg")},
        }
    )
    return TinyGPT(config, weights, tokenizer)


def test_full_cached_forward_matches_regular_forward_and_records_every_head() -> None:
    model = _tiny_model()
    token_ids = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    expected = model.forward(token_ids)
    actual, cache = model.forward_cached(token_ids)

    np.testing.assert_array_equal(actual, expected)
    assert isinstance(cache, KVCache)
    assert cache.batch_size == 2
    assert cache.length == 3
    assert len(cache.layers) == model.config.n_layer
    assert all(len(layer) == model.config.n_head for layer in cache.layers)
    for layer in cache.layers:
        for head in layer:
            assert head.key.shape == (2, 3, model.config.head_size)
            assert head.value.shape == head.key.shape


def test_incremental_cached_logits_match_full_prefix_forward() -> None:
    model = _tiny_model()
    token_ids = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
    cache = None

    for position in range(token_ids.shape[1]):
        actual, cache = model.forward_cached(token_ids[:, position : position + 1], cache)
        expected = model.forward(token_ids[:, : position + 1])[:, -1:, :]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    assert cache is not None and cache.length == model.config.block_size


def test_cached_forward_rejects_growth_past_context_window() -> None:
    model = _tiny_model()
    _, cache = model.forward_cached(np.asarray([[0, 1, 2, 3]], dtype=np.int64))

    with pytest.raises(ValueError, match="rebuild"):
        model.forward_cached(np.asarray([[4]], dtype=np.int64), cache)


def test_cached_and_uncached_generation_match_across_window_rebuilds() -> None:
    model = _tiny_model()
    prompt = np.asarray([[0, 1], [2, 3]], dtype=np.int64)

    cached = model.generate(prompt, max_new_tokens=11, use_kv_cache=True)
    uncached = model.generate(prompt, max_new_tokens=11, use_kv_cache=False)

    np.testing.assert_array_equal(cached, uncached)


def test_generate_integrates_sampling_options_and_seeded_rng() -> None:
    model = _tiny_model()
    prompt = np.asarray([0, 1], dtype=np.int64)

    first = model.generate(
        prompt,
        8,
        temperature=0.8,
        top_k=4,
        top_p=0.9,
        rng=np.random.default_rng(123),
    )
    second = model.generate(
        prompt,
        8,
        temperature=0.8,
        top_k=4,
        top_p=0.9,
        rng=np.random.default_rng(123),
    )
    uncached = model.generate(
        prompt,
        8,
        temperature=0.8,
        top_k=4,
        top_p=0.9,
        rng=np.random.default_rng(123),
        use_kv_cache=False,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, uncached)

    greedy = model.generate(prompt, 8, temperature=0)
    top_one = model.generate(prompt, 8, temperature=1.0, top_k=1)
    np.testing.assert_array_equal(top_one, greedy)


def test_generate_text_keeps_old_call_and_accepts_sampling_options() -> None:
    model = _tiny_model()

    greedy = model.generate_text("ab", 3)
    sampled = model.generate_text(
        "ab",
        3,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        rng=np.random.default_rng(9),
    )

    assert sampled == greedy
    assert greedy.startswith("ab")


@pytest.mark.parametrize("temperature", [-1.0, np.nan, np.inf, True])
def test_generate_validates_temperature(temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        _tiny_model().generate(
            np.asarray([0]),
            1,
            temperature=temperature,  # type: ignore[arg-type]
        )
