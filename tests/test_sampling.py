from __future__ import annotations

import numpy as np
import pytest

from infer.sampling import (
    sample_token,
    scale_temperature,
    stable_softmax,
    top_k_filter,
    top_p_filter,
)


def test_stable_softmax_handles_large_logits_and_masks() -> None:
    probabilities = stable_softmax([10_000.0, 9_999.0, -np.inf])

    np.testing.assert_allclose(
        probabilities,
        [0.7310585786300049, 0.2689414213699951, 0.0],
    )
    assert probabilities.sum() == pytest.approx(1.0)


def test_stable_softmax_operates_along_requested_axis() -> None:
    probabilities = stable_softmax([[1.0, 2.0], [3.0, 4.0]], axis=0)

    np.testing.assert_allclose(probabilities.sum(axis=0), [1.0, 1.0])


@pytest.mark.parametrize("logits", [[-np.inf, -np.inf], [0.0, np.nan], [0.0, np.inf]])
def test_stable_softmax_rejects_invalid_slices(logits: list[float]) -> None:
    with pytest.raises(ValueError):
        stable_softmax(logits)


def test_temperature_scales_a_copy_of_logits() -> None:
    logits = np.array([1.0, 2.0, 3.0])

    scaled = scale_temperature(logits, 0.5)

    np.testing.assert_array_equal(scaled, [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(logits, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("temperature", [0, -1, np.inf, np.nan, True, "hot"])
def test_temperature_must_be_positive_and_finite(temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        scale_temperature([1.0], temperature)  # type: ignore[arg-type]


def test_top_k_keeps_exactly_k_tokens_with_deterministic_ties() -> None:
    filtered = top_k_filter([4.0, 4.0, 3.0, 2.0], top_k=1)

    np.testing.assert_array_equal(filtered, [4.0, -np.inf, -np.inf, -np.inf])


def test_top_k_supports_batches_and_large_k() -> None:
    logits = np.array([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]])

    np.testing.assert_array_equal(
        top_k_filter(logits, 2),
        [[-np.inf, 3.0, 2.0], [6.0, -np.inf, 5.0]],
    )
    np.testing.assert_array_equal(top_k_filter(logits, 10), logits)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_top_k_must_be_a_positive_integer(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        top_k_filter([1.0, 2.0], top_k)  # type: ignore[arg-type]


def test_top_p_keeps_smallest_nucleus_in_original_token_order() -> None:
    logits = np.log([0.10, 0.60, 0.25, 0.05])

    filtered = top_p_filter(logits, 0.70)

    assert np.isneginf(filtered[[0, 3]]).all()
    np.testing.assert_allclose(filtered[[1, 2]], logits[[1, 2]])


def test_top_p_always_keeps_at_least_one_token_and_supports_batches() -> None:
    logits = np.array([[3.0, 2.0, 1.0], [1.0, 4.0, 2.0]])

    filtered = top_p_filter(logits, 1e-9)

    np.testing.assert_array_equal(np.isfinite(filtered), [[True, False, False], [False, True, False]])


@pytest.mark.parametrize("top_p", [0, -0.1, 1.1, np.inf, np.nan, True])
def test_top_p_must_be_in_valid_interval(top_p: object) -> None:
    with pytest.raises(ValueError, match="top_p"):
        top_p_filter([1.0, 2.0], top_p)  # type: ignore[arg-type]


def test_seeded_sampling_is_deterministic() -> None:
    logits = [0.1, 0.2, 0.3, 0.4]
    first_rng = np.random.default_rng(2026)
    second_rng = np.random.default_rng(2026)

    first = [sample_token(logits, rng=first_rng) for _ in range(20)]
    second = [sample_token(logits, rng=second_rng) for _ in range(20)]

    assert first == second


def test_sampling_respects_combined_top_k_and_top_p_filters() -> None:
    rng = np.random.default_rng(7)

    samples = [
        sample_token([5.0, 4.0, 3.0, 2.0], top_k=2, top_p=0.5, rng=rng)
        for _ in range(20)
    ]

    assert samples == [0] * 20


def test_sample_token_requires_one_dimensional_logits_and_generator() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        sample_token([[1.0, 2.0]])
    with pytest.raises(TypeError, match="Generator"):
        sample_token([1.0, 2.0], rng=object())  # type: ignore[arg-type]
