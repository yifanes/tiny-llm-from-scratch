from pathlib import Path

import numpy as np
import pytest

from infer.numpy_engine import layer_norm, linear, load_model, softmax


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "exports"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "tiny_gpt.pt"


def test_numpy_primitives() -> None:
    x = np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    weight = np.asarray([[1.0, 2.0], [-1.0, 1.0]], dtype=np.float32)
    bias = np.asarray([0.5, -0.5], dtype=np.float32)
    np.testing.assert_allclose(
        linear(x, weight, bias),
        np.asarray([[5.5, 0.5], [13.5, 1.5]], dtype=np.float32),
    )

    probabilities = softmax(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-7)
    assert np.argmax(probabilities) == 2

    normalized = layer_norm(x, np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(normalized.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[:, 0], -normalized[:, 1], atol=1e-6)


@pytest.mark.skipif(not (EXPORT_DIR / "model.bin").exists(), reason="exported weights not present")
def test_model_load_forward_tokenizer_and_generation() -> None:
    model = load_model(EXPORT_DIR)
    prompt = "Tiny GPT"
    prompt_ids = np.asarray(model.tokenizer.encode(prompt), dtype=np.int64)
    assert model.tokenizer.decode(prompt_ids) == prompt

    logits = model.forward(prompt_ids[None, :])
    assert logits.shape == (1, len(prompt), model.config.vocab_size)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))

    generated = model.generate(prompt_ids, max_new_tokens=3)
    assert generated.shape == (len(prompt_ids) + 3,)
    np.testing.assert_array_equal(generated[: len(prompt_ids)], prompt_ids)


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists() or not (EXPORT_DIR / "model.bin").exists(),
    reason="checkpoint and exported weights are required",
)
def test_numpy_logits_match_pytorch_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    from train.tiny_gpt import GPTConfig, TinyGPTLanguageModel

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    config = GPTConfig(**checkpoint["gpt_config"])
    pytorch_model = TinyGPTLanguageModel(checkpoint["vocab_size"], config)
    pytorch_model.load_state_dict(checkpoint["model"])
    pytorch_model.eval()

    token_ids = np.asarray(
        [[0, 1, 2, 3, 4], [53, 100, 200, 300, 45]],
        dtype=np.int64,
    )
    with torch.no_grad():
        expected = pytorch_model(torch.from_numpy(token_ids))[0].numpy()
    actual = load_model(EXPORT_DIR).forward(token_ids)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
