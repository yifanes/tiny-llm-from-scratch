from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from export.export_weights import export_checkpoint, read_model_bin, write_model_bin


def test_model_bin_round_trip(tmp_path: Path) -> None:
    tensors = [
        ("matrix.weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("vector.bias", torch.tensor([-1.5, 0.25])),
    ]
    path = tmp_path / "model.bin"

    written_header = write_model_bin(tensors, path)
    loaded_header, arrays = read_model_bin(path)

    assert loaded_header == written_header
    assert loaded_header["tensor_count"] == 2
    np.testing.assert_array_equal(arrays["matrix.weight"], tensors[0][1].numpy())
    np.testing.assert_array_equal(arrays["vector.bias"], tensors[1][1].numpy())


def test_export_checkpoint_excludes_rebuildable_mask(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "tiny_gpt.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    output_dir = tmp_path / "exports"
    torch.save(
        {
            "model": {
                "token_embedding_table.weight": torch.arange(12).reshape(3, 4),
                "blocks.0.sa.heads.0.tril": torch.tril(torch.ones(2, 2)),
                "blocks.0.sa.heads.0.key.weight": torch.arange(8).reshape(2, 4),
            },
            "model_type": "tiny_gpt",
            "vocab_size": 3,
            "gpt_config": {
                "block_size": 2,
                "n_embd": 4,
                "n_head": 2,
                "n_layer": 1,
                "dropout": 0.0,
            },
        },
        checkpoint_path,
    )
    tokenizer_path.write_text(
        json.dumps(
            {
                "type": "char",
                "vocab_size": 3,
                "stoi": {"a": 0, "b": 1, "c": 2},
                "itos": {"0": "a", "1": "b", "2": "c"},
            }
        ),
        encoding="utf-8",
    )

    config = export_checkpoint(checkpoint_path, tokenizer_path, output_dir)
    header, arrays = read_model_bin(output_dir / "model.bin")

    assert config["head_size"] == 2
    assert config["tensor_count"] == 2
    assert header["tensor_count"] == 2
    assert "blocks.0.sa.heads.0.tril" not in arrays
    assert set(arrays) == {
        "token_embedding_table.weight",
        "blocks.0.sa.heads.0.key.weight",
    }


def test_export_rejects_tokenizer_vocab_mismatch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "tiny_gpt.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "model_type": "tiny_gpt",
            "vocab_size": 2,
            "gpt_config": {
                "block_size": 2,
                "n_embd": 4,
                "n_head": 2,
                "n_layer": 1,
            },
        },
        checkpoint_path,
    )
    tokenizer_path.write_text(
        json.dumps(
            {
                "type": "char",
                "vocab_size": 1,
                "stoi": {"a": 0},
                "itos": {"0": "a"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vocab sizes differ"):
        export_checkpoint(checkpoint_path, tokenizer_path, tmp_path / "exports")
