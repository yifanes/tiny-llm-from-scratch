from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "tiny_gpt.pt"
TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer.json"
OUTPUT_DIR = PROJECT_ROOT / "exports"

MAGIC = b"TLLMWGT1"
FORMAT_NAME = "tiny-llm-model"
FORMAT_VERSION = 1
HEADER_LENGTH_STRUCT = struct.Struct("<Q")
REQUIRED_CONFIG_KEYS = ("block_size", "n_embd", "n_head", "n_layer")


def describe_tensor(name: str) -> str:
    if name == "token_embedding_table.weight":
        return "Maps token ids to token embeddings."
    if name == "position_embedding_table.weight":
        return "Maps sequence positions to position embeddings."
    if ".sa.heads." in name:
        role = name.rsplit(".", 2)[-2]
        return f"Attention head {role} projection."
    if ".sa.proj.weight" in name:
        return "Mixes concatenated attention head outputs."
    if ".sa.proj.bias" in name:
        return "Bias for the attention output projection."
    if ".ffwd.net.0." in name:
        return "Feed-forward expansion projection."
    if ".ffwd.net.2." in name:
        return "Feed-forward contraction projection."
    if ".ln1." in name:
        return "LayerNorm before multi-head attention."
    if ".ln2." in name:
        return "LayerNorm before the feed-forward network."
    if name.startswith("ln_f."):
        return "Final LayerNorm parameter."
    if name.startswith("lm_head."):
        return "Projects hidden states to vocabulary logits."
    return "Model parameter."


def tensor_to_bytes(tensor: torch.Tensor) -> tuple[np.ndarray, bytes]:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    array = array.astype(np.dtype("<f4"), copy=False)
    return array, array.tobytes(order="C")


def write_model_bin(
    tensors: list[tuple[str, torch.Tensor]],
    path: Path,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    offset = 0

    for name, tensor in tensors:
        array, payload = tensor_to_bytes(tensor)
        entries.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": "float32",
                "offset": offset,
                "nbytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "description": describe_tensor(name),
            }
        )
        payloads.append(payload)
        offset += len(payload)

    header: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "byte_order": "little",
        "offset_basis": "start_of_tensor_data",
        "tensor_count": len(entries),
        "data_nbytes": offset,
        "tensors": entries,
    }
    header_bytes = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        file.write(MAGIC)
        file.write(HEADER_LENGTH_STRUCT.pack(len(header_bytes)))
        file.write(header_bytes)
        for payload in payloads:
            file.write(payload)
    temporary_path.replace(path)
    return header


def read_model_bin(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with path.open("rb") as file:
        if file.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} has an invalid model magic header")

        header_length_bytes = file.read(HEADER_LENGTH_STRUCT.size)
        if len(header_length_bytes) != HEADER_LENGTH_STRUCT.size:
            raise ValueError(f"{path} is missing its header length")
        (header_length,) = HEADER_LENGTH_STRUCT.unpack(header_length_bytes)

        header_bytes = file.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{path} contains a truncated JSON header")
        header = json.loads(header_bytes)
        if header.get("format") != FORMAT_NAME or header.get("version") != FORMAT_VERSION:
            raise ValueError(f"{path} uses an unsupported weight format")

        tensor_data = file.read()

    if len(tensor_data) != header["data_nbytes"]:
        raise ValueError(
            f"{path} tensor data size mismatch: "
            f"expected {header['data_nbytes']}, got {len(tensor_data)}"
        )

    arrays: dict[str, np.ndarray] = {}
    for entry in header["tensors"]:
        start = int(entry["offset"])
        end = start + int(entry["nbytes"])
        payload = tensor_data[start:end]
        if len(payload) != entry["nbytes"]:
            raise ValueError(f"{path} tensor {entry['name']} is truncated")
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"{path} tensor {entry['name']} failed checksum validation")

        array = np.frombuffer(payload, dtype=np.dtype("<f4")).reshape(entry["shape"])
        arrays[entry["name"]] = array.copy()

    if len(arrays) != header["tensor_count"]:
        raise ValueError(f"{path} tensor count does not match its header")
    return header, arrays


def load_tokenizer(path: Path) -> dict[str, Any]:
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {"type", "vocab_size", "stoi", "itos"}
    missing = required_keys - tokenizer.keys()
    if missing:
        raise ValueError(f"tokenizer is missing keys: {sorted(missing)}")
    if tokenizer["type"] != "char":
        raise ValueError(f"unsupported tokenizer type: {tokenizer['type']}")
    if len(tokenizer["stoi"]) != tokenizer["vocab_size"]:
        raise ValueError("tokenizer stoi size does not match vocab_size")
    if len(tokenizer["itos"]) != tokenizer["vocab_size"]:
        raise ValueError("tokenizer itos size does not match vocab_size")
    return tokenizer


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")

    required_keys = {"model", "model_type", "vocab_size", "gpt_config"}
    missing = required_keys - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    if checkpoint["model_type"] != "tiny_gpt":
        raise ValueError(f"unsupported model type: {checkpoint['model_type']}")
    if not isinstance(checkpoint["model"], dict):
        raise ValueError("checkpoint model must be a state dictionary")
    return checkpoint


def export_checkpoint(
    checkpoint_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    verify: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    tokenizer = load_tokenizer(tokenizer_path)
    gpt_config = checkpoint["gpt_config"]

    missing_config = [key for key in REQUIRED_CONFIG_KEYS if key not in gpt_config]
    if missing_config:
        raise ValueError(f"gpt_config is missing keys: {missing_config}")

    vocab_size = int(checkpoint["vocab_size"])
    if tokenizer["vocab_size"] != vocab_size:
        raise ValueError(
            "checkpoint and tokenizer vocab sizes differ: "
            f"{vocab_size} != {tokenizer['vocab_size']}"
        )

    n_embd = int(gpt_config["n_embd"])
    n_head = int(gpt_config["n_head"])
    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    state_dict = checkpoint["model"]
    tensors = [
        (name, tensor)
        for name, tensor in state_dict.items()
        if not name.endswith(".tril")
    ]
    if not tensors:
        raise ValueError("checkpoint does not contain exportable model parameters")
    if any(not isinstance(tensor, torch.Tensor) for _, tensor in tensors):
        raise ValueError("checkpoint model entries must be tensors")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.bin"
    header = write_model_bin(tensors, model_path)

    parameter_count = sum(tensor.numel() for _, tensor in tensors)
    config = {
        "format_version": 1,
        "model_type": "tiny_gpt",
        "vocab_size": vocab_size,
        "block_size": int(gpt_config["block_size"]),
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": int(gpt_config["n_layer"]),
        "head_size": n_embd // n_head,
        "dropout": float(gpt_config.get("dropout", 0.0)),
        "parameter_count": parameter_count,
        "tensor_count": len(tensors),
        "weights": {
            "file": model_path.name,
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "dtype": "float32",
        },
    }
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(tokenizer_path, output_dir / "tokenizer.json")

    if verify:
        loaded_header, arrays = read_model_bin(model_path)
        if loaded_header != header:
            raise ValueError("model.bin header changed during round-trip verification")
        for name, tensor in tensors:
            expected, _ = tensor_to_bytes(tensor)
            if not np.array_equal(arrays[name], expected):
                raise ValueError(f"model.bin tensor {name} failed round-trip verification")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a tiny GPT checkpoint for the NumPy inference engine."
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = export_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        verify=not args.no_verify,
    )
    print(f"config: {args.output_dir / 'config.json'}")
    print(f"tokenizer: {args.output_dir / 'tokenizer.json'}")
    print(f"weights: {args.output_dir / 'model.bin'}")
    print(f"tensors: {config['tensor_count']}")
    print(f"parameters: {config['parameter_count']:,}")


if __name__ == "__main__":
    main()
