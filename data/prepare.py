from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw.txt"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
TRAIN_PATH = DATA_DIR / "train.bin"
VAL_PATH = DATA_DIR / "val.bin"
SPLIT_RATIO = 0.9


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[idx] for idx in ids)


def save_tokenizer(stoi: dict[str, int], path: Path) -> None:
    tokenizer = {
        "type": "char",
        "vocab_size": len(stoi),
        "stoi": stoi,
        "itos": {str(idx): ch for ch, idx in stoi.items()},
    }
    path.write_text(
        json.dumps(tokenizer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_tokens(ids: list[int], path: Path) -> None:
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)


def main() -> None:
    text = RAW_PATH.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"{RAW_PATH} is empty")

    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)

    decoded = decode(ids, itos)
    if decoded != text:
        raise ValueError("decode(encode(text)) did not match original text")

    split_idx = int(len(ids) * SPLIT_RATIO)
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    save_tokenizer(stoi, TOKENIZER_PATH)
    save_tokens(train_ids, TRAIN_PATH)
    save_tokens(val_ids, VAL_PATH)

    print(f"raw chars: {len(text)}")
    print(f"vocab size: {len(stoi)}")
    print(f"total tokens: {len(ids)}")
    print(f"train tokens: {len(train_ids)}")
    print(f"val tokens: {len(val_ids)}")
    print(f"tokenizer: {TOKENIZER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"train bin: {TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"val bin: {VAL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

