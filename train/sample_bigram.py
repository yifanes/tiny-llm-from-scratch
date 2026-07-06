from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bigram import (
    CHECKPOINT_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    BigramLanguageModel,
    decode,
    load_tokenizer,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained bigram checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "bigram.pt",
        help="Path to the bigram checkpoint.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device()
    tokenizer = load_tokenizer(DATA_DIR / "tokenizer.json")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint["model_type"] != "bigram":
        raise ValueError(f"expected bigram checkpoint, got {checkpoint['model_type']}")

    vocab_size = int(checkpoint["vocab_size"])
    model = BigramLanguageModel(vocab_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint.relative_to(PROJECT_ROOT)}")
    print(f"vocab_size: {vocab_size}")
    print("sample:")
    print(decode(generated, tokenizer))


if __name__ == "__main__":
    main()

