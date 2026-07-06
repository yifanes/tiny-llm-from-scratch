from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    block_size: int = 32
    max_iters: int = 500
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 1e-2
    seed: int = 1337


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(idx)

        if targets is None:
            return logits, None

        batch_size, block_size, vocab_size = logits.shape
        logits_for_loss = logits.view(batch_size * block_size, vocab_size)
        targets_for_loss = targets.view(batch_size * block_size)
        loss = F.cross_entropy(logits_for_loss, targets_for_loss)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokens(path: Path) -> torch.Tensor:
    tokens = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    return torch.from_numpy(tokens)


def load_tokenizer(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    max_start = len(data) - config.block_size - 1
    if max_start <= 0:
        raise ValueError(f"{split} data is too short for block_size={config.block_size}")

    ix = torch.randint(max_start, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
    device: str,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for step in range(config.eval_iters):
            xb, yb = get_batch(split, train_data, val_data, config, device)
            _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("loss should not be None during evaluation")
            losses[step] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def decode(ids: list[int], tokenizer: dict) -> str:
    itos = {int(idx): ch for idx, ch in tokenizer["itos"].items()}
    return "".join(itos[idx] for idx in ids)


def save_checkpoint(
    model: BigramLanguageModel,
    config: TrainConfig,
    vocab_size: int,
    device: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": "bigram",
            "vocab_size": vocab_size,
            "train_config": asdict(config),
            "device": device,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal bigram language model.")
    parser.add_argument("--max-iters", type=int, default=TrainConfig.max_iters)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--block-size", type=int, default=TrainConfig.block_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
    )

    torch.manual_seed(config.seed)
    device = select_device()

    tokenizer = load_tokenizer(DATA_DIR / "tokenizer.json")
    vocab_size = int(tokenizer["vocab_size"])
    train_data = load_tokens(DATA_DIR / "train.bin")
    val_data = load_tokens(DATA_DIR / "val.bin")

    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"device: {device}")
    print(f"vocab_size: {vocab_size}")
    print(f"train tokens: {len(train_data)}")
    print(f"val tokens: {len(val_data)}")
    print(f"config: {asdict(config)}")

    for iter_idx in range(config.max_iters + 1):
        if iter_idx % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config, device)
            print(
                f"step {iter_idx}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train", train_data, val_data, config, device)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("loss should not be None during training")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=120)[0].tolist()
    print("sample:")
    print(decode(generated, tokenizer))

    checkpoint_path = CHECKPOINT_DIR / "bigram.pt"
    save_checkpoint(model, config, vocab_size, device, checkpoint_path)
    print(f"checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

