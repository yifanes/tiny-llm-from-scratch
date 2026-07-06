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
class GPTConfig:
    batch_size: int = 16
    block_size: int = 64
    max_iters: int = 500
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    seed: int = 1337


class Head(nn.Module):
    def __init__(self, config: GPTConfig, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, time_steps = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(time_steps, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        _, _, vocab_size = logits.shape
        logits_for_loss = logits.view(batch_size * time_steps, vocab_size)
        targets_for_loss = targets.view(batch_size * time_steps)
        loss = F.cross_entropy(logits_for_loss, targets_for_loss)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
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


def decode(ids: list[int], tokenizer: dict) -> str:
    itos = {int(idx): ch for idx, ch in tokenizer["itos"].items()}
    return "".join(itos[idx] for idx in ids)


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: GPTConfig,
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
    model: TinyGPTLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: GPTConfig,
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


def save_checkpoint(
    model: TinyGPTLanguageModel,
    config: GPTConfig,
    vocab_size: int,
    device: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": "tiny_gpt",
            "vocab_size": vocab_size,
            "gpt_config": asdict(config),
            "device": device,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny GPT language model.")
    parser.add_argument("--max-iters", type=int, default=GPTConfig.max_iters)
    parser.add_argument("--batch-size", type=int, default=GPTConfig.batch_size)
    parser.add_argument("--block-size", type=int, default=GPTConfig.block_size)
    parser.add_argument("--learning-rate", type=float, default=GPTConfig.learning_rate)
    parser.add_argument("--n-embd", type=int, default=GPTConfig.n_embd)
    parser.add_argument("--n-head", type=int, default=GPTConfig.n_head)
    parser.add_argument("--n-layer", type=int, default=GPTConfig.n_layer)
    parser.add_argument("--dropout", type=float, default=GPTConfig.dropout)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GPTConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    if config.n_embd % config.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    torch.manual_seed(config.seed)
    device = select_device()

    tokenizer = load_tokenizer(DATA_DIR / "tokenizer.json")
    vocab_size = int(tokenizer["vocab_size"])
    train_data = load_tokens(DATA_DIR / "train.bin")
    val_data = load_tokens(DATA_DIR / "val.bin")

    model = TinyGPTLanguageModel(vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    param_count = sum(param.numel() for param in model.parameters())

    print(f"device: {device}")
    print(f"vocab_size: {vocab_size}")
    print(f"train tokens: {len(train_data)}")
    print(f"val tokens: {len(val_data)}")
    print(f"parameters: {param_count:,}")
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
    generated = model.generate(context, max_new_tokens=200)[0].tolist()
    print("sample:")
    print(decode(generated, tokenizer))

    checkpoint_path = CHECKPOINT_DIR / "tiny_gpt.pt"
    save_checkpoint(model, config, vocab_size, device, checkpoint_path)
    print(f"checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
