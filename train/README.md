# Training

这里实现 PyTorch 版本 tiny GPT。

第一版目标：

- decoder-only Transformer
- causal self-attention
- cross entropy loss
- AdamW optimizer
- checkpoint 保存

## Minimal Training Loop

先用 `BigramLanguageModel` 跑通最小训练闭环：

```bash
uv sync --group train
uv run python train/bigram.py
```

它会完成：

```text
train.bin/val.bin
  -> get_batch()
  -> x/y
  -> model(x)
  -> cross entropy loss
  -> loss.backward()
  -> optimizer.step()
  -> checkpoints/bigram.pt
```

Bigram 模型只根据当前 token 预测下一个 token，不具备 Transformer 的上下文建模能力。它的价值是帮助我们先看懂训练程序。

