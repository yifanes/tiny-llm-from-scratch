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

## Generate From Checkpoint

训练后可以单独加载 checkpoint 生成文本：

```bash
uv run python train/sample_bigram.py --max-new-tokens 200
```

这一步不使用 `train.bin` 更新权重，只使用：

```text
BigramLanguageModel
checkpoints/bigram.pt
data/tokenizer.json
```

## Tiny GPT

训练一个最小 GPT：

```bash
uv run python train/tiny_gpt.py --max-iters 300
```

为了快速验证，也可以用更小配置：

```bash
uv run python train/tiny_gpt.py \
  --max-iters 100 \
  --batch-size 8 \
  --block-size 32 \
  --n-embd 64 \
  --n-head 4 \
  --n-layer 2
```

Tiny GPT 相比 Bigram 增加了：

```text
token embedding
position embedding
causal self-attention
MLP
LayerNorm
residual connection
lm_head
```
