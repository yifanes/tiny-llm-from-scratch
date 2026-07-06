# Tiny GPT Training

这篇笔记记录第一版 Tiny GPT 训练。

## What Changed From Bigram

Bigram 的模型内部只有：

```text
当前 token id -> 下一个 token logits
```

Tiny GPT 的模型内部变成：

```text
token ids
  -> token embedding
  +  position embedding
  -> Transformer blocks
  -> LayerNorm
  -> lm_head
  -> logits
```

外层训练流程基本不变：

```text
get_batch()
model(x, y)
loss.backward()
optimizer.step()
```

变化的是 `model(x, y)` 里面的计算更强。

## Implemented Components

`train/tiny_gpt.py` 已实现：

```text
token embedding
position embedding
single attention head
multi-head attention
causal mask
feed-forward MLP
LayerNorm
residual connection
lm_head
generate
checkpoint save
```

## First Verification Run

命令：

```bash
uv run python train/tiny_gpt.py \
  --max-iters 100 \
  --batch-size 8 \
  --block-size 32 \
  --n-embd 64 \
  --n-head 4 \
  --n-layer 2
```

结果：

```text
device: mps
vocab_size: 380
train tokens: 2709
val tokens: 301
parameters: 150,780
step 0: train loss 6.0771, val loss 6.1207
step 100: train loss 4.7696, val loss 5.4344
checkpoint: checkpoints/tiny_gpt.pt
```

loss 下降说明 Tiny GPT 的训练闭环已经跑通。

生成文本仍然不稳定，主要原因是：

```text
数据很小
训练步数很少
模型也很小
字符级 tokenizer 难度更高
```

但这一步已经完成了从 Bigram 到 Transformer/GPT 架构的升级。
