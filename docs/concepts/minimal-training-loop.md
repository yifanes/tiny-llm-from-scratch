# Minimal Training Loop

这篇笔记记录第一版最小训练闭环：`BigramLanguageModel`。

它不是 GPT，也没有 attention。它的目标是先跑通训练程序：

```text
train.bin
  -> get_batch()
  -> x/y
  -> model(x)
  -> loss
  -> backward
  -> optimizer.step()
  -> checkpoint
```

## Why Bigram First

Bigram 模型只做一件事：

```text
根据当前 token，预测下一个 token
```

它不理解长上下文。

例如：

```text
看到 我，预测下一个 token
看到 爱，预测下一个 token
看到 中，预测下一个 token
```

它不会真正利用：

```text
我 爱 中
```

这个完整上下文。

所以它生成的文本会很乱，但这不影响它作为训练闭环 demo 的价值。

## What get_batch Does

`get_batch()` 从 `train.bin` 或 `val.bin` 中随机切出多组 `x/y`。

如果：

```text
tokens = [10, 20, 30, 40, 50]
block_size = 3
```

那么一次样本可以是：

```text
x = [10, 20, 30]
y = [20, 30, 40]
```

多个样本会组成 batch：

```text
x.shape = [batch_size, block_size]
y.shape = [batch_size, block_size]
```

## What The Model Learns

Bigram 模型内部只有一张 embedding 表：

```text
token_embedding_table: [vocab_size, vocab_size]
```

输入 token id 后，它查表得到这个 token 对下一个 token 的预测分数。

也就是说：

```text
当前 token id -> 对所有可能下一个 token 的 logits
```

## One Training Step

一次训练步骤是：

```text
1. get_batch("train") 得到 xb 和 yb
2. model(xb, yb) 得到 logits 和 loss
3. optimizer.zero_grad()
4. loss.backward()
5. optimizer.step()
```

其中：

```text
loss.backward()
```

负责根据 loss 计算梯度。

```text
optimizer.step()
```

负责根据梯度更新模型权重。

## First Run Result

第一次最小验证命令：

```bash
uv run python train/bigram.py --max-iters 100 --batch-size 16 --block-size 32
```

结果：

```text
device: mps
vocab_size: 380
train tokens: 2709
val tokens: 301
step 0: train loss 6.3865, val loss 6.5237
step 100: train loss 5.0200, val loss 5.8288
```

loss 下降说明：

```text
模型权重确实被训练数据纠正了
训练闭环是通的
```

生成文本仍然混乱，这是预期结果。原因是 Bigram 模型太弱，只看当前 token，不看完整上下文。

