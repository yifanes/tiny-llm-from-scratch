# From Bigram To GPT

这篇笔记记录 Bigram 到 GPT 的核心差异。

## Bigram Is Too Small

Bigram 只看当前 token。

例如：

```text
我 爱 中
```

当 Bigram 要预测下一个 token 时，它只看最后一个 token：

```text
中 -> ?
```

它不知道前面是：

```text
我 爱 中
位 于 中
学 习 中
```

这些上下文完全不同，但 Bigram 看到的都是 `中`。

所以 Bigram 只能学习：

```text
当前 token -> 下一个 token
```

它学不到：

```text
一整段上下文 -> 下一个 token
```

## What GPT Adds

GPT 要解决的是：

```text
不只看当前 token，而是看前面一整段上下文。
```

Bigram：

```text
当前 token -> 下一个 token
```

GPT：

```text
前面一段 token -> 下一个 token
```

训练流程基本不变：

```text
get_batch()
x/y
model(x, y)
loss
loss.backward()
optimizer.step()
```

变化的是模型内部。

## GPT Structure

一个最小 GPT 大概是：

```text
token ids
  -> token embedding
  +  position embedding
  -> Transformer block
  -> Transformer block
  -> LayerNorm
  -> lm_head
  -> logits
```

## Token Embedding

token id 是离散数字。

例如：

```text
我 = 123
爱 = 88
中 = 45
```

这些数字本身没有语义大小关系。

token embedding 把 token id 映射成向量：

```text
我 -> [0.12, -0.03, 0.88, ...]
爱 -> [0.51, 0.20, -0.11, ...]
中 -> [-0.04, 0.73, 0.31, ...]
```

作用是：

```text
把离散 token id 转成可计算的向量表示
```

## Position Embedding

Transformer 默认不知道 token 的顺序。

例如：

```text
我 爱 你
你 爱 我
```

如果没有位置信息，模型很难区分它们。

position embedding 给每个位置一个向量：

```text
位置 0 -> vector
位置 1 -> vector
位置 2 -> vector
```

最终输入是：

```text
token embedding + position embedding
```

也就是：

```text
这个 token 是什么 + 这个 token 在哪里
```

## Self-Attention

self-attention 让每个位置从上下文中读取信息。

例如：

```text
什么 是 tokenizer ？
回答 ：
```

当模型处理 `回答` 这个位置时，它可以关注：

```text
什么
tokenizer
？
```

attention 会给上下文里不同 token 分配不同权重。

粗略理解：

```text
回答 关注 什么: 0.2
回答 关注 tokenizer: 0.6
回答 关注 ？: 0.1
回答 关注 回答: 0.1
```

这样当前 token 的向量就混入了上下文信息。

## Causal Mask

GPT 训练时不能偷看未来。

如果：

```text
x = 我 爱 中
y = 爱 中 国
```

当模型在位置 `我` 预测 `爱` 时，不能看到后面的：

```text
爱 中
```

causal mask 规定：

```text
位置 0 只能看位置 0
位置 1 只能看位置 0,1
位置 2 只能看位置 0,1,2
```

这保证模型只能根据当前和过去预测未来。

## MLP, LayerNorm, Residual

可以先这样理解：

```text
attention: 从上下文里收集信息
MLP: 对收集后的信息进行加工
LayerNorm: 稳定训练
Residual: 保留原信息并帮助梯度传播
```

Transformer block 常见形式：

```text
x = x + attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

## lm_head

最后模型需要输出下一个 token 的分数。

如果词表大小是 380，那么每个位置都要输出 380 个 logits。

`lm_head` 做的就是：

```text
hidden vector -> vocab_size 个 logits
```

## Summary

Bigram：

```text
token id
  -> 查表
  -> 下一个 token logits
```

GPT：

```text
token ids
  -> token embedding + position embedding
  -> self-attention 读取上下文
  -> MLP 加工信息
  -> 多层重复
  -> lm_head
  -> 下一个 token logits
```

一句话：

```text
Bigram 只记当前 token 后面常跟什么。
GPT 用 attention 把一段上下文融合成表示，再预测下一个 token。
```

