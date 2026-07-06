# From Training Data To Samples

这篇笔记记录我们对 `train.bin -> x/y -> 模型预测 -> loss -> 更新权重` 这条链路的理解。

## Core Mental Model

模型训练不是把知识写进一个数据库，而是用数据反复纠正模型的预测分布，最后把规律压缩进权重里。

可以先这样理解：

```text
模型 = 固定结构 + 一堆可训练权重

train.bin = 训练文本经过 tokenizer 后得到的一长串 token id

x = 题目
y = 标准答案

训练 = 模型做题 + loss 判分 + optimizer 更新权重
```

## What Is Inside train.bin And val.bin

`train.bin` 和 `val.bin` 里面不是文字，而是一串 token id。

例如原文：

```text
我爱中国
```

经过字符级 tokenizer 之后可能变成：

```text
[1, 2, 3, 4]
```

那么 `.bin` 文件里保存的就是这些数字的二进制形式。

区别是：

```text
train.bin: 用来训练模型，更新权重
val.bin:   用来验证模型，不更新权重
```

## What Is A Training Sample

`train.bin` 不是一个训练样本，它是训练样本的来源。

训练时会从 token 序列中切出一小段：

```python
chunk = tokens[i : i + block_size + 1]

x = chunk[:-1]
y = chunk[1:]
```

比如：

```text
chunk = 我 爱 中 国
x     = 我 爱 中
y     = 爱 中 国
```

`x` 和 `y` 来自同一段 token，只是错开了一个位置。

## Why y Is x Shifted By One

GPT 的训练目标是预测下一个 token。

如果：

```text
x = 我 爱 中
y = 爱 中 国
```

它代表三道预测题：

```text
看到 我        预测 爱
看到 我 爱     预测 中
看到 我 爱 中  预测 国
```

这就是 next token prediction。

## What Happens During One Training Step

一次训练大概是：

```text
1. 从 train.bin 里切出 block_size + 1 个 token
2. 前 block_size 个 token 作为 x
3. 后 block_size 个 token 作为 y
4. 把 x 输入模型
5. 模型预测每个位置的下一个 token
6. 用 y 作为标准答案计算 loss
7. 反向传播根据 loss 计算梯度
8. optimizer 更新模型权重
```

训练开始时，权重大多是随机的，所以模型预测基本不靠谱。

例如：

```text
输入: 我爱
正确答案: 中
模型可能预测: 吃
```

此时 loss 会比较高。训练程序会根据这个错误调整权重，让模型以后在类似上下文里更倾向于预测 `中`，而不是 `吃`。

## Important Detail: Not A Rule Table

模型不会直接记录一条规则：

```text
我爱 -> 中
```

它做的是：

```text
计算预测分布和正确答案之间的差距
根据差距微调大量权重参数
```

比如模型当前认为：

```text
中: 0.10
吃: 0.60
你: 0.20
```

正确答案是：

```text
中
```

训练会让模型提高 `中` 的概率，降低错误 token 的概率。

这个过程不是写死规则，而是调整 embedding、attention、MLP、输出层等结构里的浮点数。

## Model Structure And Weights

更准确地说：

```text
模型结构 + 权重 = 当前这个模型
```

模型结构包括：

```text
embedding
attention
MLP
LayerNorm
output layer
```

这些结构是固定的。

权重是结构里面的数字，会随着训练不断变化。

embedding 不是一开始就内置知识，它是一张可训练表：

```text
token id -> vector
```

训练开始时，embedding 里的向量大多也是随机的；训练过程中，它会和其他权重一起被更新。

## Training Is Parallel In Practice

概念上可以理解为模型逐个位置做题：

```text
看到 我        预测 爱
看到 我 爱     预测 中
看到 我 爱 中  预测 国
```

但真实训练中，Transformer 通常不是用 Python 循环逐个 token 计算。

它会一次把整个 `x` 输入模型，然后并行输出每个位置的预测结果。

所以：

```text
x = 我 爱 中
```

模型一次 forward 后，会同时得到：

```text
位置 0 的下一个 token 预测
位置 1 的下一个 token 预测
位置 2 的下一个 token 预测
```

然后一次性和 `y` 比较并计算 loss。

## Final Summary

可以把训练过程记成：

```text
train.bin 提供题库
x 是题目
y 是标准答案
模型权重负责做题
loss 负责判分
optimizer 负责改模型权重
```

重复很多次之后，模型权重会逐渐贴近训练数据中的语言模式。

