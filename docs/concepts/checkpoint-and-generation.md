# Checkpoint And Generation

这篇笔记记录 Bigram 阶段第一次把训练和生成拆开。

## Training And Generation Are Different

训练脚本负责更新权重：

```text
train/bigram.py
  -> 读取 train.bin 和 val.bin
  -> 构造 x/y
  -> 计算 loss
  -> backward
  -> optimizer.step()
  -> 保存 checkpoints/bigram.pt
```

生成脚本负责使用权重：

```text
train/sample_bigram.py
  -> 读取 tokenizer.json
  -> 读取 checkpoints/bigram.pt
  -> 重建 Bigram 模型结构
  -> 加载权重
  -> 一个 token 一个 token 生成文本
```

训练会更新模型权重。

生成不会更新模型权重。

## What Is In A Checkpoint

`checkpoints/bigram.pt` 里保存：

```text
model: 模型权重
model_type: bigram
vocab_size: 词表大小
train_config: 训练配置
device: 训练时使用的设备
```

checkpoint 不是训练数据。

checkpoint 保存的是训练后得到的模型参数。

## What Is Needed For Generation

生成文本需要三样东西：

```text
模型结构
模型权重
tokenizer
```

在当前阶段：

```text
模型结构: BigramLanguageModel 代码
模型权重: checkpoints/bigram.pt
tokenizer: data/tokenizer.json
```

如果只有权重，没有模型结构，程序不知道这些数字该怎么计算。

如果只有模型结构，没有权重，模型只是随机初始化的大脑。

如果没有 tokenizer，模型生成的 token id 无法解码回文本。

## Why Generation No Longer Needs train.bin

训练时需要 `train.bin`，因为它负责出题和给标准答案。

生成时不需要 `train.bin`，因为模型已经把学到的统计模式压缩进权重里了。

生成时流程是：

```text
初始 token
  -> 模型预测下一个 token 的概率
  -> 采样一个 token
  -> 拼回上下文
  -> 重复
  -> decode 成文本
```

Bigram 生成效果会很差，因为它只看当前 token。

这一步的价值是理解：

```text
训练产生权重
推理加载权重
tokenizer 负责 token id 和文本之间的转换
```

