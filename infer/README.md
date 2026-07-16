# Inference

这里实现不依赖 PyTorch 的 NumPy 推理引擎。

第一版使用 Numpy，不直接调用 PyTorch 模型。

目标：

- 加载 config
- 加载 tokenizer
- 加载权重
- 实现 forward
- 实现 generate
- 后续加入 KV cache

## Usage

```python
import numpy as np

from infer import load_model

model = load_model("exports")
token_ids = np.asarray([model.tokenizer.encode("Tiny GPT")])
logits = model.forward(token_ids)  # (batch, time, vocab_size)

generated_ids = model.generate(token_ids, max_new_tokens=20)
text = model.tokenizer.decode(generated_ids[0])
```

也可以直接生成文本：

```python
text = model.generate_text("Tiny GPT", max_new_tokens=20)
```

`temperature=0`（默认）使用 greedy decoding。传入正温度后可以组合 top-k 和
top-p sampling，并注入 NumPy Generator 复现实验：

```python
rng = np.random.default_rng(42)
text = model.generate_text(
    "Tiny GPT",
    max_new_tokens=20,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    rng=rng,
)
```

`generate` 默认启用 KV cache。首次前向会缓存每层每个注意力头的 K/V，后续只计算
新 token。输入长度达到 `block_size` 后，模型会保留最近的上下文并重建缓存：模型使用
learned absolute position embeddings，简单丢弃最旧 K/V 会改变位置语义，无法与无缓存
推理精确对齐。调试时可传入 `use_kv_cache=False` 关闭缓存。

实现位于 `numpy_engine.py`，包含：

- `model.bin` 的格式、长度和 checksum 校验
- 字符级 tokenizer 的 encode/decode
- Linear、LayerNorm、softmax 和 causal self-attention
- Multi-head attention、FeedForward、Transformer Block 和完整 forward
- temperature、top-k、top-p 与可复现随机采样
- 逐层逐头 KV cache 和超出窗口后的精确重建
- greedy 或 sampled 逐 token generation
