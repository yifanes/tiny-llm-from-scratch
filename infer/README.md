# Inference

这里实现自研推理引擎。

第一版使用 Numpy，不直接调用 PyTorch 模型。

目标：

- 加载 config
- 加载 tokenizer
- 加载权重
- 实现 forward
- 实现 generate
- 后续加入 KV cache

