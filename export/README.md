# Export

这里将 PyTorch checkpoint 导出为自研 NumPy 推理引擎可直接读取的文件。

运行：

```bash
uv run --group train python export/export_weights.py
```

默认读取：

```text
checkpoints/tiny_gpt.pt
data/tokenizer.json
```

并生成：

```text
exports/config.json
exports/tokenizer.json
exports/model.bin
```

## Files

- `config.json`: 推理所需的模型结构、参数数量和权重格式。
- `tokenizer.json`: 训练时使用的字符级 tokenizer 原样副本。
- `model.bin`: 自描述的 little-endian float32 权重文件。

`model.bin` 的布局：

```text
8 bytes       magic: TLLMWGT1
8 bytes       little-endian unsigned JSON header length
N bytes       UTF-8 JSON header
remaining     contiguous tensor data
```

JSON header 为每个 Tensor 记录：

```text
name
shape
dtype
offset (relative to the start of tensor data)
nbytes
sha256
description
```

导出脚本默认会重新读取 `model.bin`，校验所有 checksum，并逐 Tensor 与 checkpoint
做精确比较。因果遮罩 `tril` 是可由 `block_size` 重建的非训练 buffer，不会写入权重文件。

自定义输入输出路径：

```bash
uv run --group train python export/export_weights.py \
  --checkpoint checkpoints/tiny_gpt.pt \
  --tokenizer data/tokenizer.json \
  --output-dir exports
```
