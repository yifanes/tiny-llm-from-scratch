# tiny-llm-from-scratch

从零实现一个 tiny LLM 的完整学习项目。

目标不是训练出一个聪明的大模型，而是把大模型从数据、训练、权重导出、推理引擎到 OpenAI API 风格服务的完整链路亲手打通。

## Project Goal

在单张 8GB GPU 可承受的范围内，实现一个教学级 tiny GPT：

1. 准备训练数据
2. 训练一个小型 Transformer/GPT
3. 导出权重文件
4. 自己写推理引擎加载权重
5. 实现 tokenizer、KV cache、采样
6. 包一层 OpenAI API 风格接口
7. 用客户端发起对话

## Principles

- 完整性优先于模型效果
- 可解释优先于工程抽象
- 先跑通最小闭环，再逐步替换更真实的组件
- 每天学习和实现都同步到 GitHub

## Initial Scope

第一阶段会使用字符级 tokenizer 和小型 decoder-only GPT。

推荐起始规模：

```text
n_layer: 4
n_head: 4
n_embd: 256
block_size: 256
batch_size: 按显存调整
```

## Repository Layout

```text
data/       数据准备脚本和小样本数据说明
train/      PyTorch 训练代码
export/     权重导出代码
infer/      自研推理引擎
server/     OpenAI API 风格服务
client/     CLI 或 Web 客户端
docs/       学习日志和知识梳理
```

## Current Status

Phase 0-7 已全部跑通：字符级数据准备、Tiny GPT 训练、权重导出、纯 NumPy
推理、采样与 KV cache、OpenAI 风格 API、CLI/Web 客户端。

NumPy forward 已与 PyTorch checkpoint 对齐，当前测试模型的 logits 最大绝对误差
约为 `4.3e-6`。

## Run The Complete Chain

导出当前 checkpoint：

```bash
uv run --group train python export/export_weights.py
```

启动 API：

```bash
uv run --group server uvicorn server.app:app --reload
```

另开终端启动 CLI：

```bash
uv run python -m client.chat
```

或者启动 Web 客户端：

```bash
uv run python -m http.server 8080 --directory client
```

然后访问 `http://127.0.0.1:8080/web.html`。

## Environment

本项目使用 `uv` 管理 Python 环境和依赖。

```bash
uv sync
uv run python --version
```

更多说明见 `docs/environment.md`。

## Learning Notes

- `docs/concepts/training-data-to-samples.md`: 解释训练数据如何变成训练样本，以及模型如何通过 loss 更新权重。
- `docs/concepts/minimal-training-loop.md`: 解释第一版 Bigram 最小训练闭环。
- `docs/concepts/checkpoint-and-generation.md`: 解释 checkpoint、权重加载和生成之间的关系。
- `docs/concepts/from-bigram-to-gpt.md`: 解释 Bigram 到 GPT 的结构升级。
- `docs/concepts/tiny-gpt-training.md`: 记录第一版 Tiny GPT 的训练实现和验证结果。
