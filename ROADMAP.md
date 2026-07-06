# Roadmap

## Phase 0: Project Setup

- [x] 创建 Git 仓库
- [x] 创建公开 GitHub 仓库
- [x] 写清楚学习目标、边界和目录结构

## Phase 1: Data And Tokenizer

- [x] 准备最小训练语料
- [x] 实现字符级 tokenizer
- [x] 将文本转换为 token ids
- [x] 构造训练集和验证集

## Phase 2: Training Tiny GPT

- [x] 实现最小训练闭环 Bigram 模型
- [x] 实现 token embedding 和 position embedding
- [x] 实现 causal self-attention
- [x] 实现 MLP、LayerNorm、Residual
- [x] 实现训练循环
- [x] 保存 PyTorch checkpoint
- [x] 用 PyTorch 版本生成文本

## Phase 3: Weight Export

- [ ] 固化 config.json
- [ ] 固化 tokenizer.json
- [ ] 导出模型权重
- [ ] 记录每个权重 tensor 的 shape 和含义

## Phase 4: Inference Engine

- [ ] 用 Numpy 加载权重
- [ ] 实现 embedding lookup
- [ ] 实现 LayerNorm
- [ ] 实现 matmul、attention、MLP
- [ ] 对齐 PyTorch forward 输出
- [ ] 实现逐 token 生成

## Phase 5: Generation Features

- [ ] 实现 temperature
- [ ] 实现 top-k sampling
- [ ] 实现 top-p sampling
- [ ] 实现 KV cache

## Phase 6: OpenAI-Compatible API

- [ ] 实现 `POST /v1/chat/completions`
- [ ] 支持 `messages`
- [ ] 支持 `temperature`
- [ ] 支持 `max_tokens`
- [ ] 支持非流式响应
- [ ] 支持流式响应

## Phase 7: Client

- [ ] 实现 CLI chat
- [ ] 实现简单 Web chat
