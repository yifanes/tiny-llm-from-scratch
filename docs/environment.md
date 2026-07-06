# Environment

本项目使用 `uv` 管理 Python 版本、虚拟环境和依赖。

## Setup

创建或同步虚拟环境：

```bash
uv sync
```

安装训练依赖：

```bash
uv sync --group train
```

安装 API 服务依赖：

```bash
uv sync --group server
```

安装全部依赖：

```bash
uv sync --all-groups
```

## Run Commands

优先使用 `uv run` 执行项目命令：

```bash
uv run python --version
uv run python data/prepare.py
```

## Dependency Strategy

第一阶段先固定最小依赖：

- `numpy`: 数据文件、推理引擎和数值计算
- `pytest`: 后续写最小测试
- `torch`: 训练阶段使用
- `fastapi` / `uvicorn`: API 服务阶段使用

## Verified Baseline

当前已验证：

```text
Python 3.12.11
numpy 2.5.1
torch 2.12.1
torch.backends.mps.is_available() = True
```

验证命令：

```bash
uv run python --version
uv run python -c "import numpy as np; print(np.__version__)"
uv run python -c "import torch; print(torch.__version__)"
```
