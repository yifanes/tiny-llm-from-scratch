# Data

这里放训练数据准备脚本和小规模教学语料。

第一阶段只需要纯文本文件，例如：

```text
data/raw.txt
```

然后通过脚本生成：

```text
data/train.bin
data/val.bin
```

## v0 Dataset

`raw.txt` 是第一版手写教学语料，特点是：

- 小规模
- 中文为主
- 主题集中在大模型学习
- 混合普通短文、问答格式、用户/助手格式
- 目标是让 tiny GPT 学到文本风格和固定格式
