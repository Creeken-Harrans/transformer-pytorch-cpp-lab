# PyTorch 后端：从公式到完整 Transformer

这个目录可以独立运行完整 encoder–decoder Transformer。PyTorch 负责张量与
autograd；attention、LayerNorm、位置编码、mask 和 cross-entropy 都在源码中
展开。模型采用 Post-LN、float64、ReLU、dropout=0；优化器是普通 SGD。

## 环境与首次运行

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在本目录运行：

```bash
uv sync --locked
uv run transformer-lab --help
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`.python-version` 固定 Python 3.12，uv 可自动获取解释器。`uv.lock` 锁定 PyTorch
2.8.0、NumPy 2.3.2、pytest 8.4.1、ruff 0.12.9 及间接依赖。普通依赖来自官方
PyPI，PyTorch 使用其官方 CPU wheel 索引；不需要 NVIDIA GPU 或 CUDA toolkit。
NumPy 用于满足 PyTorch 的数组互操作初始化，核心公式不调用 NumPy。

## 四个命令完成一次实验

```bash
# 两条变长序列；PAD=0，BOS=1，EOS=2；源包含 EOS，目标使用 teacher forcing。
uv run transformer-lab export copy.tlab --task copy --seed 7 \
  --vocab-size 8 --sequence-length 2 --learning-rate 0.2

# 从原始权重计算输出、梯度，并记录一次 SGD 后的全部权重。
uv run transformer-lab inspect copy.tlab python-result.tlab

# 在同一个小 batch 上反复训练，打印 initial_loss、final_loss 和步数。
uv run transformer-lab train copy.tlab trained.tlab --steps 250

# 每行对应一条源序列，打印含 BOS/EOS 的整数 token；提前结束的样本补 PAD。
uv run transformer-lab generate trained.tlab --bos 1 --eos 2 --max-new-tokens 8
```

`--task reverse` 将内容逆序后作为目标。`--sequence-length` 是最长**内容**长度；
实际 src/tgt/labels 长度均为它加一，以容纳 BOS 或 EOS。batch 内各条内容长度
依次减小，因此默认 batch=2、内容长度=4 的 case 包含 PAD。
内容 token 从词表中排除 PAD、BOS、EOS 后采样。特殊 token 可以修改，例如：

```bash
uv run transformer-lab export custom.tlab --pad-id 2 --bos 0 --eos 1
uv run transformer-lab generate custom.tlab --bos 0 --eos 1 --max-new-tokens 8
```

三个特殊 ID 必须互不相同。checkpoint 保留原始配置、src、tgt、labels、学习率，
因此可以继续 `train`，也可以直接交给 C++ `train` 或 `generate`。这些实验测量
模型是否能记住一个 batch；研究泛化能力时需另设训练、验证和测试数据。
`train --steps 0` 仅校验并另存 checkpoint，权重保持不变；步数不能为负。

`export` 支持全部配置参数：`--vocab-size`、`--d-model`、`--num-heads`、
`--d-ff`、`--num-encoder-layers`、`--num-decoder-layers`、`--max-len`、
`--pad-id`、`--eps`。默认值见 `TransformerConfig`；CLI 用单线程运行小矩阵，
库 API 不修改全局线程配置。

## 推荐阅读顺序

| 文件 | 先观察的公式或约定 |
| --- | --- |
| `src/transformer_lab/model.py` | 位置编码 → `scaled_dot_product_attention` → `MultiHeadAttention` → LayerNorm → encoder/decoder → loss |
| `src/transformer_lab/cli.py` | `toy_batch` 的目标右移、`inspect_case` 的 backward/SGD、`train_case` 的梯度清空 |
| `src/transformer_lab/format.py` | double 的 17 位有效数字、row-major 排列、严格 shape 与 token 校验 |
| `tests/test_model.py` | 手算 attention、mask 隔离、gradcheck、全模型中央差分、训练和 EOS/PAD 行为 |
| `tests/test_format_cli.py` | 格式错误输入、精确数值往返、四个命令串联 |

设 B=batch、S=源长度、T=目标长度、D=d_model、H=头数、K=D/H。embedding 输出
`[B,L,D]`，乘 `sqrt(D)` 后加位置编码。线性投影后拆头为 `[B,H,L,K]`，注意力
分数是 `[B,H,T,S]`；softmax 沿 S 轴，输出拼回 `[B,T,D]`。`nn.Linear` 权重
存成 `[输出,输入]`，所以代码计算的是 `X @ W.T + b`。

mask 使用 **True=允许**。source mask `[B,1,1,S]` 屏蔽源 PAD key，同时用于
encoder self-attention 与 decoder cross-attention。target mask 是下三角因果
矩阵与非 PAD key 的交集。PAD query 可以有输出；PAD labels 从 loss 排除。
全屏蔽行没有合法概率分布，因此抛出 `ValueError`。

## 直接使用 Python API

```python
import torch
from transformer_lab.model import Transformer, TransformerConfig, masked_cross_entropy
from transformer_lab.format import make_case, write_bundle, read_bundle, load_case

torch.manual_seed(7)
model = Transformer(TransformerConfig(vocab_size=8))
src = torch.tensor([[3, 4, 2, 0]])
tgt = torch.tensor([[1, 3, 4]])     # BOS, y1, y2
labels = torch.tensor([[3, 4, 2]])  # y1, y2, EOS

logits, trace = model.forward_with_trace(src, tgt)
loss = masked_cross_entropy(logits, labels, pad_id=0)
loss.backward()
print(logits.shape, loss.item())   # [1,3,8]
print(trace["encoder.0.output"].shape)
print(trace["decoder_output"].shape)
write_bundle("case.tlab", make_case(model, src, tgt, labels, learning_rate=0.1))
loaded, src, tgt, labels, lr = load_case(read_bundle("case.tlab"))
tokens = loaded.greedy_decode(src, bos_id=1, eos_id=2, max_new_tokens=8)
```

`forward_with_trace` 包含 `encoder.{i}.output`、`decoder.{i}.output` 和最后的
`encoder_output`、`decoder_output`，保留 autograd 图；查看时可以用 `.detach()`。
整数 token 接受 PyTorch 的整数 dtype，内部转为 long；浮点 token 即使数值为
整数也拒绝。每条源序列至少一个非 PAD，每条目标第一个 token 必须非 PAD。
生成的 `max_new_tokens` 可以是零，且包括 BOS 后的长度必须不超过 `max_len`。

## 与 C++ 对照

从仓库根目录编译 C++ 后，可把同一 case 交给两个实现：

```bash
../cpp/build/transformer_cpp inspect copy.tlab cpp-result.tlab
uv run python ../verify.py
```

验收比较每层边界、logits、有效 token 平均 loss、**全部**参数梯度和一步 SGD，
再检查 greedy token 完全一致。TLAB1 文件还拒绝重复记录、额外尾部 token、
截断、非有限数、错误 rank 和超大尺寸；reader 最多允许 10,000 条记录及
10,000,000 个元素，单文件 256 MiB，每个 token/记录名字最多 1024 UTF-8 字节，
便于处理错误输入。

公式依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)；
LayerNorm 的总体方差约定参见
[PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。
