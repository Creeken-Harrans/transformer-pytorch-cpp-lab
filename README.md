# Transformer PyTorch / C++ Lab

[![Verify both implementations](https://github.com/Creeken-Harrans/transformer-pytorch-cpp-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Creeken-Harrans/transformer-pytorch-cpp-lab/actions/workflows/ci.yml)

用两套独立实现读懂完整的 **encoder–decoder Transformer**：Python 用 PyTorch 表达数学，C++17 用标准库手写张量算子和反向传播。两边读取相同的权重与输入，逐项核对前向值、全部参数梯度和一次 SGD 更新。

沿着一个无需下载数据集的复制任务，可以学会：

- 跟踪 token、嵌入、Q/K/V、多头布局、encoder memory 与 logits 的形状。
- 解释 teacher forcing、因果与 padding mask、Post-LN 残差块和交叉熵如何组成训练。
- 从局部 backward 追到共享计算图的梯度累加，再观察 SGD 与逐 token 生成。

## 先运行，再读代码

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)、Make、CMake 3.16 或更新版本，以及支持 C++17 的编译器（如 GCC 或 Clang）。uv 会按 `python/.python-version` 获取 Python 3.12 并使用提交的 `python/uv.lock` 安装依赖；CMake 和编译器需自行安装。首次安装依赖需要联网。

在仓库根目录执行：

```bash
git clone https://github.com/Creeken-Harrans/transformer-pytorch-cpp-lab.git
cd transformer-pytorch-cpp-lab
make help   # 查看命令与阅读入口
make demo   # 两端训练复制任务，交叉加载 checkpoint 并生成
make check  # 静态检查、独立测试与跨语言数值验证
```

`make demo` 训练一个固定小批次，断言两端精确复制每个内容 token、遇到 EOS 停止且 loss 小于 0.1。`make check` 比较各层输出、logits、loss、所有梯度和一步更新，并检查多步训练、checkpoint 互读、生成和错误输入。实测环境与数值见 [VALIDATION.md](VALIDATION.md)，容差与验证思路见[学习指南第 7 节](LEARNING.md#7-怎样判断实现可信)。

想先看清一次训练的输入、形状与更新，再读长篇推导，可运行：

```bash
make walkthrough
```

这个[单步例子](python/examples/walkthrough.py)打印实际 token、允许矩阵、attention 概率、各层形状、有效 label 数、loss、梯度范数和一次 SGD 后的 loss。继续阅读 [LEARNING.md](LEARNING.md) 的第 1–3 节，再按第 8 节对照两套代码。

## 两套实现，一份数学契约

| | [Python](python/) | [C++](cpp/) |
|---|---|---|
| 运行环境 | Python 3.12、PyTorch、uv | C++17、CMake、标准库 |
| 张量运算 | PyTorch 张量原语 | 手写 CPU 循环与张量布局 |
| 反向传播 | PyTorch autograd | 手写计算图与 reverse-mode autograd |
| 模型 | 多层 encoder + decoder，原论文风格 Post-LN | 相同结构与参数名字 |
| 数值与训练 | float64、dropout=0、无 momentum 的 SGD | double、相同 loss 与更新公式 |
| 模型交换 | 文本格式 `TLAB1` | 同一格式，可读取 Python 导出的权重 |
| 推理 | greedy decode、EOS 停止、batch 中已结束样本补 PAD | 相同规则 |

架构依据 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。仓库使用独立的源/目标嵌入与输出投影，省略 dropout、权重绑定、label smoothing、Adam 与学习率调度，方便逐步追踪计算和梯度；训练配方与原论文不同。共享约定详见 [DESIGN.md](DESIGN.md)。

## 把公式定位到代码

| 概念 | Python 符号 | C++ 符号 |
|---|---|---|
| 嵌入与位置编码 | [`Transformer._embed`、`sinusoidal_position_encoding`](python/src/transformer_lab/model.py) | [`Transformer::embed`](cpp/src/model.cpp)、[`embedding`](cpp/src/tensor.cpp) |
| Q/K/V 与多头注意力 | [`MultiHeadAttention`、`scaled_dot_product_attention`](python/src/transformer_lab/model.py) | [`Transformer::multi_head`](cpp/src/model.cpp)、[`attention`](cpp/src/tensor.cpp) |
| 因果与 PAD mask | [`Transformer._encode`、`Transformer._decode`](python/src/transformer_lab/model.py) | [`Transformer::forward`](cpp/src/model.cpp) |
| Post-LN、残差与 FFN | [`EncoderLayer`、`DecoderLayer`、`ManualLayerNorm`、`FeedForward`](python/src/transformer_lab/model.py) | [`Transformer::forward`、`norm`、`ffn`](cpp/src/model.cpp)、[`add`、`layer_norm`](cpp/src/tensor.cpp) |
| 交叉熵与 backward | [`masked_cross_entropy`](python/src/transformer_lab/model.py)、PyTorch `loss.backward()` | [`cross_entropy`、`Tensor::backward`](cpp/src/tensor.cpp) |
| SGD 与生成 | [`train_case`](python/src/transformer_lab/cli.py)、[`Transformer.greedy_decode`](python/src/transformer_lab/model.py) | [`Transformer::sgd`、`Transformer::generate`](cpp/src/model.cpp) |
| 权重与 case 交换 | [`make_case`、`load_case`](python/src/transformer_lab/format.py) | [`make_case`、`load_case`](cpp/src/format.cpp) |

## 拆开一次实验

以下命令都在根目录执行。导出同一份初始 case，分别检查、训练并生成：

```bash
mkdir -p .artifacts
uv run --locked --project python transformer-lab export .artifacts/case.tlab \
  --task copy --seed 7 --batch-size 2 --sequence-length 4 --learning-rate 0.1
uv run --locked --project python transformer-lab inspect \
  .artifacts/case.tlab .artifacts/python-result.tlab
uv run --locked --project python transformer-lab train \
  .artifacts/case.tlab .artifacts/python-checkpoint.tlab --steps 300

make build
cpp/build/transformer_cpp inspect .artifacts/case.tlab .artifacts/cpp-result.tlab
cpp/build/transformer_cpp train .artifacts/case.tlab .artifacts/cpp-checkpoint.tlab --steps 300

# 交叉读取训练后的权重
uv run --locked --project python transformer-lab generate \
  .artifacts/cpp-checkpoint.tlab --bos 1 --eos 2 --max-new-tokens 8
cpp/build/transformer_cpp generate .artifacts/python-checkpoint.tlab \
  --bos 1 --eos 2 --max-new-tokens 8
```

`inspect` 保存中间输出、`logits`、`loss`、`grad.*` 和 `updated.*`，用于逐项核对。训练与生成读取原始 case 或 `train` 保存的 checkpoint。全部参数与指令见 [Python 运行说明](python/README.md) 和 [C++ 运行说明](cpp/README.md)。

直接运行验证器时，它会自动构建 C++，默认显示简洁的检查表；`--json` 输出完整报告，`--json-output` 另存报告文件：

```bash
uv run --locked --project python python verify.py
uv run --locked --project python python verify.py --json-output .artifacts/validation.json
```

## 按问题选择入口

| 想理解的问题 | 从哪里开始 |
|---|---|
| 每一步的形状、公式、mask 和训练含义 | [LEARNING.md](LEARNING.md) |
| C++ 的局部 backward 与共享节点 | [张量算子](cpp/src/tensor.cpp) 与 [AUTOGRAD.md](cpp/AUTOGRAD.md) |
| 两种语言如何交换完全相同的实验 | [DESIGN.md](DESIGN.md) 与两边 `format` 模块 |
| 如何复现和定位跨语言差异 | [verify.py](verify.py) |
| 如何改进代码和补充实验 | [CONTRIBUTING.md](CONTRIBUTING.md) |

## 教学边界

模型支持 batch、多层、不同源/目标长度、teacher forcing 和 greedy 自回归生成。小模型、CPU、float64 与简单 SGD 便于逐值检查；C++ 的直接循环适合阅读和调试。复制任务演示用于观察固定批次的拟合，翻译能力和未见序列的泛化需用独立数据评估。

tokenizer、真实翻译数据管线、beam search、KV cache、混合精度、CUDA kernel、分布式训练及性能优化留作后续扩展。

采用 [MIT License](LICENSE)。
