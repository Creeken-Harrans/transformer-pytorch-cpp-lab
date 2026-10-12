# Transformer PyTorch / C++ Lab

[![Verify both implementations](https://github.com/Creeken-Harrans/transformer-pytorch-cpp-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Creeken-Harrans/transformer-pytorch-cpp-lab/actions/workflows/ci.yml)

用两套独立实现读懂完整的 **encoder–decoder Transformer**：Python 用 PyTorch 表达数学，C++17 用标准库手写张量算子和反向传播。两边共享模型结构、参数与输入文件，可以直接对照前向结果、每个参数的梯度和一次 SGD 更新。

这里保留最值得动手理解的部分：多头注意力、因果与 padding mask、位置编码、Post-LN 残差块、teacher forcing、交叉熵、训练和自回归生成。代码注释解释张量形状、数学公式及容易出错的边界。无需下载数据集。

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

架构依据 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。本仓库选择独立的源/目标嵌入与输出投影，并省略 dropout、权重绑定、label smoothing、Adam 与学习率调度，方便追踪完整的计算和梯度；训练配方与原论文不同。

## 先运行，再读代码

准备好 [uv](https://docs.astral.sh/uv/)、CMake 和支持 C++17 的编译器。在仓库根目录执行：

```bash
git clone https://github.com/Creeken-Harrans/transformer-pytorch-cpp-lab.git
cd transformer-pytorch-cpp-lab
make check
```

这会运行格式与静态检查、Python/C++ 独立测试，以及跨语言数值验证。第一次运行需要安装 Python 依赖；后续使用提交的 `python/uv.lock`。

也可以直接运行跨语言验证器，它会自动构建 C++：

```bash
uv run --locked --project python python verify.py
```

验证器使用相同权重和输入，比较 encoder/decoder 输出、logits、loss、所有参数梯度和一次 SGD 后的权重。它还用 PyTorch 官方 encoder/decoder 层检查前向组合，抽样做完整模型有限差分，并检查多步训练、checkpoint 互读、生成与错误输入。官方层只用于验证；教学实现自行组装张量原语。验收容差与检查思路见 [学习指南](LEARNING.md#7-怎样判断实现可信)。

## 一次完整的训练与生成

运行固定小批次的复制任务演示：

```bash
make demo
```

演示分别训练两套实现，再交叉读取 checkpoint 进行生成，最后断言两端都精确复制每个 token、遇到 EOS 停止且 loss 小于 0.1。它用于观察小样本过拟合、loss 与 greedy decode 的关系；评估未见样本的泛化需要另建训练集和测试集。本次交付的实测环境与数值见 [VALIDATION.md](VALIDATION.md)。

以下命令可拆开观察每一步。首先进入 Python 子目录，导出包含权重、配置和数据的 case，再做前向、反向与一次更新：

```bash
cd python
uv sync --locked
uv run --locked transformer-lab export ../case.tlab --task copy --seed 7 --batch-size 2 --sequence-length 4 --learning-rate 0.1
uv run --locked transformer-lab inspect ../case.tlab ../python-result.tlab
uv run --locked transformer-lab train ../case.tlab ../python-checkpoint.tlab --steps 300
uv run --locked transformer-lab generate ../python-checkpoint.tlab --bos 1 --eos 2 --max-new-tokens 8
cd ..
```

接着在根目录构建 C++，读取同一份初始 case：

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
cpp/build/transformer_cpp inspect case.tlab cpp-result.tlab
cpp/build/transformer_cpp train case.tlab cpp-checkpoint.tlab --steps 300
cpp/build/transformer_cpp generate cpp-checkpoint.tlab --bos 1 --eos 2 --max-new-tokens 8
```

Python 可以加载 `cpp-checkpoint.tlab`，C++ 也可以加载 `python-checkpoint.tlab`。`inspect` 文件保存 `logits`、`loss`、中间输出、`grad.*` 和 `updated.*`；它是数值检查结果，训练与生成应读取原始 case 或训练保存的 checkpoint。

## 按问题选择入口

| 想理解的问题 | 从哪里开始 |
|---|---|
| 每一步的形状、公式、mask 和训练含义 | [LEARNING.md](LEARNING.md) |
| 如何从张量原语组装 Transformer | [Python 模型](python/src/transformer_lab/model.py) |
| 如何给算子写 backward，并处理共享节点 | [C++ 张量与 autograd](cpp/src/tensor.cpp) |
| C++ 如何组装 encoder、decoder 和训练 | [C++ 模型](cpp/src/model.cpp) |
| 两种语言如何交换完全相同的实验 | [数值契约](DESIGN.md) 与两边 `format` 模块 |
| 如何复现和定位跨语言差异 | [verify.py](verify.py) |
| 如何改进代码和补充实验 | [CONTRIBUTING.md](CONTRIBUTING.md) |

完整指令也可在 [Python 运行说明](python/README.md) 和 [C++ 运行说明](cpp/README.md) 中查看。

## 教学边界

模型支持 batch、多层、不同源/目标长度、teacher forcing 和 greedy 自回归生成。为使数学与代码逐步对应，仓库使用小模型、CPU、float64 和简单 SGD；C++ 的直接循环适合阅读与调试。

tokenizer、真实翻译数据管线、beam search、KV cache、混合精度、CUDA kernel、分布式训练及性能优化留作后续扩展。复制任务的训练 loss 下降只说明该固定批次被拟合，不能据此推断翻译能力或未见序列的泛化。

采用 [MIT License](LICENSE)。
