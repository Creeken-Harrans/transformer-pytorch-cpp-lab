# C++17 Transformer：可训练的数学实现

本目录仅依赖 C++17 标准库、CMake 和编译器。`double` 张量、矩阵乘法、注意力、
LayerNorm、交叉熵及反向传播均手写。模型是原论文风格的 Post-LN encoder–decoder，
支持 batch、多个 head、多层、PAD mask、teacher forcing、SGD 和 greedy 生成。

先读根目录 [DESIGN.md](../DESIGN.md) 了解两种实现共享的约定，再读
[AUTOGRAD.md](AUTOGRAD.md) 推导各个反向算子。整体学习路径见
[LEARNING.md](../LEARNING.md)。

## 独立构建与测试

以下命令在 `cpp/` 目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j2
ctest --test-dir build --output-on-failure

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j2
ctest --test-dir build-release --output-on-failure

cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DTLAB_ENABLE_SANITIZERS=ON
cmake --build build-sanitize -j2
ctest --test-dir build-sanitize --output-on-failure
```

sanitizer 选项适用于支持 AddressSanitizer/UndefinedBehaviorSanitizer 的 GCC/Clang。
编译默认启用 `-Wall -Wextra -Wpedantic`；不使用 `-ffast-math`。
测试不需要 Python、PyTorch、下载数据或预先生成权重。

三组 CTest 检查：

| 测试 | 会捕获的问题 |
| --- | --- |
| `core_math` | 注意力解析 forward；linear/embedding/LayerNorm/attention/CE 的中央差分；重复 token 与共享 DAG 节点梯度；mask、shape、全 PAD 标签错误 |
| `model_format` | 两层模型；causal 不泄漏未来；PAD key 隔离；训练下降；EOS 结束；位置长度；config/token 校验；TLAB1 截断、重复、尾部、非有限值和溢出；case 快照 |
| `cli_contract` | 真正启动 CLI；inspect 全部梯度与 SGD 关系；零步训练保存；重复 SGD；保留 batch/config；生成行格式；无效参数和文件错误 |

`tlab_fixture` 是测试辅助程序，代码全部位于 `tests/`，不承担生产模型初始化 API。

## 使用 CLI

权重输入是同目录 Python 后端能够读写的 `TLAB1` case。三个命令：

```bash
build/transformer_cpp inspect case.tlab cpp-result.tlab
build/transformer_cpp train case.tlab cpp-trained.tlab --steps 100
build/transformer_cpp generate cpp-trained.tlab --bos 1 --eos 2 --max-new-tokens 8
```

`inspect` 使用原始权重做一次 forward/backward/SGD，结果包含：

- `logits`、`loss`、`encoder_output`、`decoder_output`；
- 每层 `encoder.{i}.output`、`decoder.{i}.output`；
- 每个参数对应的 `grad.<name>` 和 `updated.<name>`。

`train` 每轮重新创建 DAG，保存的 checkpoint 仍是完整 case，保留原来的配置、
学习率、src/tgt/labels。默认训练 100 步；`--steps 0` 保存原值。
stdout 显示初始和最终 loss。

`generate` 每个 batch 样本输出一行空格分隔整数，含开头的 BOS。
默认 BOS=1、EOS=2、max-new-tokens=8；这些 ID 必须在词表内、彼此不同且不等于 PAD。
某个样本产生 EOS 后，后续位置补 PAD；所有样本结束时提前停止。
零步预算只返回 BOS。BOS 加上预算不得超过 `max_len`。
argmax 平局取最小词表索引，与 PyTorch 相同。诊断输出到 stderr，错误返回非零退出码。

若要从 Python 导出同一份权重，在仓库根目录运行：

```bash
mkdir -p .artifacts
uv run --project python transformer-lab export .artifacts/copy.tlab --task copy --seed 7
cpp/build/transformer_cpp train .artifacts/copy.tlab .artifacts/cpp-trained.tlab --steps 300
uv run --project python transformer-lab generate .artifacts/cpp-trained.tlab --bos 1 --eos 2 --max-new-tokens 8
```

跨实现完整验证由根目录 `verify.py` 完成。比较时应载入相同权重；C++ 和 PyTorch
各自的随机初始化函数不承诺生成相同数列。

## 代码阅读地图

| 文件 | 内容 |
| --- | --- |
| `include/tlab/tensor.hpp` | row-major Tensor、Node DAG、算子接口 |
| `src/tensor.cpp` | 全部数学 forward/backward；只用循环与标准库 |
| `include/tlab/model.hpp`、`src/model.cpp` | 配置、参数名、PE、mask、encoder/decoder、SGD、生成 |
| `include/tlab/format.hpp`、`src/format.cpp` | TLAB1 parser/writer、case 载入和深拷贝快照 |
| `src/main.cpp` | 三个 CLI 命令及错误报告 |
| `tests/` | 数值、模型、序列化和进程级测试 |

第一次读源码可按下面三步推进：

1. 先对照 [tensor.hpp](include/tlab/tensor.hpp) 的形状契约与
   [tensor.cpp](src/tensor.cpp) 的 `linear`：前缀维变成 rows，最后一维是通道。
2. 再读 `layer_norm` 与 `attention`，同时对照 [AUTOGRAD.md](AUTOGRAD.md)
   的公式。注意力的循环标出 score、稳定 softmax、加权 V 三步，反向沿相反方向返回。
3. 最后读 [model.cpp](src/model.cpp) 的 `Transformer::forward`：先构造 mask，
   再串起 encoder、decoder 与词表投影。`src/main.cpp::train` 展示图的完整寿命。

张量的最后一维连续。例如 `[B,L,D]` 的元素 `(b,t,d)` 位于
`(b*L+t)*D+d`。Linear 权重存为 `[out,in]`，所以每个 token 的 forward 是
`y_o = sum_i x_i*W_oi + b_o`，与 PyTorch `nn.Linear` 的权重方向一致。
它也接受 `[I]`、`[R,I]` 等输入：所有前缀维都并入 rows，输出保留前缀、替换最后一维。
多头注意力在最后一维解释 `[H,d_h]`，每个 head 的 `d_h=D/H` 个元素连续。
Q 为 `[B,Lq,D]`，K/V 为 `[B,Lk,D]`，输出为 `[B,Lq,D]`；cross-attention
允许 `Lq != Lk`。mask 为展平的 `[B,Lq,Lk]`，True（非零）表示允许，并在所有 head 上共享。

`Tensor` 拷贝共享同一个 `Node`，用来表达残差和同一参数的复用。
因此 `auto copy = original` 后修改 `copy.data()`，也会改变 `original.data()`。
需要数值快照时显式执行 `Tensor(original.shape(), original.data())`；
`make_case` 已对模型参数和 batch 做深拷贝。
`Node::parents` 和回调只持有祖先节点，不持有输出节点，避免共享指针所有权环。

训练的库级调用方式如下；每轮的局部输出离开作用域后，中间图随之释放：

```cpp
auto fixture = tlab::load_case(tlab::read_bundle("case.tlab"));
for (std::size_t step = 0; step < 2; ++step) {
    auto prediction = fixture.model.forward(
        fixture.src, fixture.batch, fixture.source_length, fixture.tgt, fixture.target_length);
    auto loss = tlab::cross_entropy(
        prediction.logits, fixture.labels, fixture.model.config.pad_id);
    loss.backward();
    fixture.model.sgd(fixture.learning_rate);
}
```

`backward()` 接受一个元素的输出（`numel==1`），先清零全部可达节点的梯度，
再将输出梯度设为 1；同一次调用的共享路径通过 `+=` 累加。
因此上面的循环无需另调 `zero_grad`，不同 `backward()` 调用也不会积累上一轮梯度。
未参与本次输出的节点不在这个清零范围内。

从 forward 到 backward 之间，输入和参数数值必须保持不变：部分回调直接读取
父节点 `data`，另外一些读取 forward 保存的缓存。SGD 原地改动参数、保留梯度，
会使旧图的数值与缓存不再匹配；下一训练步必须重新 forward，再 backward。

## 文件与输入边界

TLAB1 是空白分隔文本：magic、记录数、每条记录的名字/rank/dims、row-major 数值。
writer 使用 classic locale 和 17 位有效数字。reader 拒绝重复名字、截断、额外 token、
非有限数、错误 rank/dims、整数溢出；case loader 再检查参数集合、shape、config、
整数 token、词表范围和 mask 前提。rank 为 1..4；所有维度严格为正。
单张量及文件总数值最多 10,000,000，文件记录最多 10,000，文件最多 256 MiB；
配置总参数最多 10,000,000，每种层数最多 1,000。

源序列每行至少有一个非 PAD token；目标每行第一个 token 必须非 PAD；
所有 label 都为 PAD 会报错。允许 PAD query 的输出存在，只在 key 和 loss 处排除 PAD。
核心固定 dropout=0、float64/double、无 momentum/weight decay 的 SGD。
这是适合逐项理解的 CPU 实现；它每步重算完整 prefix，没有 KV cache 或 CUDA kernel。
