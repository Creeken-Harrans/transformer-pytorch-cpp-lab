# 交付验证记录

## 2026-10-13：可读性整理

本次使用 `humanizer-zh`、`anti-defensive-writing` 整理中文材料，并按 `superpowers:requesting-code-review` 进行独立审阅。README 增加运行路线与公式/源码对应表，LEARNING 增加目录、变长批次和 mask 自检，C++ 注释补齐张量形状、梯度路径及计算图寿命。两套核心模型的数学契约与计算顺序保持不变。

新增 `make help`、`make walkthrough`，默认验证结果改为检查表。`verify.py --json` 输出可解析的完整 JSON；`--json-output` 在普通验证与复制任务验证中都保存报告，修复了复制任务忽略该参数的问题。

| 本次检查 | 实测结果 |
| --- | --- |
| `make check` | lint、98 项 Python 测试、3 组 CTest、六个数值 case 全部通过 |
| 六个 case 的最大绝对误差 | 1.22e-15；有限差分最大误差 4.38e-11；官方 PyTorch 层参考最大误差 6.66e-16 |
| `make walkthrough` | 5 个有效 labels；logits `[2,3,8]`；梯度范数 1.21562491766；一步 loss 2.45675282567 → 2.32730867428 |
| `make demo` | 两端 300 步 loss 均降至 0.0113658583467；交叉加载后逐 token 复制、EOS/PAD 规则通过 |
| C++ ASan / UBSan | 3 组 CTest 全部通过，无报告 |
| C++ 严格 Debug 构建与格式 | `-Werror` 构建、CTest、clang-format 通过 |
| JSON 报告实际 CLI | 两种模式的 stdout 都可直接解析，且与保存文件完全相同 |
| 文档与独立审阅 | 111 个本地链接、代码围栏、diff 格式通过；未发现需修复的问题 |

新增的 5 项 Python 测试覆盖报告显示、JSON 保存、复制任务报告、实际单步训练和默认 Make 入口。以下命令可重现可读输出与完整报告：

```bash
make walkthrough
uv run --locked --project python python verify.py --json-output .artifacts/validation.json
uv run --locked --project python python verify.py --json
```

## 2026-10-12：首次交付

2026-10-12 在 Linux x86_64 上运行，Python 3.12.14、PyTorch 2.8.0+cpu、uv 0.12.23、GCC 11.4.0、CMake 3.22.1。依赖来自提交的 `python/uv.lock`，浮点为 float64/double，未启用 fast-math。

[GitHub Actions 独立验证](https://github.com/Creeken-Harrans/transformer-pytorch-cpp-lab/actions/runs/38272236845) 已通过：从远程克隆、锁定环境安装、lint、两端测试、数值一致性、复制任务与 sanitizer 都成功。测试提交为 `db8ecaf`；之后的提交仅补充这份交付记录与计划完成状态。

### 可复现入口

```bash
make check
make demo
cmake -S cpp -B cpp/build-sanitize -DCMAKE_BUILD_TYPE=Debug -DTLAB_ENABLE_SANITIZERS=ON
cmake --build cpp/build-sanitize --parallel 2
ctest --test-dir cpp/build-sanitize --output-on-failure
```

| 检查 | 实测结果 |
| --- | --- |
| uv 锁定环境同步 | 通过 |
| ruff 静态与格式检查 | 通过 |
| Python 行为与数学测试 | 93 项通过 |
| C++ 独立测试 | 3 组 CTest 全通过：core_math、model_format、cli_contract |
| C++ AddressSanitizer / UndefinedBehaviorSanitizer | 3 组 CTest 全通过，无报告 |
| C++ clang-format / 编译告警 | 格式检查通过，构建无告警 |
| 错误 case | 12 个均被两个实现拒绝，并给出诊断 |
| checkpoint 与生成 | 双向互读、EOS 提前停止、混合结束时间后的 PAD、零步训练通过 |

### 数值一致性

验证器逐个比较每层边界、logits、loss、所有参数梯度、一次 SGD 后的所有权重。容差 `atol=1e-9, rtol=1e-8`。

| case | 比较记录数 | 最大绝对误差 |
| --- | ---: | ---: |
| baseline | 98 | 5.55e-16 |
| different_lengths | 98 | 8.88e-16 |
| two_layers_three_heads | 184 | 8.88e-16 |
| odd_dimension_nonzero_pad | 98 | 8.88e-16 |
| one_head | 98 | 1.22e-15 |
| one_token | 98 | 4.44e-16 |

每个 case 另对 8 个代表性参数做中央有限差分，最大绝对误差 **4.38e-11**；与 PyTorch 官方 encoder/decoder 层独立组装的前向结果比较，最大绝对误差 **6.66e-16**。官方层仅用于测试，两个核心实现自行组装。

同一个 baseline 连续训练 12 步，loss `3.29256263 → 2.01704011`；两端全部 checkpoint 权重的最大绝对误差 `2.22e-16`。

### 复制任务

默认固定 batch、seed=7、learning_rate=0.1，分别训练 300 步：

- 初始 loss：`3.22212031766`。
- Python 最终 loss：`0.0113658583467`。
- C++ 最终 loss：`0.0113658583467`。
- 两端交叉读取 checkpoint，生成均精确等于 `BOS + source`。

```text
1 9 15 7 7 2
1 6 8 5 2 0
```

第二条序列先结束，最后的 `0` 是 PAD。以上证明这个固定 batch 被正确拟合，泛化能力需要另外设计数据划分。不同平台的舍入误差与运行时间可能不同，验收以容差和行为断言为准。
