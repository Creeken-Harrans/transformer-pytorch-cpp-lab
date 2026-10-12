# 交付验证记录

2026-10-12 在 Linux x86_64 上运行，Python 3.12.14、PyTorch 2.8.0+cpu、uv 0.12.23、GCC 11.4.0、CMake 3.22.1。依赖来自提交的 `python/uv.lock`，浮点为 float64/double，未启用 fast-math。

## 可复现入口

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

## 数值一致性

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

## 复制任务

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
