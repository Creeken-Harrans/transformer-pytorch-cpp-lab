# 贡献指南

欢迎改进数学解释、代码可读性、错误检查和教学实验。先阅读 [DESIGN.md](DESIGN.md) 的数值契约，并按[学习指南第 8 节](LEARNING.md#8-代码阅读路线)找到对应函数；涉及数学或接口的改动需让两套实现保持一致。

## 先确定改动解决什么问题

适合核心仓库的改动包括：修正公式或梯度、补充真实边界的测试、完善输入与文件验证、简化代码，以及给容易误解的地方添加形状和推导注释。新增功能应说明对学习目标的帮助，以及是否影响两边的接口或数值契约。

性能实验可以保留为独立扩展，但核心 C++ 继续仅依赖标准库，Python 继续由 uv 锁定依赖。避免让新的工程层遮住数学路径。涉及架构、mask、参数名字或 TLAB 格式的变化，应同时修改两个后端、验证器和文档。

## 本地检查

首次运行的工具与环境要求见根目录 [README.md](README.md#先运行再读代码)。以下命令都在仓库根目录执行：

| 命令 | 用途 |
|---|---|
| `make help` | 查看所有入口与阅读路线 |
| `make walkthrough` | 观察一次实际 forward、backward 与 SGD，核对文档中的形状和 mask |
| `make lint` | Python 静态检查与格式检查 |
| `make test` | Python 与 C++ 各自的测试 |
| `make parity` | 跨语言 forward、backward、SGD、checkpoint 与生成对照 |
| `make demo` | 固定小批次的复制训练与交叉加载演示 |
| `make check` | 汇总 lint、test 与 parity，提交前运行 |

需要保存跨语言验证的完整证据时：

```bash
uv run --locked --project python python verify.py --json-output .artifacts/validation.json
```

默认终端输出便于阅读的检查表；`--json` 提供可解析的 JSON stdout。报告 PR 数值结果时，注明 case、比较对象和误差字段，保留比较条件。

C++ 构建与 CTest 可单独运行：

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

使用提交的 `python/uv.lock`。增加或更新 Python 依赖时用 uv 管理并提交锁文件，说明必要性；不要手改锁文件。生成的 `.venv`、build 目录与实验 checkpoint 不加入版本控制。

## 测试应证明行为

数学或梯度改动应有能暴露原问题的测试。优先选择很小、可手算的输入，再增加 batch、mask 或共享节点的情况。中央有限差分适合检验局部 backward；涉及 ReLU 时避开不可微点，写清步长和容差。

模型行为测试应覆盖实际约束，例如未来 token 不影响早期 logits、源 PAD 不影响有效输出、重复父节点正确累加、全 PAD labels 被拒绝、不同源/目标长度可运行，以及 EOS 后的 batch 行为。随机测试使用固定种子；跨语言测试读取同一组权重，而非分别初始化。

跨语言比较遵守[学习指南第 7 节](LEARNING.md#7-怎样判断实现可信)的统一容差。超出容差时，先报告差异最大的记录和元素，再沿形状、mask、广播、transpose、归一化和梯度累加定位原因。改变精度或算法的实验，单独说明比较目标与容差依据。

## 注释与文档

注释优先回答三件事：张量的形状、公式为何成立、哪里容易出错。对直接可读的赋值，无需复述语法。公开接口、命令、参数名字或生成规则变化时，同步更新 README、学习指南与相应子目录说明。

中文解释保持具体；可以保留 encoder、decoder、logits、mask、backward 等便于对照代码的术语。引用数学与库行为时优先链接原论文或官方文档。

## 提交与评审

一次提交围绕一个可解释的改动。PR 描述先写问题和修改后的行为，再列相关验证命令及结果；数值修复附一个最小可复现输入或明确的测试名称。若改动影响两套实现，指出如何核对一致性。

报告 bug 时，请提供运行命令、配置、环境、实际结果与预期结果；如涉及 `.tlab`，尽量提供一个缩小后的 case。学习上的疑问也欢迎提出，尤其是无法从公式、注释和测试中连起来的一段路径。

贡献按本仓库的 [MIT License](LICENSE) 发布。
