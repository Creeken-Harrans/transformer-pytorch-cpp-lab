# Transformer Learning Lab Implementation Plan

**Goal:** 交付可对照学习、可训练、通过数值一致性验证的 PyTorch/C++ Transformer 公共仓库。

**Architecture:** 两个实现独立编写，遵循 DESIGN.md 的同一数学与文件契约。根目录负责学习导航、跨语言验收和 CI。

**Tech Stack:** Python 3.12、PyTorch、uv、pytest、ruff；C++17、CMake、CTest。

**Spec:** [DESIGN.md](DESIGN.md)

## Global Constraints

Post-LN、float64、dropout=0、SGD、PAD key 和 label masks、共享参数名字、TLAB1 精度 17 位；不依赖下载数据集；中文解释与详细数学注释。

## Review Focus

全 PAD 输入/标签、非整数或越界 token；多层及不同序列长度；残差共享节点的梯度累加；mask 对未来/PAD 的真正隔离；文件的截断/重复/错误尺寸及生成 EOS 边界。

## Task 1 — PyTorch 后端（独立实现代理）

- [x] 先写数学、mask、输入验证及反传的失败测试。
- [x] 实现 model.py、format.py 和 cli.py，注释 shapes 与公式。
- [x] 提供 fixture/export/inspect/train/generate 入口，uv.lock 和运行说明。
- [x] pytest、ruff、训练 loss 下降通过；返回接口与证据。

## Task 2 — C++ 后端（独立实现代理）

- [x] 先写算子数值与梯度的失败测试。
- [x] 实现张量/DAG autograd、线性/嵌入/LayerNorm/attention/loss。
- [x] 完成共享模型和 TLAB1 CLI，支持训练和生成。
- [x] CTest、严格编译告警及必要有限差分检查通过。

## Task 3 — 学习材料（文档代理）

- [x] 根 README 与 LEARNING.md，数学、形状、阅读映射、实验及局限。
- [x] 根 Makefile、MIT license 与贡献规范由主代理核对。
- [x] 检查所有运行命令与最终接口匹配。

## Task 4 — 集成与交付（主代理）

- [x] 写 verify.py，逐权重比较 forward/backward/SGD，加入 padded/multilayer cases 与生成。
- [x] 添加 CI、gitignore；运行 uv 锁定检查、lint、Python 测试、C++ 测试、parity 与训练示例。
- [x] 独立审阅实现和文档，修复发现的问题并重跑相关验证。
- [ ] 建立分阶段 Git 提交、创建公开 GitHub 仓库、推送并核验远端可访问及工作区干净。
