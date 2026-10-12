# Run from the repository root. Each target uses the committed uv lockfile.
UV := uv run --locked --project python
CMAKE_BUILD := cpp/build
.DEFAULT_GOAL := help

.PHONY: help sync build walkthrough test lint parity check demo clean

help:
	@echo '从仓库根目录运行；Python 命令均使用已提交的 uv.lock。'
	@echo '  make sync         安装锁定的 Python 环境'
	@echo '  make walkthrough  查看输入、mask、注意力与一次 SGD 更新'
	@echo '  make demo         编译两端并训练、验证复制任务'
	@echo '  make check        运行 lint、两端测试和数值一致性检查'
	@echo '  make test         运行 Python 与 C++ 各自的测试（含 C++ 构建）'
	@echo '  make lint         运行 Python 静态与格式检查'
	@echo '  make build        仅编译 C++ 后端'
	@echo '  make parity       仅运行两端数值一致性检查（含 C++ 构建）'
	@echo '  make clean        删除构建产物与生成的 case'
	@echo '阅读：LEARNING.md 对照公式；cpp/AUTOGRAD.md 推导手写梯度。'

sync:
	uv sync --locked --project python

build:
	cmake -S cpp -B $(CMAKE_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(CMAKE_BUILD) --parallel 2

walkthrough:
	$(UV) python python/examples/walkthrough.py

test: build
	$(UV) pytest python/tests
	ctest --test-dir $(CMAKE_BUILD) --output-on-failure

lint:
	$(UV) ruff check --config python/pyproject.toml python verify.py
	$(UV) ruff format --check --config python/pyproject.toml python verify.py

parity: build
	$(UV) python verify.py --cpp-binary $(CMAKE_BUILD)/transformer_cpp

check: lint test parity

# Deliberately overfit one tiny batch: this makes every training step observable.
demo: build
	mkdir -p .artifacts
	$(UV) transformer-lab export .artifacts/copy.tlab --task copy --learning-rate 0.1
	$(UV) transformer-lab train .artifacts/copy.tlab .artifacts/python-trained.tlab --steps 300
	$(CMAKE_BUILD)/transformer_cpp train .artifacts/copy.tlab .artifacts/cpp-trained.tlab --steps 300
	$(UV) transformer-lab generate .artifacts/cpp-trained.tlab --bos 1 --eos 2 --max-new-tokens 8
	$(CMAKE_BUILD)/transformer_cpp generate .artifacts/python-trained.tlab --bos 1 --eos 2 --max-new-tokens 8
	$(UV) python verify.py --cpp-binary $(CMAKE_BUILD)/transformer_cpp --copy-checkpoints .artifacts/python-trained.tlab .artifacts/cpp-trained.tlab

clean:
	rm -rf cpp/build cpp/build-sanitize .artifacts
