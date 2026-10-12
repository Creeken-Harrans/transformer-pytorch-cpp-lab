# Run from the repository root. Each target uses the committed uv lockfile.
UV := uv run --locked --project python
CMAKE_BUILD := cpp/build

.PHONY: sync build test lint parity check demo clean

sync:
	uv sync --locked --project python

build:
	cmake -S cpp -B $(CMAKE_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(CMAKE_BUILD) --parallel 2

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
