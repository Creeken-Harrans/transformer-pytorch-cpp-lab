"""逐层检查两个实现是否在计算同一个 Transformer。

运行：uv run --locked --project python python verify.py

这里把 PyTorch 与 C++ 当作独立实现，而不是让一个调用另一个。
共享的是初始权重、输入和数学契约；比较覆盖 forward、backward、SGD、
checkpoint 互读和 greedy decoding。float64 缩小舍入误差，便于定位错误。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import torch
from transformer_lab.format import load_case, make_case, read_bundle, write_bundle
from transformer_lab.model import (
    Transformer,
    TransformerConfig,
    masked_cross_entropy,
)

ROOT = Path(__file__).resolve().parent
ATOL = 1e-9
RTOL = 1e-8


def run(command: list[str]) -> str:
    """失败时保留后端的原始诊断，成功时不打印重复的构建输出。"""
    result = subprocess.run(
        command, text=True, capture_output=True, check=False, timeout=180
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout


def build_cpp() -> Path:
    build = ROOT / "cpp" / "build"
    run(
        [
            "cmake",
            "-S",
            str(ROOT / "cpp"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run(["cmake", "--build", str(build), "--parallel", "2"])
    return build / "transformer_cpp"


def tokens(
    config: TransformerConfig, batch: int, source_length: int, target_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """产生右侧 PAD 的批次，源和目标长度刻意不总相等。

    BOS=1、EOS=2；PAD 可以为 3，从而检查代码是否把 PAD 写死为 0。
    这里构造的是数值测试输入，不用它衡量任务泛化。
    """
    ordinary = [i for i in range(config.vocab_size) if i not in {config.pad_id, 1, 2}]
    src = torch.full((batch, source_length), config.pad_id, dtype=torch.long)
    tgt = torch.full((batch, target_length), config.pad_id, dtype=torch.long)
    labels = torch.full_like(tgt, config.pad_id)
    for b in range(batch):
        source_count = max(1, source_length - b)
        target_count = max(1, target_length - b)
        for s in range(source_count):
            src[b, s] = ordinary[(b * 3 + s) % len(ordinary)]
        tgt[b, 0] = 1
        for t in range(1, target_count):
            tgt[b, t] = ordinary[(b + t - 1) % len(ordinary)]
        # Teacher forcing：decoder 输入右移一位，预测对应位置的下一个 token。
        if target_count > 1:
            labels[b, : target_count - 1] = tgt[b, 1:target_count]
        labels[b, target_count - 1] = 2
    return src, tgt, labels


def expected_inspection(
    model: Transformer,
    src: torch.Tensor,
    tgt: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    logits, trace = model.forward_with_trace(src, tgt)
    loss = masked_cross_entropy(logits, labels, model.config.pad_id)
    loss.backward()
    result = {"logits": logits.detach(), "loss": loss.detach().reshape(1)}
    result.update({name: value.detach() for name, value in trace.items()})
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Parameter disconnected from loss: {name}")
        result[f"grad.{name}"] = parameter.grad.detach().clone()
        result[f"updated.{name}"] = (
            parameter.detach() - learning_rate * parameter.grad.detach()
        )
    return result


def compare(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], context: str
) -> dict[str, float | int]:
    """先检查名字和形状，避免只比相同元素数量而漏掉布局错误。"""
    if expected.keys() != actual.keys():
        raise AssertionError(
            f"{context}: missing={sorted(expected.keys() - actual.keys())}, "
            f"unexpected={sorted(actual.keys() - expected.keys())}"
        )
    maximum = 0.0
    elements = 0
    for name in sorted(expected):
        # TLAB1 统一存储 double；token/config 的整数语义由 load_case 校验。
        left, right = expected[name].double(), actual[name]
        if left.shape != right.shape:
            raise AssertionError(f"{context}/{name}: {left.shape} != {right.shape}")
        if not torch.isfinite(right).all():
            raise AssertionError(f"{context}/{name}: non-finite C++ output")
        torch.testing.assert_close(
            right,
            left,
            atol=ATOL,
            rtol=RTOL,
            msg=lambda message, n=name: f"{context}/{n}: {message}",
        )
        maximum = max(maximum, (left - right).abs().max().item())
        elements += left.numel()
    return {"records": len(expected), "elements": elements, "max_abs_error": maximum}


def finite_difference(
    model: Transformer, src: torch.Tensor, tgt: torch.Tensor, labels: torch.Tensor
) -> dict[str, float | int]:
    """用真实 loss 的中央差分检查完整模型的代表性参数。

    两个实现相等仍可能共同犯错；差分提供额外的数学检查。
    每个参数选择绝对梯度最大的元素，避免落在 ReLU 零梯度区域。
    """
    selectors = (
        "src_embedding.weight",
        "tgt_embedding.weight",
        "encoder.0.self_attn.q_proj.weight",
        "encoder.0.norm1.weight",
        "encoder.0.ffn.linear1.weight",
        "decoder.0.cross_attn.k_proj.weight",
        "decoder.0.self_attn.v_proj.weight",
        "generator.weight",
    )
    params = dict(model.named_parameters())
    maximum = 0.0
    step = 1e-5
    for name in selectors:
        parameter = params[name]
        assert parameter.grad is not None
        index = parameter.grad.abs().reshape(-1).argmax().item()
        analytic = parameter.grad.reshape(-1)[index].item()
        flat = parameter.reshape(-1)
        original = flat[index].item()
        with torch.no_grad():
            try:
                flat[index] = original + step
                plus = masked_cross_entropy(
                    model(src, tgt), labels, model.config.pad_id
                ).item()
                flat[index] = original - step
                minus = masked_cross_entropy(
                    model(src, tgt), labels, model.config.pad_id
                ).item()
            finally:
                flat[index] = original
        numeric = (plus - minus) / (2 * step)
        error = abs(analytic - numeric)
        if error > 2e-7 + 2e-5 * abs(analytic):
            raise AssertionError(f"finite difference {name}: {analytic} != {numeric}")
        maximum = max(maximum, error)
    return {"parameters_sampled": len(selectors), "max_abs_error": maximum}


def verify_torch_reference(
    model: Transformer, src: torch.Tensor, tgt: torch.Tensor
) -> float:
    """额外与 PyTorch 官方层比较；教学实现本身不调用这些层。

    官方 nn.Transformer 会增加末尾的归一化，因此这里直接叠 EncoderLayer
    和 DecoderLayer，以精确匹配原论文的 Post-LN 定义。官方 MHA 的布尔
    mask 是 True=屏蔽，与本仓库的自定义 attention mask 相反。
    """
    config = model.config
    weights = model.state_dict()
    options = {
        "d_model": config.d_model,
        "nhead": config.num_heads,
        "dim_feedforward": config.d_ff,
        "dropout": 0.0,
        "activation": "relu",
        "layer_norm_eps": config.eps,
        "batch_first": True,
        "norm_first": False,
        "dtype": torch.float64,
    }

    def transfer_attention(module: torch.nn.MultiheadAttention, prefix: str) -> None:
        module.in_proj_weight.copy_(
            torch.cat([weights[f"{prefix}.{p}_proj.weight"] for p in ("q", "k", "v")])
        )
        module.in_proj_bias.copy_(
            torch.cat([weights[f"{prefix}.{p}_proj.bias"] for p in ("q", "k", "v")])
        )
        module.out_proj.weight.copy_(weights[f"{prefix}.out_proj.weight"])
        module.out_proj.bias.copy_(weights[f"{prefix}.out_proj.bias"])

    def transfer_rest(module: torch.nn.Module, prefix: str, norms: int) -> None:
        for linear in ("linear1", "linear2"):
            getattr(module, linear).weight.copy_(weights[f"{prefix}.ffn.{linear}.weight"])
            getattr(module, linear).bias.copy_(weights[f"{prefix}.ffn.{linear}.bias"])
        for number in range(1, norms + 1):
            norm = f"norm{number}"
            getattr(module, norm).weight.copy_(weights[f"{prefix}.{norm}.weight"])
            getattr(module, norm).bias.copy_(weights[f"{prefix}.{norm}.bias"])

    # 以标量公式独立重建位置编码，避免重用待验证实现的函数。
    position = torch.empty(config.max_len, config.d_model, dtype=torch.float64)
    for p in range(config.max_len):
        for channel in range(config.d_model):
            angle = p / (10000 ** (2 * (channel // 2) / config.d_model))
            position[p, channel] = math.sin(angle) if channel % 2 == 0 else math.cos(angle)
    with torch.no_grad():
        memory = torch.nn.functional.embedding(src, weights["src_embedding.weight"])
        memory = memory * math.sqrt(config.d_model) + position[: src.shape[1]]
        for i in range(config.num_encoder_layers):
            layer = torch.nn.TransformerEncoderLayer(**options)
            transfer_attention(layer.self_attn, f"encoder.{i}.self_attn")
            transfer_rest(layer, f"encoder.{i}", 2)
            memory = layer(memory, src_key_padding_mask=src == config.pad_id)
        hidden = torch.nn.functional.embedding(tgt, weights["tgt_embedding.weight"])
        hidden = hidden * math.sqrt(config.d_model) + position[: tgt.shape[1]]
        future_mask = torch.ones(tgt.shape[1], tgt.shape[1], dtype=torch.bool).triu(1)
        for i in range(config.num_decoder_layers):
            layer = torch.nn.TransformerDecoderLayer(**options)
            transfer_attention(layer.self_attn, f"decoder.{i}.self_attn")
            transfer_attention(layer.multihead_attn, f"decoder.{i}.cross_attn")
            transfer_rest(layer, f"decoder.{i}", 3)
            hidden = layer(
                hidden,
                memory,
                tgt_mask=future_mask,
                tgt_key_padding_mask=tgt == config.pad_id,
                memory_key_padding_mask=src == config.pad_id,
            )
        reference = torch.nn.functional.linear(
            hidden, weights["generator.weight"], weights["generator.bias"]
        )
        actual = model(src, tgt)
        torch.testing.assert_close(actual, reference, atol=ATOL, rtol=RTOL)
    return (actual - reference).abs().max().item()


def verify_training_and_generation(
    binary: Path, directory: Path, case_path: Path
) -> dict[str, float | int]:
    bundle = read_bundle(case_path)
    model, src, tgt, labels, learning_rate = load_case(bundle)
    steps = 12
    cpp_checkpoint = directory / "cpp-checkpoint.tlab"
    run([str(binary), "train", str(case_path), str(cpp_checkpoint), "--steps", str(steps)])
    initial_loss = masked_cross_entropy(model(src, tgt), labels, model.config.pad_id).item()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        loss = masked_cross_entropy(model(src, tgt), labels, model.config.pad_id)
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                assert parameter.grad is not None
                parameter -= learning_rate * parameter.grad
    final_loss = masked_cross_entropy(model(src, tgt), labels, model.config.pad_id).item()
    if not final_loss < initial_loss:
        raise AssertionError(f"SGD did not reduce loss: {initial_loss} -> {final_loss}")
    expected_case = make_case(model, src, tgt, labels, learning_rate)
    checkpoint_report = compare(expected_case, read_bundle(cpp_checkpoint), "12 SGD steps")
    # 双向互通：Python 读取 C++ checkpoint；C++ 读取 Python checkpoint。
    cpp_model, _, _, _, _ = load_case(read_bundle(cpp_checkpoint))
    python_checkpoint = directory / "python-checkpoint.tlab"
    write_bundle(python_checkpoint, expected_case)
    expected_tokens = model.greedy_decode(src, 1, 2, 6)
    torch.testing.assert_close(cpp_model.greedy_decode(src, 1, 2, 6), expected_tokens)
    stdout = run(
        [
            str(binary),
            "generate",
            str(python_checkpoint),
            "--bos",
            "1",
            "--eos",
            "2",
            "--max-new-tokens",
            "6",
        ]
    )
    actual_tokens = torch.tensor(
        [[int(token) for token in row.split()] for row in stdout.strip().splitlines()],
        dtype=torch.long,
    )
    torch.testing.assert_close(actual_tokens, expected_tokens, atol=0, rtol=0)
    # 显式覆盖 EOS：将 generator 设为恒定输出 EOS，应只产生 BOS,EOS。
    with torch.no_grad():
        model.generator.weight.zero_()
        model.generator.bias.fill_(-10)
        model.generator.bias[2] = 10
    eos_case = directory / "eos.tlab"
    write_bundle(eos_case, make_case(model, src, tgt, labels, learning_rate))
    eos_stdout = run(
        [
            str(binary),
            "generate",
            str(eos_case),
            "--bos",
            "1",
            "--eos",
            "2",
            "--max-new-tokens",
            "6",
        ]
    )
    eos_tokens = [[int(t) for t in row.split()] for row in eos_stdout.strip().splitlines()]
    if eos_tokens != [[1, 2]] * src.shape[0]:
        raise AssertionError(f"C++ EOS early stopping failed: {eos_tokens}")
    torch.testing.assert_close(model.greedy_decode(src, 1, 2, 6), torch.tensor(eos_tokens))
    # 混合结束时间：用两个不同源的 BOS 隐藏状态构造一个分离超平面。
    # row0 的 EOS 分数为 +2，row1 为 -2；另一个 token 的恒定分数为 0。
    # 因此第一步只有 row0 结束，第二步必须给它追加 PAD，而 row1 继续。
    with torch.no_grad():
        _, trace = model.forward_with_trace(src, tgt[:, :1])
        first, second = trace["decoder_output"][:, 0]
        difference = first - second
        separation = difference.square().sum().item()
        if separation < 1e-12:
            raise AssertionError("Mixed-EOS fixture needs distinct source representations")
        eos_weight = difference * (4 / separation)
        model.generator.weight.zero_()
        model.generator.bias.fill_(-10)
        model.generator.bias[4] = 0
        model.generator.weight[2].copy_(eos_weight)
        model.generator.bias[2] = -((first + second) * eos_weight).sum() / 2
    mixed_case = directory / "mixed-eos.tlab"
    write_bundle(mixed_case, make_case(model, src, tgt, labels, learning_rate))
    mixed_expected = model.greedy_decode(src, 1, 2, 3)
    mixed_stdout = run(
        [
            str(binary),
            "generate",
            str(mixed_case),
            "--bos",
            "1",
            "--eos",
            "2",
            "--max-new-tokens",
            "3",
        ]
    )
    mixed_actual = torch.tensor(
        [[int(t) for t in row.split()] for row in mixed_stdout.strip().splitlines()]
    )
    torch.testing.assert_close(mixed_actual, mixed_expected, atol=0, rtol=0)
    if mixed_actual.shape[1] < 3 or mixed_actual[0, 1] != 2 or mixed_actual[1, 1] != 4:
        raise AssertionError(f"Mixed-EOS separation failed: {mixed_actual}")
    if not (mixed_actual[0, 2:] == model.config.pad_id).all():
        raise AssertionError(f"Finished row must receive PAD: {mixed_actual}")
    return {
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "checkpoint_max_abs_error": checkpoint_report["max_abs_error"],
        "generation_rows": src.shape[0],
        "mixed_eos_padding_verified": True,
    }


def verify_rejections(binary: Path, directory: Path, case_path: Path) -> int:
    """文件可解析并不表示数据可训练：分别检查格式和模型语义错误。"""
    original = read_bundle(case_path)
    corruptions = {}
    for name in (
        "negative_token",
        "fractional_token",
        "invalid_heads",
        "source_all_pad",
        "target_starts_pad",
        "labels_all_pad",
        "missing_weight",
    ):
        corruptions[name] = {key: value.clone() for key, value in original.items()}
    corruptions["negative_token"]["src"][0, 0] = -1
    corruptions["fractional_token"]["src"] = original["src"].double().clone()
    corruptions["fractional_token"]["src"][0, 0] = 4.5
    corruptions["invalid_heads"]["config"][2] = 3
    pad = int(original["config"][7].item())
    corruptions["source_all_pad"]["src"][0].fill_(pad)
    corruptions["target_starts_pad"]["tgt"][0, 0] = pad
    corruptions["labels_all_pad"]["labels"].fill_(pad)
    del corruptions["missing_weight"]["generator.weight"]

    invalid_paths = []
    for name, bundle in corruptions.items():
        path = directory / f"invalid-{name}.tlab"
        write_bundle(path, bundle)
        invalid_paths.append(path)
    text = case_path.read_text(encoding="utf-8")
    for name, contents in {
        "duplicate": "TLAB1\n2\nx 1 1\n0\nx 1 1\n1\n",
        "nonfinite": "TLAB1\n1\nx 1 1\nnan\n",
        "truncated": " ".join(text.split()[:-1]),
        "trailing": text + "unexpected\n",
        "oversized": "TLAB1\n1\nx 2 999999999999 999999999999\n",
    }.items():
        path = directory / f"invalid-{name}.tlab"
        path.write_text(contents, encoding="utf-8")
        invalid_paths.append(path)
    for path in invalid_paths:
        try:
            model, src, tgt, labels, _ = load_case(read_bundle(path))
            masked_cross_entropy(model(src, tgt), labels, model.config.pad_id)
        except (ValueError, TypeError, RuntimeError, KeyError):
            pass
        else:
            raise AssertionError(f"Python accepted invalid case: {path.stem}")
        result = subprocess.run(
            [str(binary), "inspect", str(path), str(directory / "invalid-result.tlab")],
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        if result.returncode <= 0 or not result.stderr.strip():
            raise AssertionError(
                f"C++ must reject with a diagnostic, not crash: {path.stem}\n"
                f"exit={result.returncode}, stderr={result.stderr}"
            )
    return len(invalid_paths)


def verify_copy_checkpoint(binary: Path, path: Path) -> dict[str, float | int]:
    """demo 的验收：两端必须实际生成所复制的 token，而不只打印 loss。"""
    model, src, tgt, labels, _ = load_case(read_bundle(path))
    if src.shape != labels.shape or not torch.equal(src, labels):
        raise AssertionError("Copy demo must have labels equal to the source sequence")
    expected = torch.cat((torch.ones(src.shape[0], 1, dtype=torch.long), src), dim=1)
    generated = model.greedy_decode(src, 1, 2, src.shape[1])
    torch.testing.assert_close(generated, expected, atol=0, rtol=0)
    stdout = run(
        [
            str(binary),
            "generate",
            str(path),
            "--bos",
            "1",
            "--eos",
            "2",
            "--max-new-tokens",
            str(src.shape[1]),
        ]
    )
    cpp_tokens = torch.tensor(
        [[int(token) for token in row.split()] for row in stdout.strip().splitlines()]
    )
    torch.testing.assert_close(cpp_tokens, expected, atol=0, rtol=0)
    with torch.no_grad():
        loss = masked_cross_entropy(model(src, tgt), labels, model.config.pad_id).item()
    if not loss < 0.1:
        raise AssertionError(f"Copy demo needs final loss < 0.1, got {loss}")
    return {"rows_copied_exactly": src.shape[0], "loss": loss}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-binary", type=Path, help="Use an existing C++ build")
    parser.add_argument(
        "--copy-checkpoints",
        type=Path,
        nargs="+",
        help="Verify trained copy-task checkpoints",
    )
    parser.add_argument(
        "--json-output", type=Path, help="Optionally save the validation report"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    binary = args.cpp_binary.resolve() if args.cpp_binary else build_cpp()
    if args.copy_checkpoints:
        copy_report = {
            str(path): verify_copy_checkpoint(binary, path)
            for path in args.copy_checkpoints
        }
        print(json.dumps(copy_report, ensure_ascii=False, indent=2))
        print("PASS: both backends reproduce every copy-task token and stop at EOS.")
        return
    specifications = [
        ("baseline", {}, 2, 4, 4),
        ("different_lengths", {}, 2, 3, 6),
        (
            "two_layers_three_heads",
            {
                "d_model": 12,
                "num_heads": 3,
                "d_ff": 19,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
            },
            2,
            5,
            4,
        ),
        (
            "odd_dimension_nonzero_pad",
            {"d_model": 9, "num_heads": 3, "d_ff": 13, "pad_id": 3, "eps": 1e-4},
            2,
            4,
            5,
        ),
        ("one_head", {"d_model": 4, "num_heads": 1, "d_ff": 7}, 3, 4, 3),
        ("one_token", {}, 1, 1, 1),
    ]
    report: dict[str, object] = {"atol": ATOL, "rtol": RTOL, "cases": {}}
    with tempfile.TemporaryDirectory(prefix="transformer-parity-") as temporary:
        directory = Path(temporary)
        for number, (name, changes, batch, source_length, target_length) in enumerate(
            specifications
        ):
            torch.manual_seed(1729 + number)
            config = TransformerConfig(**changes)
            model = Transformer(config).double()
            src, tgt, labels = tokens(config, batch, source_length, target_length)
            learning_rate = 0.05
            case_path, result_path = (
                directory / f"{name}.tlab",
                directory / f"{name}-cpp.tlab",
            )
            write_bundle(case_path, make_case(model, src, tgt, labels, learning_rate))
            expected = expected_inspection(model, src, tgt, labels, learning_rate)
            run([str(binary), "inspect", str(case_path), str(result_path)])
            case_report = compare(expected, read_bundle(result_path), name)
            case_report["finite_difference"] = finite_difference(model, src, tgt, labels)
            case_report["official_pytorch_layer_max_abs_error"] = verify_torch_reference(
                model, src, tgt
            )
            report["cases"][name] = case_report
            if number == 0:
                report["training_and_generation"] = verify_training_and_generation(
                    binary, directory, case_path
                )
                report["invalid_cases_rejected_by_both"] = verify_rejections(
                    binary, directory, case_path
                )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print("PASS: layer outputs, losses, all gradients, SGD updates and generation agree.")


if __name__ == "__main__":
    main()
