"""格式测试同时防止数值精度损失、错误尺寸及 CLI 静默丢弃训练状态。"""

import importlib
import importlib.util
import subprocess
import sys

import pytest
import torch


def format_api():
    assert importlib.util.find_spec("transformer_lab.format") is not None, (
        "尚未实现 transformer_lab.format"
    )
    return importlib.import_module("transformer_lab.format")


def test_tlab_roundtrip_keeps_all_double_bits_and_shapes(tmp_path):
    fmt = format_api()
    mapping = {
        "matrix": torch.tensor([[1 / 3, 1e-200], [1e200, -0.125]], dtype=torch.float64),
        "four_dims": torch.arange(8).reshape(1, 2, 2, 2),
    }
    path = tmp_path / "case.tlab"
    fmt.write_bundle(path, mapping)
    actual = fmt.read_bundle(path)
    assert set(actual) == set(mapping)
    for name in mapping:
        assert torch.equal(actual[name], mapping[name].double())


@pytest.mark.parametrize(
    "content",
    [
        "BAD 0",
        "TLAB1 -1",
        "TLAB1 1 x 0 2",
        "TLAB1 1 x 5 1 1 1 1 1 0",
        "TLAB1 1 x 1 0",
        "TLAB1 1 x 1 -2 3 4",
        "TLAB1 1 x 1 99999999999",
        "TLAB1 1 x 1 2 1",
        "TLAB1 1 x 1 1 nan",
        "TLAB1 1 x 1 1 inf",
        "TLAB1 2 x 1 1 3 x 1 1 4",
        "TLAB1 0 extra",
        "TLAB1 1 x 1.0 1 3",
        "TLAB1 1 x 1 1.0 3",
        "TLAB1 1.0",
        "TLAB1 1 x 1 1 nope",
    ],
)
def test_tlab_reader_rejects_malformed_records(tmp_path, content):
    path = tmp_path / "bad.tlab"
    path.write_text(content)
    with pytest.raises(ValueError):
        format_api().read_bundle(path)


@pytest.mark.parametrize(
    "name,value",
    [
        ("has space", torch.ones(1)),
        ("", torch.ones(1)),
        ("scalar", torch.tensor(1)),
        ("empty", torch.ones(0)),
        ("nan", torch.tensor([float("nan")])),
        ("complex", torch.tensor([1j])),
    ],
)
def test_tlab_writer_rejects_invalid_records(tmp_path, name, value):
    with pytest.raises((ValueError, TypeError)):
        format_api().write_bundle(tmp_path / "bad.tlab", {name: value})


@pytest.mark.parametrize("name", ["a" * 1025, "字" * 342], ids=["ASCII", "UTF8"])
def test_tlab_writer_rejects_record_names_over_1024_utf8_bytes(tmp_path, name):
    with pytest.raises(ValueError):
        format_api().write_bundle(tmp_path / "long.tlab", {name: torch.ones(1)})


@pytest.mark.parametrize("name", ["a" * 1024, "字" * 341], ids=["ASCII", "UTF8"])
def test_tlab_name_boundary_roundtrip_counts_utf8_bytes(tmp_path, name):
    fmt = format_api()
    path = tmp_path / "name.tlab"
    fmt.write_bundle(path, {name: torch.ones(1)})
    assert torch.equal(fmt.read_bundle(path)[name], torch.ones(1).double())


@pytest.mark.parametrize(
    "content",
    ["TLAB1 1 " + "a" * 1025 + " 1 1 0", "TLAB1 1 x 1 1 0." + "0" * 1024],
    ids=["name", "value"],
)
def test_tlab_reader_rejects_tokens_over_1024_utf8_bytes(tmp_path, content):
    path = tmp_path / "long-token.tlab"
    path.write_text(content)
    with pytest.raises(ValueError):
        format_api().read_bundle(path)


def test_case_roundtrip_preserves_model_and_training_inputs(tmp_path):
    fmt = format_api()
    m = importlib.import_module("transformer_lab.model")
    torch.manual_seed(7)
    model = m.Transformer(m.TransformerConfig(vocab_size=8))
    src, tgt, labels = (
        torch.tensor([[3, 2, 0]]),
        torch.tensor([[1, 3]]),
        torch.tensor([[3, 2]]),
    )
    case = fmt.make_case(model, src, tgt, labels, 0.1)
    path = tmp_path / "case.tlab"
    fmt.write_bundle(path, case)
    loaded, actual_src, actual_tgt, actual_labels, lr = fmt.load_case(fmt.read_bundle(path))
    assert lr == 0.1
    assert torch.equal(src, actual_src) and torch.equal(tgt, actual_tgt)
    assert torch.equal(labels, actual_labels)
    assert "position_encoding" not in case
    torch.testing.assert_close(loaded(src, tgt), model(src, tgt), rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutation",
    ["fractional_token", "bad_weight", "missing", "extra", "lr", "config", "labels"],
)
def test_load_case_rejects_inconsistent_training_case(mutation):
    fmt = format_api()
    m = importlib.import_module("transformer_lab.model")
    model = m.Transformer(m.TransformerConfig(vocab_size=8))
    case = fmt.make_case(
        model, torch.tensor([[3, 2]]), torch.tensor([[1, 3]]), torch.tensor([[3, 2]]), 0.1
    )
    if mutation == "fractional_token":
        case["src"][0, 0] = 3.5
    elif mutation == "bad_weight":
        case["generator.weight"] = torch.zeros(1, 1)
    elif mutation == "missing":
        del case["generator.bias"]
    elif mutation == "extra":
        case["surprise"] = torch.zeros(1)
    elif mutation == "lr":
        case["learning_rate"][0] = -1
    elif mutation == "config":
        case["config"][1] = 7.5
    elif mutation == "labels":
        case["labels"].zero_()
    with pytest.raises((ValueError, TypeError)):
        fmt.load_case(case)


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "transformer_lab.cli", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_export_inspect_train_and_generate(tmp_path):
    fmt = format_api()
    case, result, checkpoint = [
        tmp_path / name for name in ["case.tlab", "result.tlab", "trained.tlab"]
    ]
    exported = run_cli("export", case, "--task", "reverse", "--seed", 7)
    assert exported.returncode == 0, exported.stderr
    model, src, tgt, labels, lr = fmt.load_case(fmt.read_bundle(case))
    assert (src == 0).any() and (labels == 0).any()
    assert torch.equal(tgt[:, 0], torch.ones(src.shape[0], dtype=torch.long))
    inspected = run_cli("inspect", case, result)
    assert inspected.returncode == 0, inspected.stderr
    output = fmt.read_bundle(result)
    assert {"logits", "loss", "encoder_output", "decoder_output"}.issubset(output)
    for name, parameter in model.named_parameters():
        assert f"grad.{name}" in output and f"updated.{name}" in output
        torch.testing.assert_close(
            output[f"updated.{name}"], parameter - lr * output[f"grad.{name}"]
        )
    trained = run_cli("train", case, checkpoint, "--steps", 12)
    assert trained.returncode == 0, trained.stderr
    later, *_ = fmt.load_case(fmt.read_bundle(checkpoint))
    m = importlib.import_module("transformer_lab.model")
    assert m.masked_cross_entropy(later(src, tgt), labels, 0) < m.masked_cross_entropy(
        model(src, tgt), labels, 0
    )
    generated = run_cli(
        "generate", checkpoint, "--bos", 1, "--eos", 2, "--max-new-tokens", 8
    )
    assert generated.returncode == 0, generated.stderr
    lines = generated.stdout.strip().splitlines()
    assert len(lines) == src.shape[0]
    assert all(line.split()[0] == "1" for line in lines)


def test_toy_export_supports_custom_padding_and_special_tokens(tmp_path):
    fmt = format_api()
    case = tmp_path / "custom.tlab"
    result = run_cli("export", case, "--pad-id", 2, "--bos", 0, "--eos", 1)
    assert result.returncode == 0, result.stderr
    model, src, tgt, labels, _ = fmt.load_case(fmt.read_bundle(case))
    assert model.config.pad_id == 2 and (src == 2).any() and (labels == 2).any()
    assert (tgt[:, 0] == 0).all()


def test_zero_step_train_preserves_every_case_record_and_loss(tmp_path):
    fmt = format_api()
    cli = importlib.import_module("transformer_lab.cli")
    case, checkpoint = tmp_path / "case.tlab", tmp_path / "zero.tlab"
    assert run_cli("export", case).returncode == 0
    initial, final = cli.train_case(case, checkpoint, steps=0)
    assert initial == final
    before, after = fmt.read_bundle(case), fmt.read_bundle(checkpoint)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_train_rejects_negative_step_count(tmp_path):
    cli = importlib.import_module("transformer_lab.cli")
    with pytest.raises(ValueError, match="steps"):
        cli.train_case(tmp_path / "unused.tlab", tmp_path / "unused-output.tlab", steps=-1)
