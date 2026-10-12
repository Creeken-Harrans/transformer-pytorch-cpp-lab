"""学习入口的验收：真实的一步训练，以及可读、可保存的验证结果。"""

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def verifier(monkeypatch):
    # verify.py 是根目录的独立验收脚本，由同一个 uv 环境运行。
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("verify")


@pytest.fixture
def report():
    return {
        "atol": 1e-9,
        "rtol": 1e-8,
        "cases": {
            "baseline": {
                "records": 98,
                "elements": 4193,
                "max_abs_error": 1.23e-15,
                "finite_difference": {"parameters_sampled": 8, "max_abs_error": 4.38e-11},
                "official_pytorch_layer_max_abs_error": 6.66e-16,
            }
        },
        "training_and_generation": {
            "steps": 12,
            "initial_loss": 3.29,
            "final_loss": 2.01,
            "checkpoint_max_abs_error": 2.22e-16,
            "generation_rows": 2,
            "mixed_eos_padding_verified": True,
        },
        "invalid_cases_rejected_by_both": 12,
    }


def test_default_report_shows_cases_and_error_magnitude(verifier, report, capsys):
    verifier.emit_report(report)
    stdout = capsys.readouterr().out
    assert "baseline" in stdout and "1.23e-15" in stdout
    assert "SGD" in stdout and "PAD" in stdout
    assert "4.38e-11" in stdout and "6.66e-16" in stdout
    assert not stdout.lstrip().startswith("{")


def test_json_stdout_and_saved_report_are_parseable(verifier, report, capsys, tmp_path):
    destination = tmp_path / "nested" / "report.json"
    verifier.emit_report(report, json_stdout=True, json_output=destination)
    assert json.loads(capsys.readouterr().out) == report
    assert json.loads(destination.read_text(encoding="utf-8")) == report


def test_copy_checkpoint_report_also_honors_json_output(verifier, capsys, tmp_path):
    report = {"checkpoint.tlab": {"rows_copied_exactly": 2, "loss": 0.011}}
    destination = tmp_path / "copy.json"
    verifier.emit_report(report, copy_mode=True, json_output=destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    stdout = capsys.readouterr().out
    assert "checkpoint.tlab" in stdout and "0.011" in stdout


def test_walkthrough_runs_one_real_sgd_step():
    result = subprocess.run(
        [sys.executable, str(ROOT / "python" / "examples" / "walkthrough.py")],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "valid_labels=5" in result.stdout
    assert "logits: [2, 3, 8]" in result.stdout
    values = {
        name: float(value)
        for name, value in re.findall(
            r"(loss_before|loss_after|generator_grad_norm)=([0-9.eE+-]+)", result.stdout
        )
    }
    assert values["generator_grad_norm"] > 0
    assert 0 < values["loss_after"] < values["loss_before"]


def test_plain_make_lists_learning_commands():
    result = subprocess.run(
        ["make"], cwd=ROOT, text=True, capture_output=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ("walkthrough", "demo", "check"))
