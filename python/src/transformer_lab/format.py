"""TLAB1：Python/C++ 共享的纯文本 float64 权重、输入与验收结果格式。

每条记录是名字、rank、dims、row-major 数值。17 位有效数字可以无损往返
IEEE-754 double。这里不用 pickle，C++ 也无需第三方序列化库。
"""

import math
import re
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import torch
from torch import Tensor

from .model import Transformer, TransformerConfig, masked_cross_entropy

MAX_RECORDS = 10_000
MAX_ELEMENTS = 10_000_000
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOKEN_BYTES = 1024
CONFIG_NAMES = tuple(field.name for field in fields(TransformerConfig))
CASE_FIELDS = {"config", "learning_rate", "src", "tgt", "labels"}


def _validate_record(name: str, tensor: Tensor) -> None:
    if not isinstance(name, str) or not name or any(c.isspace() for c in name):
        raise ValueError("TLAB1 记录名字不能为空或含空白")
    if len(name.encode("utf-8")) > MAX_TOKEN_BYTES:
        raise ValueError("TLAB1 记录名字超过 1024 UTF-8 字节")
    if not isinstance(tensor, Tensor) or tensor.is_complex():
        raise TypeError(f"{name}: TLAB1 只支持实数 Tensor")
    if not 1 <= tensor.ndim <= 4 or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name}: rank 必须在 1..4，且各维度为正")
    if tensor.numel() > MAX_ELEMENTS:
        raise ValueError(f"{name}: 元素数量超过 TLAB1 上限")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name}: TLAB1 数值必须全部有限")


def write_bundle(path: str | Path, mapping: Mapping[str, Tensor]) -> None:
    """用 17 位有效数字写出全部记录；先校验，避免产生半个无效文件。"""
    if len(mapping) > MAX_RECORDS:
        raise ValueError("TLAB1 记录数量超过上限")
    total = 0
    for name, tensor in mapping.items():
        _validate_record(name, tensor)
        total += tensor.numel()
    if total > MAX_ELEMENTS:
        raise ValueError("TLAB1 总元素数量超过上限")
    with Path(path).open("w", encoding="utf-8", newline="\n") as file:
        file.write(f"TLAB1\n{len(mapping)}\n")
        for name, tensor in mapping.items():
            shape = " ".join(str(size) for size in tensor.shape)
            file.write(f"{name} {tensor.ndim} {shape}\n")
            values = (
                tensor.detach().to(device="cpu", dtype=torch.float64).contiguous().flatten()
            )
            file.write(" ".join(format(value, ".17g") for value in values.tolist()) + "\n")


def read_bundle(path: str | Path) -> dict[str, Tensor]:
    """严格读取 TLAB1；拒绝重复、截断、非有限值、过大尺寸与尾部 token。"""
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("TLAB1 文件超过 256 MiB 上限")
    with path.open(encoding="utf-8") as file:
        # 逐行分词，不必一次构造整个文件的字符串列表。
        tokens = (token for line in file for token in line.split())

        def take() -> str:
            try:
                token = next(tokens)
            except StopIteration as exc:
                raise ValueError("TLAB1 文件截断") from exc
            if len(token.encode("utf-8")) > MAX_TOKEN_BYTES:
                raise ValueError("TLAB1 token 超过 1024 UTF-8 字节")
            return token

        def integer(description: str, minimum: int, maximum: int) -> int:
            word = take()
            if not re.fullmatch(r"[0-9]+", word):
                raise ValueError(f"{description} 必须用整数文本表示")
            value = int(word)
            if not minimum <= value <= maximum:
                raise ValueError(f"{description} 越出允许范围")
            return value

        if take() != "TLAB1":
            raise ValueError("TLAB1 文件头不匹配")
        count = integer("记录数", 0, MAX_RECORDS)
        result: dict[str, Tensor] = {}
        total = 0
        for _ in range(count):
            name = take()
            if name in result:
                raise ValueError(f"TLAB1 重复名字: {name}")
            rank = integer("rank", 1, 4)
            shape = tuple(integer("维度", 1, MAX_ELEMENTS) for _ in range(rank))
            numel = math.prod(shape)
            total += numel
            if total > MAX_ELEMENTS:
                raise ValueError("TLAB1 总元素数量超过上限")
            values = []
            for _ in range(numel):
                try:
                    value = float(take())
                except ValueError as exc:
                    raise ValueError(f"{name}: 无效或截断的浮点数") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{name}: TLAB1 数值必须全部有限")
                values.append(value)
            result[name] = torch.tensor(values, dtype=torch.float64).reshape(shape)
        if next(tokens, None) is not None:
            raise ValueError("TLAB1 有额外尾部 token")
        return result


def _check_learning_rate(learning_rate: float) -> None:
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, int | float)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate 必须为有限正数")


def make_case(
    model: Transformer, src: Tensor, tgt: Tensor, labels: Tensor, learning_rate: float
) -> dict[str, Tensor]:
    """导出独立 case：全部模型参数、9 项配置、batch 与 SGD 学习率。

    clone 使 bundle 成为快照：之后 optimizer.step() 不会悄悄改写导出的权重。
    token 在文件中用 double 存储，load_case 会验证精确整数后再转为 long。
    """
    _check_learning_rate(learning_rate)
    with torch.no_grad():
        # 用公开的 forward/loss 校验全部输入，包含 PAD 与 label 有效计数。
        masked_cross_entropy(model(src, tgt), labels, model.config.pad_id)
    bundle = {
        name: tensor.detach().to(device="cpu", dtype=torch.float64).clone()
        for name, tensor in model.state_dict().items()
    }
    bundle.update(
        config=torch.tensor(
            [getattr(model.config, name) for name in CONFIG_NAMES], dtype=torch.float64
        ),
        learning_rate=torch.tensor([learning_rate], dtype=torch.float64),
        src=src.detach().cpu().double().clone(),
        tgt=tgt.detach().cpu().double().clone(),
        labels=labels.detach().cpu().double().clone(),
    )
    return bundle


def _exact_integers(tensor: Tensor, name: str) -> Tensor:
    if not torch.isfinite(tensor).all() or not (tensor == tensor.trunc()).all():
        raise ValueError(f"{name} 必须包含精确整数")
    # double 转 int64 超范围可能回绕，先拒绝。词表范围另由模型校验。
    if (tensor.abs() >= 2**63).any():
        raise ValueError(f"{name} 整数超出 int64 范围")
    return tensor.long()


def _check_model_size(config: TransformerConfig) -> None:
    # 先计算参数数量再分配内存，损坏的配置不能绕过文件元素数保护。
    v, d, f = config.vocab_size, config.d_model, config.d_ff
    count = (
        3 * v * d
        + v
        + config.num_encoder_layers * (4 * d * d + 9 * d + 2 * d * f + f)
        + config.num_decoder_layers * (8 * d * d + 15 * d + 2 * d * f + f)
    )
    if count > MAX_ELEMENTS or config.max_len * d > MAX_ELEMENTS:
        raise ValueError("配置的参数或位置表大小超过 TLAB1 上限")


def load_case(
    bundle: Mapping[str, Tensor],
) -> tuple[Transformer, Tensor, Tensor, Tensor, float]:
    """从 case 创建 CPU float64 模型；严格要求全部权重的名字与 shape 匹配。"""
    if not CASE_FIELDS.issubset(bundle):
        raise ValueError(f"case 缺少字段: {sorted(CASE_FIELDS - bundle.keys())}")
    if len(bundle) > MAX_RECORDS:
        raise ValueError("case 记录数超过上限")
    total = 0
    for name, value in bundle.items():
        _validate_record(name, value)
        if value.device.type != "cpu":
            raise ValueError("load_case 要求 CPU bundle")
        total += value.numel()
    if total > MAX_ELEMENTS:
        raise ValueError("case 总元素数量超过上限")
    if bundle["config"].shape != (9,):
        raise ValueError("config 必须为 [9]")
    integers = _exact_integers(bundle["config"][:8], "config").tolist()
    config = TransformerConfig(*integers, eps=bundle["config"][8].item())
    _check_model_size(config)
    model = Transformer(config)
    parameters = model.state_dict()
    expected = set(parameters) | CASE_FIELDS
    if set(bundle) != expected:
        raise ValueError(
            f"case 权重集合不匹配；缺少={sorted(expected - bundle.keys())}，"
            f"额外={sorted(bundle.keys() - expected)}"
        )
    for name, parameter in parameters.items():
        if bundle[name].shape != parameter.shape:
            raise ValueError(f"{name}: 权重 shape 不匹配")
    model.load_state_dict({name: bundle[name] for name in parameters}, strict=True)
    if bundle["learning_rate"].shape != (1,):
        raise ValueError("learning_rate 必须为 [1]")
    learning_rate = bundle["learning_rate"].item()
    _check_learning_rate(learning_rate)
    src, tgt, labels = (
        _exact_integers(bundle[name], name) for name in ("src", "tgt", "labels")
    )
    with torch.no_grad():
        masked_cross_entropy(model(src, tgt), labels, config.pad_id)
    return model, src, tgt, labels, learning_rate
