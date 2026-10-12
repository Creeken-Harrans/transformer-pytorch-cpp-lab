"""小型可复现实验 CLI：导出 case、检查梯度、SGD 训练、greedy 生成。"""

import argparse
from pathlib import Path

import torch
from torch import Tensor

from .format import CONFIG_NAMES, load_case, make_case, read_bundle, write_bundle
from .model import Transformer, TransformerConfig, masked_cross_entropy


def toy_batch(
    config: TransformerConfig,
    task: str = "copy",
    batch_size: int = 2,
    sequence_length: int = 4,
    seed: int = 7,
    bos_id: int = 1,
    eos_id: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """生成带 padding 的复制/逆序任务，无需外部数据集。

    sequence_length 指最长内容长度；各样本长度从该值递减到 1，再循环。
    源=内容+EOS，tgt=BOS+正确输出，labels=正确输出+EOS，三者尾部补 PAD。
    因为 labels 相对 tgt 左移一位，模型在位置 t 学的是下一个 token。
    """
    if task not in {"copy", "reverse"}:
        raise ValueError("task 必须是 copy 或 reverse")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if type(sequence_length) is not int or not 1 <= sequence_length < config.max_len:
        raise ValueError("sequence_length 必须为正，内容加特殊 token 后不超过 max_len")
    if any(
        type(token) is not int or not 0 <= token < config.vocab_size
        for token in (bos_id, eos_id)
    ):
        raise ValueError("BOS/EOS 必须为词表范围内的整数")
    if len({config.pad_id, bos_id, eos_id}) != 3:
        raise ValueError("PAD、BOS 和 EOS 必须互不相同；可用 --bos/--eos 指定")
    payload_ids = [
        token
        for token in range(config.vocab_size)
        if token not in {config.pad_id, bos_id, eos_id}
    ]
    if not payload_ids:
        raise ValueError("词表至少需要一个非特殊内容 token")
    generator = torch.Generator().manual_seed(seed)
    shape = (batch_size, sequence_length + 1)
    src, tgt, labels = (
        torch.full(shape, config.pad_id, dtype=torch.long) for _ in range(3)
    )
    content_vocabulary = torch.tensor(payload_ids, dtype=torch.long)
    for batch_index in range(batch_size):
        content_length = sequence_length - (batch_index % sequence_length)
        content = content_vocabulary[
            torch.randint(len(payload_ids), (content_length,), generator=generator)
        ]
        answer = content if task == "copy" else content.flip(0)
        src[batch_index, :content_length] = content
        src[batch_index, content_length] = eos_id
        tgt[batch_index, 0] = bos_id
        tgt[batch_index, 1 : content_length + 1] = answer
        labels[batch_index, :content_length] = answer
        labels[batch_index, content_length] = eos_id
    return src, tgt, labels


def inspect_case(case_path: str | Path, result_path: str | Path) -> None:
    """一次 forward/backward/SGD；结果与 C++ inspect 逐记录比较。"""
    model, src, tgt, labels, learning_rate = load_case(read_bundle(case_path))
    logits, trace = model.forward_with_trace(src, tgt)
    loss = masked_cross_entropy(logits, labels, model.config.pad_id)
    loss.backward()
    result = {"logits": logits.detach(), "loss": loss.detach().reshape(1)}
    result.update({name: value.detach() for name, value in trace.items()})
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for name, parameter in model.named_parameters():
        result[f"grad.{name}"] = parameter.grad.detach().clone()
    optimizer.step()
    result.update(
        {
            f"updated.{name}": parameter.detach()
            for name, parameter in model.named_parameters()
        }
    )
    write_bundle(result_path, result)


def train_case(
    case_path: str | Path, checkpoint_path: str | Path, steps: int
) -> tuple[float, float]:
    """重复训练 case 中的同一个 batch，保存相同格式的可继续训练 checkpoint。

    这是观察公式和优化过程的过拟合实验；真实训练还需要数据切分、mini-batch、
    验证集及更完整的优化配方。SGD 在这里是 θ←θ−η∂L/∂θ，无 momentum。
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps 必须为非负整数")
    model, src, tgt, labels, learning_rate = load_case(read_bundle(case_path))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial = masked_cross_entropy(model(src, tgt), labels, model.config.pad_id).item()
    for _ in range(steps):
        # 每一步先清空旧梯度；backward 会在本步 DAG 中累加共享节点梯度。
        optimizer.zero_grad(set_to_none=True)
        loss = masked_cross_entropy(model(src, tgt), labels, model.config.pad_id)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = masked_cross_entropy(model(src, tgt), labels, model.config.pad_id).item()
    write_bundle(checkpoint_path, make_case(model, src, tgt, labels, learning_rate))
    return initial, final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="导出固定种子的复制/逆序训练 case")
    export.add_argument("case", type=Path)
    export.add_argument("--task", choices=("copy", "reverse"), default="copy")
    export.add_argument("--seed", type=int, default=7)
    export.add_argument("--batch-size", type=int, default=2)
    export.add_argument("--sequence-length", type=int, default=4)
    export.add_argument("--learning-rate", type=float, default=0.1)
    export.add_argument("--bos", type=int, default=1)
    export.add_argument("--eos", type=int, default=2)
    defaults = TransformerConfig()
    for name in CONFIG_NAMES:
        export.add_argument(
            "--" + name.replace("_", "-"),
            type=float if name == "eps" else int,
            default=getattr(defaults, name),
        )
    inspect = commands.add_parser(
        "inspect", help="导出 logits/loss/层边界/全部梯度/一步 SGD"
    )
    inspect.add_argument("case", type=Path)
    inspect.add_argument("result", type=Path)
    train = commands.add_parser("train", help="重复 SGD 训练同一 batch 并保存 checkpoint")
    train.add_argument("case", type=Path)
    train.add_argument("checkpoint", type=Path)
    train.add_argument("--steps", type=int, default=100)
    generate = commands.add_parser("generate", help="逐行输出含 BOS 的 greedy token 序列")
    generate.add_argument("case", type=Path)
    generate.add_argument("--bos", type=int, default=1)
    generate.add_argument("--eos", type=int, default=2)
    generate.add_argument("--max-new-tokens", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    # 小矩阵由单线程执行更快，也更便于对照 C++ CPU 循环。仅 CLI 设置线程数。
    torch.set_num_threads(1)
    try:
        if args.command == "export":
            config = TransformerConfig(
                **{name: getattr(args, name) for name in CONFIG_NAMES}
            )
            torch.manual_seed(args.seed)
            model = Transformer(config)
            src, tgt, labels = toy_batch(
                config,
                args.task,
                args.batch_size,
                args.sequence_length,
                args.seed,
                args.bos,
                args.eos,
            )
            write_bundle(args.case, make_case(model, src, tgt, labels, args.learning_rate))
            print(f"exported={args.case} task={args.task} seed={args.seed}")
        elif args.command == "inspect":
            inspect_case(args.case, args.result)
            print(f"result={args.result}")
        elif args.command == "train":
            initial, final = train_case(args.case, args.checkpoint, args.steps)
            print(
                f"initial_loss={initial:.12g}\nfinal_loss={final:.12g}\nsteps={args.steps}"
            )
        else:
            model, src, _, _, _ = load_case(read_bundle(args.case))
            for sequence in model.greedy_decode(
                src, args.bos, args.eos, args.max_new_tokens
            ):
                print(" ".join(str(token) for token in sequence.tolist()))
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
