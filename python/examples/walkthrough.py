"""用一个真实 batch 观察 Transformer 的数据流与一次参数更新。

从仓库根目录运行 make walkthrough，或在 python/ 内运行：
uv run --locked python examples/walkthrough.py
"""

import math

import torch
from torch import Tensor
from transformer_lab.model import (
    Transformer,
    TransformerConfig,
    masked_cross_entropy,
    scaled_dot_product_attention,
)


def show_matrix(name: str, matrix: Tensor) -> None:
    """将张量转为普通数值展示；mask 用整数，概率保留四位小数。"""
    print(f"{name}: {list(matrix.shape)}")
    for row in matrix.detach().tolist():
        cells = (
            [f"{value:8.4f}" for value in row]
            if matrix.is_floating_point()
            else [str(int(value)) for value in row]
        )
        print("  " + " ".join(cells))


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = TransformerConfig(vocab_size=8, d_model=8, num_heads=2, d_ff=16)
    model = Transformer(config)

    # PAD=0、BOS=1、EOS=2。内容分别长 2、1；目标使用真实前缀作为输入。
    # tgt 的每个位置预测 labels 的同一位置，所以 labels 比 tgt 左移一个 token。
    src = torch.tensor([[3, 4, 2, 0], [5, 2, 0, 0]])
    tgt = torch.tensor([[1, 3, 4], [1, 5, 0]])
    labels = torch.tensor([[3, 4, 2], [5, 2, 0]])
    print("1. 输入：src=内容+EOS；tgt=BOS+内容；labels=内容+EOS")
    for name, tokens in (("src", src), ("tgt", tgt), ("labels", labels)):
        print(f"{name}: {list(tokens.shape)} {tokens.tolist()}")
    print(f"valid_labels={int((labels != config.pad_id).sum())}")

    # mask 的最后一个轴是 key。PAD query 仍可计算，PAD label 不参与 loss。
    source_allowed = (src != config.pad_id)[:, None, None, :]
    target_length = tgt.shape[1]
    causal = torch.ones(target_length, target_length, dtype=torch.bool).tril()
    target_allowed = causal[None, None] & (tgt != config.pad_id)[:, None, None, :]
    print("\n2. mask：1=允许，0=屏蔽；列对应 key，行对应 query")
    for batch_index in range(src.shape[0]):
        allowed_keys = source_allowed[batch_index, 0, 0].int().tolist()
        print(f"batch {batch_index} source keys: {allowed_keys}")
        show_matrix(
            f"batch {batch_index} target mask", target_allowed[batch_index, 0].int()
        )

    # 单独展开第一层的第一头；用模型的实际投影权重计算，而非示意矩阵。
    with torch.no_grad():
        embedded = model.src_embedding(src) * math.sqrt(config.d_model)
        embedded = embedded + model.position_encoding[: src.shape[1]]
        attention = model.encoder[0].self_attn
        batch_size, source_length, _ = embedded.shape
        head_dim = config.d_model // config.num_heads

        def split_heads(projected: Tensor) -> Tensor:
            return projected.reshape(
                batch_size, source_length, config.num_heads, head_dim
            ).transpose(1, 2)

        query = split_heads(attention.q_proj(embedded))
        key = split_heads(attention.k_proj(embedded))
        value = split_heads(attention.v_proj(embedded))
        _, probabilities = scaled_dot_product_attention(query, key, value, source_allowed)
    print("\n3. encoder 第 0 层、第 0 头的实际 attention 概率")
    for batch_index in range(src.shape[0]):
        show_matrix(f"batch {batch_index} probabilities", probabilities[batch_index, 0])
    print("PAD key 列为 0；每个 query 行的概率之和为 1。")

    print("\n4. 完整 forward：查看层边界与 logits")
    logits, trace = model.forward_with_trace(src, tgt)
    for name, hidden in trace.items():
        print(f"{name}: {list(hidden.shape)}")
    print(f"logits: {list(logits.shape)}")

    # 所有 5 个有效 labels 共用平均 loss；backward 先生成梯度，step 再更新参数。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    loss = masked_cross_entropy(logits, labels, config.pad_id)
    loss.backward()
    assert model.generator.weight.grad is not None
    print("\n5. backward → SGD：θ ← θ − 0.05 × gradient")
    print(f"loss_before={loss.item():.12g}")
    print(f"generator_grad_norm={model.generator.weight.grad.norm().item():.12g}")
    optimizer.step()
    with torch.no_grad():
        updated_loss = masked_cross_entropy(model(src, tgt), labels, config.pad_id)
    print(f"loss_after={updated_loss.item():.12g}")
    print("逐项对照公式与两端源码：见仓库根目录 LEARNING.md。")


if __name__ == "__main__":
    main()
