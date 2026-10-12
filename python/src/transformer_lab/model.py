"""由线性代数原语构成的 Post-LN encoder–decoder Transformer。

约定 B=batch、S=源长度、T=目标长度、D=d_model、H=头数、K=D/H。
所有可训练参数为 float64；token 是整数。此处不调用 nn.Transformer、
MultiheadAttention 或 fused SDPA，让每个公式及 mask 的作用都能直接阅读。
"""

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor, nn

INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


@dataclass(frozen=True)
class TransformerConfig:
    """两个语言实现共享的配置；在分配模型之前拒绝无效尺寸。"""

    vocab_size: int = 16
    d_model: int = 8
    num_heads: int = 2
    d_ff: int = 16
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    max_len: int = 32
    pad_id: int = 0
    eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "d_model",
            "num_heads",
            "d_ff",
            "num_encoder_layers",
            "num_decoder_layers",
            "max_len",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if type(self.pad_id) is not int or not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id 必须为词表范围内的整数")
        if (
            isinstance(self.eps, bool)
            or not isinstance(self.eps, Real)
            or not math.isfinite(self.eps)
            or self.eps <= 0
        ):
            raise ValueError("eps 必须为有限正数")


def sinusoidal_position_encoding(max_len: int, d_model: int) -> Tensor:
    """返回 [max_len,D] 正弦表，支持奇数 D。

    PE(pos,2i)=sin(pos/10000^(2i/D))，PE(pos,2i+1)=cos(...)。
    位置 0 的偶数通道为 0、奇数通道为 1；表中没有可训练参数。
    """
    if type(max_len) is not int or type(d_model) is not int or min(max_len, d_model) <= 0:
        raise ValueError("位置表的 max_len 与 d_model 必须为正整数")
    positions = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)
    even_channels = torch.arange(0, d_model, 2, dtype=torch.float64)
    angles = positions * torch.exp(-math.log(10000.0) * even_channels / d_model)
    pe = torch.empty(max_len, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])
    return pe


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, allowed: Tensor
) -> tuple[Tensor, Tensor]:
    """计算 softmax(QKᵀ/√K)V，返回 context 与注意力概率。

    query=[B,H,T,K]、key=[B,H,S,K]、value=[B,H,S,K]。
    allowed 必须是能广播到 [B,H,T,S] 的 bool 张量，True=允许。
    softmax 沿 key 轴计算；先减去行最大值防止 exp 溢出。
    全屏蔽行的分母为 0，数学上没有概率分布，所以显式拒绝。
    """
    if any(x.ndim != 4 or not x.is_floating_point() for x in (query, key, value)):
        raise ValueError("Q/K/V 必须是四维浮点张量")
    if (
        query.shape[:2] != key.shape[:2]
        or key.shape != value.shape
        or query.shape[-1] != key.shape[-1]
        or any(size == 0 for size in (*query.shape, *key.shape))
    ):
        raise ValueError("Q/K/V 的 batch、head、key 长度或通道尺寸不匹配")
    if not (query.dtype == key.dtype == value.dtype):
        raise TypeError("Q/K/V 必须有相同 dtype")
    if not (query.device == key.device == value.device == allowed.device):
        raise ValueError("Q/K/V 和 mask 必须在同一设备")
    if allowed.dtype != torch.bool:
        raise TypeError("attention mask 必须为 bool，True 表示允许")
    score_shape = (*query.shape[:3], key.shape[2])
    try:
        allowed = torch.broadcast_to(allowed, score_shape)
    except RuntimeError as exc:
        raise ValueError("attention mask 无法广播到 [B,H,T,S]") from exc
    if not allowed.any(dim=-1).all():
        raise ValueError("attention mask 存在全屏蔽行")
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~allowed, -torch.inf)
    exponentials = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
    return probabilities @ value, probabilities


class ManualLayerNorm(nn.Module):
    """按最后一个 D 轴标准化：γ(x−μ)/sqrt(var+eps)+β。

    var=Σ(x−μ)²/D 是总体方差；eps 位于平方根内部。直接写公式使 C++
    反向传播可逐项对照。weight=γ、bias=β 的名字与 PyTorch LayerNorm 相同。
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=torch.float64))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(variance + self.eps) * self.weight + self.bias


class MultiHeadAttention(nn.Module):
    """各头在不同的可学习子空间中执行 attention，再拼接与投影。

    Linear 的权重存为 [输出,输入]，因此数学公式是 XWᵀ+b。这一布局与
    C++ 文件契约一致。Q/K/V/O 四个投影独立，均有 bias。
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, dtype=torch.float64)
        self.k_proj = nn.Linear(config.d_model, config.d_model, dtype=torch.float64)
        self.v_proj = nn.Linear(config.d_model, config.d_model, dtype=torch.float64)
        self.out_proj = nn.Linear(config.d_model, config.d_model, dtype=torch.float64)

    def _split_heads(self, x: Tensor) -> Tensor:
        # [B,L,D] -> [B,L,H,K] -> [B,H,L,K]；并未复制或合并 batch。
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, allowed: Tensor) -> Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        context, _ = scaled_dot_product_attention(q, k, v, allowed)
        # inverse transpose+reshape 将 H 个 K 维向量按 head 顺序拼回 D。
        batch, _, length, _ = context.shape
        combined = context.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.out_proj(combined)


class FeedForward(nn.Module):
    """对每个位置独立计算 FFN(x)=ReLU(xW₁ᵀ+b₁)W₂ᵀ+b₂。"""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff, dtype=torch.float64)
        self.linear2 = nn.Linear(config.d_ff, config.d_model, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """原论文 Post-LN：先残差相加，再 LayerNorm；每个子层各有一组 γ/β。"""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.norm1 = ManualLayerNorm(config.d_model, config.eps)
        self.ffn = FeedForward(config)
        self.norm2 = ManualLayerNorm(config.d_model, config.eps)

    def forward(self, x: Tensor, source_allowed: Tensor) -> Tensor:
        x = self.norm1(x + self.self_attn(x, x, x, source_allowed))
        return self.norm2(x + self.ffn(x))


class DecoderLayer(nn.Module):
    """causal self-attention -> cross-attention -> FFN，各自含残差与 Post-LN。"""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.norm1 = ManualLayerNorm(config.d_model, config.eps)
        self.cross_attn = MultiHeadAttention(config)
        self.norm2 = ManualLayerNorm(config.d_model, config.eps)
        self.ffn = FeedForward(config)
        self.norm3 = ManualLayerNorm(config.d_model, config.eps)

    def forward(
        self, x: Tensor, memory: Tensor, target_allowed: Tensor, source_allowed: Tensor
    ) -> Tensor:
        x = self.norm1(x + self.self_attn(x, x, x, target_allowed))
        # cross-attention 的 Q 来自目标，K/V 来自最后一层 encoder 的 memory。
        x = self.norm2(x + self.cross_attn(x, memory, memory, source_allowed))
        return self.norm3(x + self.ffn(x))


def validate_tokens(tokens: Tensor, name: str, config: TransformerConfig) -> None:
    """校验 token 张量的形状、整数类型、词表范围与最大位置长度。"""
    if not isinstance(tokens, Tensor) or tokens.ndim != 2 or 0 in tokens.shape:
        raise ValueError(f"{name} 必须为非空 [B,L] 张量")
    if tokens.dtype not in INTEGER_DTYPES:
        raise TypeError(f"{name} 必须使用整数 dtype")
    # PyTorch 将 Python 标量与窄整数 tensor 比较时可能把词表/PAD 上界回绕，
    # 例如 uint8 的 256 变为 0。先提升到 int64，再进行范围与特殊 token 比较。
    tokens = tokens.long()
    if tokens.shape[1] > config.max_len:
        raise ValueError(f"{name} 长度超过 max_len")
    if (tokens < 0).any() or (tokens >= config.vocab_size).any():
        raise ValueError(f"{name} token 越出词表范围")


class Transformer(nn.Module):
    """完整 encoder–decoder；forward 输出尚未 softmax 的 [B,T,V] logits。

    训练时 tgt 是右移的目标（BOS,y₁,...），labels 是 (y₁,y₂,...,EOS)。
    并行计算整个 tgt 的因果 attention，就是 teacher forcing：每个位置的
    正确历史由数据提供，模型无法看到未来 labels。
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        # 不用 padding_idx：PAD query 可以有输出；key/label mask 明确控制影响。
        self.src_embedding = nn.Embedding(
            config.vocab_size, config.d_model, dtype=torch.float64
        )
        self.tgt_embedding = nn.Embedding(
            config.vocab_size, config.d_model, dtype=torch.float64
        )
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(config.max_len, config.d_model),
            persistent=False,
        )
        self.encoder = nn.ModuleList(
            EncoderLayer(config) for _ in range(config.num_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_decoder_layers)
        )
        # 独立输出权重，不与目标 embedding 绑定。
        self.generator = nn.Linear(config.d_model, config.vocab_size, dtype=torch.float64)

    def _validate_source(self, src: Tensor) -> None:
        validate_tokens(src, "src", self.config)
        if src.device != self.src_embedding.weight.device:
            raise ValueError("src 与模型必须在同一设备")
        if not (src.long() != self.config.pad_id).any(dim=1).all():
            raise ValueError("每条 src 至少需要一个非 PAD token")

    def _validate_inputs(self, src: Tensor, tgt: Tensor) -> None:
        self._validate_source(src)
        validate_tokens(tgt, "tgt", self.config)
        if src.shape[0] != tgt.shape[0] or src.device != tgt.device:
            raise ValueError("src/tgt 的 batch 或设备不匹配")
        if (tgt.long()[:, 0] == self.config.pad_id).any():
            raise ValueError("每条 tgt 的第一个 token 必须非 PAD")

    def _embed(self, tokens: Tensor, embedding: nn.Embedding) -> Tensor:
        return (
            embedding(tokens.long()) * math.sqrt(self.config.d_model)
            + self.position_encoding[: tokens.shape[1]]
        )

    def _encode(self, src: Tensor, trace: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        src = src.long()
        # [B,1,1,S] 在 head 和 query 轴广播：屏蔽 PAD key，保留 PAD query。
        source_allowed = (src != self.config.pad_id)[:, None, None, :]
        memory = self._embed(src, self.src_embedding)
        for index, layer in enumerate(self.encoder):
            memory = layer(memory, source_allowed)
            trace[f"encoder.{index}.output"] = memory
        trace["encoder_output"] = memory
        return memory, source_allowed

    def _decode(
        self, tgt: Tensor, memory: Tensor, source_allowed: Tensor, trace: dict[str, Tensor]
    ) -> Tensor:
        tgt = tgt.long()
        length = tgt.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=tgt.device).tril()
        target_keys = (tgt != self.config.pad_id)[:, None, None, :]
        target_allowed = causal[None, None, :, :] & target_keys
        hidden = self._embed(tgt, self.tgt_embedding)
        for index, layer in enumerate(self.decoder):
            hidden = layer(hidden, memory, target_allowed, source_allowed)
            trace[f"decoder.{index}.output"] = hidden
        trace["decoder_output"] = hidden
        return self.generator(hidden)

    def forward_with_trace(
        self, src: Tensor, tgt: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """返回 logits 与各层边界张量；trace 保留 autograd 图供学习与数值验收。"""
        self._validate_inputs(src, tgt)
        trace: dict[str, Tensor] = {}
        memory, source_allowed = self._encode(src, trace)
        return self._decode(tgt, memory, source_allowed, trace), trace

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.forward_with_trace(src, tgt)[0]

    @torch.no_grad()
    def greedy_decode(
        self, src: Tensor, bos_id: int, eos_id: int, max_new_tokens: int
    ) -> Tensor:
        """逐步 argmax 生成 [B,L]，L 包含 BOS；全 batch 结束便提前停止。

        首次产生 EOS 的样本标记 finished，此后追加 PAD。未结束样本继续使用
        自己的预测历史。只缓存固定 encoder memory，不缓存 decoder K/V。
        max_new_tokens=0 返回仅有 BOS 的序列。临时 eval 后恢复原训练模式。
        """
        self._validate_source(src)
        for name, token in (("bos_id", bos_id), ("eos_id", eos_id)):
            if type(token) is not int or not 0 <= token < self.config.vocab_size:
                raise ValueError(f"{name} 必须为词表范围内的整数")
            if token == self.config.pad_id:
                raise ValueError(f"{name} 必须不同于 PAD")
        if bos_id == eos_id:
            raise ValueError("BOS 和 EOS 必须不同")
        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 0
            or max_new_tokens + 1 > self.config.max_len
        ):
            raise ValueError("max_new_tokens 必须非负，且包括 BOS 后不超过 max_len")
        was_training = self.training
        self.eval()
        try:
            memory, source_allowed = self._encode(src, {})
            tokens = torch.full(
                (src.shape[0], 1), bos_id, dtype=torch.long, device=src.device
            )
            finished = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
            for _ in range(max_new_tokens):
                next_token = self._decode(tokens, memory, source_allowed, {})[:, -1].argmax(
                    -1
                )
                next_token = torch.where(finished, self.config.pad_id, next_token)
                tokens = torch.cat((tokens, next_token[:, None]), dim=1)
                finished |= next_token == eos_id
                if finished.all():
                    break
            return tokens
        finally:
            self.train(was_training)


def masked_cross_entropy(logits: Tensor, labels: Tensor, pad_id: int) -> Tensor:
    """只在非 PAD labels 上平均 −log softmax(logits)[label]。

    沿 V 轴减去最大值：logsumexp(z)=m+log Σ exp(z−m)。忽略 PAD 的位置
    不贡献 loss 或梯度；分母是有效 token 数，不是 B*T。
    """
    if logits.ndim != 3 or 0 in logits.shape or not logits.is_floating_point():
        raise ValueError("logits 必须为非空 [B,T,V] 浮点张量")
    if labels.shape != logits.shape[:2] or labels.device != logits.device:
        raise ValueError("labels 必须为匹配 logits 的 [B,T]，并在同一设备")
    if labels.dtype not in INTEGER_DTYPES:
        raise TypeError("labels 必须为整数 dtype")
    labels = labels.long()
    if type(pad_id) is not int or not 0 <= pad_id < logits.shape[-1]:
        raise ValueError("pad_id 越出词表范围")
    if (labels < 0).any() or (labels >= logits.shape[-1]).any():
        raise ValueError("labels token 越出词表范围")
    valid = labels != pad_id
    if not valid.any():
        raise ValueError("labels 不能全部是 PAD")
    if not torch.isfinite(logits).all():
        raise ValueError("logits 必须全部有限")
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_probabilities = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()
    losses = -log_probabilities.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    return losses[valid].mean()
