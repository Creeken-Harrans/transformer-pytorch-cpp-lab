# 双实现 Transformer：设计与数值契约

这是供学习者对照阅读、运行和修改的完整 encoder–decoder Transformer 核心。两个主要子目录是 `python/` 与 `cpp/`。Python 使用 PyTorch 和 uv；C++17 仅依赖标准库、CMake 和编译器，手写 CPU 张量运算及反向传播。代码以可读性和数学透明度为目标。

## 相同的模型

- 原论文风格 **Post-LN**：`LayerNorm(x + Sublayer(x))`。
- 正弦位置编码，词嵌入乘 `sqrt(d_model)`；源和目标嵌入独立，输出投影不绑定权重。
- 多头缩放点积注意力，Q/K/V/O 都有独立线性投影及 bias。
- 编码器：self-attention → norm1 → ReLU FFN → norm2。
- 解码器：causal self-attention → norm1 → cross-attention → norm2 → ReLU FFN → norm3。
- 源 PAD 在编码器及 cross-attention 的 key 位置屏蔽；目标 self-attention 同时屏蔽未来及 PAD key。PAD query 的输出允许存在，PAD label 从 loss 排除。
- 自定义 mask 使用 **True 表示允许参加 attention**。全屏蔽的 attention 行必须抛出错误；每条源序列至少有一个非 PAD token，每条目标序列第一个 token 非 PAD。
- LayerNorm 使用总体方差（除以 D），epsilon 放在平方根内部。
- token cross-entropy 在非 PAD labels 上取均值；全 PAD labels 拒绝。
- 为使核心清楚且跨语言可复现，dropout=0；优化器采用无 momentum、无 weight decay 的 SGD。浮点运算采用 float64/double。
- 支持多层、batch、不同源/目标长度；支持 teacher forcing 训练及 greedy 自回归生成。不是论文的 WMT 训练配方，也不包含 tokenizer、KV cache、CUDA kernel 或高性能库。

## Python 公共接口

包名 `transformer_lab`；模块 `model.py`, `format.py`, `cli.py`。

`TransformerConfig(vocab_size=16, d_model=8, num_heads=2, d_ff=16, num_encoder_layers=1, num_decoder_layers=1, max_len=32, pad_id=0, eps=1e-5)`；必须校验正整数、head 整除、PAD 范围、eps 有限且为正。

`Transformer(config)`：`forward(src, tgt) -> logits[B,T,V]`；`forward_with_trace(src,tgt) -> (logits, dict[str, Tensor])` 返回 `encoder_output` 与 `decoder_output`；`greedy_decode(src,bos_id,eos_id,max_new_tokens) -> tokens[B,L]`，L 包含 BOS，已结束样本后续补 PAD。

`masked_cross_entropy(logits, labels, pad_id) -> scalar Tensor`。

state_dict / C++ 参数采用完全一致的命名：

```
src_embedding.weight                      [V,D]
tgt_embedding.weight                      [V,D]
encoder.{i}.self_attn.{q,k,v,out}_proj.weight [D,D]
encoder.{i}.self_attn.{q,k,v,out}_proj.bias   [D]
encoder.{i}.norm{1,2}.{weight,bias}          [D]
encoder.{i}.ffn.linear1.weight              [F,D]
encoder.{i}.ffn.linear1.bias                [F]
encoder.{i}.ffn.linear2.weight              [D,F]
encoder.{i}.ffn.linear2.bias                [D]
decoder.{i}.{self_attn,cross_attn}.{q,k,v,out}_proj.{weight,bias}
decoder.{i}.norm{1,2,3}.{weight,bias}        [D]
decoder.{i}.ffn.linear{1,2}.{weight,bias}
generator.weight                          [V,D]
generator.bias                            [V]
```

Position encoding 是非持久 buffer，不进入权重文件。Embedding 不设置 padding_idx，以免反向传播规则产生隐含差异；PAD key/label mask 控制其梯度。

## 无依赖的互通文件 TLAB1

UTF-8、空白分隔的文本，精度为 17 个有效数字，所有数值必须有限。

```
TLAB1
record_count
name rank dim_0 ... dim_{rank-1}
value_0 value_1 ... value_{numel-1}
...repeat records...
```

rank 取 1..4；dims 必须正，values 为连续 row-major double。禁止重复名字、截断、额外尾部 token。文件读取设置合理的元素数量上限以拒绝错误尺寸。整数概念用 double 存储，但读取后必须精确整数且校验范围。

输入 case 除全部权重外，包含：

- `config` shape `[9]`：依次 V,D,H,F,N_encoder,N_decoder,max_len,pad_id,eps。
- `learning_rate` shape `[1]`，有限且大于 0。
- `src` shape `[B,S]`、`tgt` 和 `labels` shape `[B,T]`，0 <= token < V。

Python format API：`write_bundle(path, mapping[str,Tensor])`、`read_bundle(path) -> dict[str,Tensor]`；`make_case(model,src,tgt,labels,learning_rate) -> dict`；`load_case(bundle) -> (model,src,tgt,labels,learning_rate)`。

## C++ CLI

可执行文件名 `transformer_cpp`，CMake 生成于 `cpp/build/`。

```
transformer_cpp inspect CASE.tlab RESULT.tlab
transformer_cpp train CASE.tlab CHECKPOINT.tlab --steps 100
transformer_cpp generate CASE.tlab --bos 1 --eos 2 --max-new-tokens 8
```

inspect 从原始权重做 forward、backward、一次 SGD，输出 `logits`, `loss` `[1]`, `encoder_output`, `decoder_output`, `grad.<parameter_name>`, `updated.<parameter_name>`。训练保存同格式 case，替换权重，不丢弃配置与 batch；打印初始/最终 loss。generate 逐样本输出整数 token 序列，并遵循 Python 相同的 EOS/PAD 行为。

## 验收

1. Python 独立测试：attention 公式、causal 无信息泄漏、PAD 屏蔽、cross attention、位置编码、无效输入、反传、复制任务 loss 下降及生成。
2. C++ 独立测试：基础算子的中央差分梯度、DAG 梯度累加、mask 和文件/配置错误；训练与生成可运行。
3. `verify.py` 生成固定种子且有 padding 的多个 case（包含两层和不同长度）。比较每层边界结果、logits、loss、**所有参数梯度**、一次 SGD 所有权重，容差 atol=1e-9、rtol=1e-8；生成结果精确一致。抽样模型参数的有限差分为第三种校验。
4. 锁定 uv 依赖，lint/format、CTest、跨实现验证进入 GitHub Actions。
5. 中文学习指南解释形状、公式、反向传播、文件格式、阅读顺序与实验；每个子目录有独立运行指引。
6. 独立 Git 仓库，main，MIT license，忽略构建产物与环境；有清晰分阶段提交，并创建用户账户下公开远程仓库并推送。

参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)、[uv 项目与锁文件](https://docs.astral.sh/uv/concepts/projects/layout/)。
