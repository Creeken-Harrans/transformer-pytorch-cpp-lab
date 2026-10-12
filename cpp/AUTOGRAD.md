# 从 C++ 循环读懂 Transformer 的反向传播

`src/tensor.cpp` 把每个算子写成两部分：计算 `data`，再给输出节点登记
`parents` 与 `propagate` 回调。下文的上游梯度记为 `g` 或 `dY`。
对应代码中的 `+=` 表示把这个子节点的贡献加到父节点上。

## 1. DAG 和链式法则

若 `z = a+a`，`a=2x`，则 `dz/dx=4`。同一个 `a` 出现两次，但只有一个节点。
反向先执行 z 的回调，使 `a.grad += 1` 两次，再执行 a 的回调，使 `x.grad += 2*2`。
若把 `+=` 改为 `=`，重复边、残差边、共享 encoder memory 和重复词都会出错。

`backward()` 先做后序 DFS，得到「父节点在前、输出在后」的拓扑列表。
它把本图全部节点的梯度清零，设置标量输出的梯度为 1，然后逆序执行回调。
visited 集合确保共享节点只执行一次，执行时各个子节点已经积累完上游贡献。

节点只向祖先保存 `shared_ptr`。回调从 `Node& out` 参数读取输出梯度，
不捕获输出自身。训练循环离开作用域后，中间节点和保存的 softmax 结果被释放；
模型的参数叶节点仍由 `parameters` 持有。

## 2. Linear 和 embedding

把 `[B,L,I]` 的前两维合并为行数 `R=B*L`：

$$
Y_{ro}=\sum_i X_{ri}W_{oi}+b_o.
$$

对元素直接用链式法则：

$$
dX_{ri}=\sum_o dY_{ro}W_{oi},\qquad
dW_{oi}=\sum_r dY_{ro}X_{ri},\qquad
db_o=\sum_r dY_{ro}.
$$

C++ 用 `(r*I+i)` 与 `(o*I+i)` 索引。三个累加都在同一组 r/o/i 循环内完成。
没有预设 batch=1 或 sequence length 固定。

embedding 是按词 ID 取权重行：`Y[r,d]=W[id[r],d]`。
所以 `dW[id[r],d] += dY[r,d]`，其余行不增加。
连续多个 token 指向同一词时需要 scatter-add。源与目标 embedding 是不同参数；
本模型不设置 padding_idx，PAD 的屏蔽发生在 attention key 与 loss，而不是 lookup。

## 3. LayerNorm

每个 token 独立在 D 个通道上归一化：

$$
\mu=\frac1D\sum_d x_d,\quad
v=\frac1D\sum_d(x_d-\mu)^2,\quad
r=(v+\epsilon)^{-1/2},\quad
\hat x_d=(x_d-\mu)r,\quad y_d=\gamma_d\hat x_d+\beta_d.
$$

先反传 affine：令 `u_d = g_d*gamma_d`，并跨 token 累加
`dgamma_d += g_d*xhat_d`，`dbeta_d += g_d`。
接着分别沿直接的 x、均值和方差三条路径反传，可合并为：

$$
dx_d=r\left[u_d-\frac1D\sum_j u_j
              -\hat x_d\frac1D\sum_j u_j\hat x_j\right].
$$

回调保存 `xhat` 与 `r`，每行先算 `sum(u)` 和 `sum(u*xhat)`，再计算 dx。
这里方差除以 D，而非 D−1；epsilon 位于平方根内部。
若一行所有值相同，归一化输出为零，affine 输出恰为 beta；epsilon 仍保证导数有限。

## 4. 注意力的 forward

一头的输入形状：Q 为 `[Q,d_h]`，K/V 为 `[K,d_h]`，Q 和 K 允许不同长度。
每头各自计算：

$$
S=QK^T/\sqrt{d_h},\quad P=\operatorname{softmax}(S),\quad O=PV.
$$

每个 query 只在 mask=1 的 key 上归一化。先减去允许位置的最大值再 exp，
防止指数溢出。禁止位置的 P 直接为零，不参与最大值或分母。
若某一行没有允许的 key，直接报错，不制造 NaN。

C++ 把最后一维 D 看成 `[H,d_h]`，第 h 头的输入索引为
`(b*L+t)*D+h*d_h+d`。P 的物理形状为 `[B,H,Q,K]`。
这是一整个自定义算子，因此不必为每次标量乘法建立节点；公式仍与逐算子 autograd 相同。

## 5. 注意力的 backward

由 `O=PV` 得：

$$
dV=P^T dO,\qquad dP=dO V^T.
$$

softmax 的 Jacobian 是
`dP_j/dS_l = P_j*(delta_jl-P_l)`。把上游 dP 与它相乘，得到每行：

$$
dS_j=P_j\left(dP_j-\sum_l P_l dP_l\right).
$$

所以无须保存 `[K,K]` 的 Jacobian，只需一条长度 K 的 `dp` 和一个加权和。
mask 位置 P=0，故 dS=0；它不会给 Q/K 或 V 引入梯度。
最后沿缩放点积返回：

$$
dQ=dS K/\sqrt{d_h},\qquad dK=dS^T Q/\sqrt{d_h}.
$$

代码把 `1/sqrt(d_h)` 合入 `ds`，然后同时加到 q.grad 和 k.grad。
模型外层的 Q/K/V/O 四个 projection 都通过普通 Linear 回传。
self-attention 的三条投影最终回到同一个输入，cross-attention 的 K/V 都回到
encoder memory；DAG 的累加规则自动处理这些共享路径。

## 6. Masked cross-entropy

设有效 token 数为 N，不包括 label=PAD 的行。对 logits z：

$$
L=\frac1N\sum_{r:\,label_r\ne PAD}
  \left[\log\sum_c e^{z_{rc}}-z_{r,label_r}\right].
$$

计算时用最大值 m 改写成
`log(sum_c exp(z_c-m))-(z_label-m)`，避免大正数溢出或大负数下溢。
对有效行：

$$
\frac{\partial L}{\partial z_{rc}}=
\frac{p_{rc}-\mathbf 1[c=label_r]}N.
$$

PAD 行的梯度是零。N=0 时拒绝输入。最后 SGD 更新
`theta -= learning_rate*theta.grad`；更新前检查所有候选值是否有限。

## 7. 数值验证

`tests/test_main.cpp` 先保存解析梯度，再独立扰动每个输入元素：

$$
g_i^{FD}=\frac{L(\theta_i+h)-L(\theta_i-h)}{2h},\quad h=10^{-6}.
$$

每次计算中央差分都重新建立 forward 图，避免使用旧 cache。
覆盖 Linear 的 x/W/b、embedding 重复索引、LayerNorm 的 x/gamma/beta、
attention 的 Q/K/V、稳定 CE 与 PAD 行；ReLU 测试选在零点外。
有限差分核验导数与其 forward 一致，手算注意力 fixture 和根目录跨语言
验证进一步检查 forward 公式本身。
