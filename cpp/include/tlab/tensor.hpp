#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace tlab {
using Shape = std::vector<std::size_t>;

// 一个 Node 保存叶张量或一次算子的结果；data/grad 都按 shape 连续 row-major 存储。
// parents 保留祖先的寿命；propagate 从参数接收输出节点，不能捕获输出自身。
struct Node {
    Shape shape;
    std::vector<double> data, grad;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(Node &)> propagate;
};

// Tensor 是 Node 的共享句柄：拷贝会共享 data、grad 和计算图，不会复制数值。
// 数值快照需显式构造 Tensor(original.shape(), original.data())，得到新叶节点。
struct Tensor {
    std::shared_ptr<Node> node;

    // rank 为 1..4、各维为正、元素数 <= 10,000,000；初始数值必须有限。
    Tensor(Shape shape, std::vector<double> data);
    explicit Tensor(Shape shape, double fill = 0.0);
    const Shape &shape() const {
        return node->shape;
    }
    const std::vector<double> &data() const {
        return node->data;
    }
    // forward 到 backward 之间不得改动图中的数值：部分回调会读取父节点 data。
    // 原地更新叶节点后，旧图的缓存已经过期；下一步必须重新 forward。
    std::vector<double> &data() {
        return node->data;
    }
    const std::vector<double> &grad() const {
        return node->grad;
    }
    std::size_t size() const {
        return node->data.size();
    }
    // 仅接受一个元素的输出（numel==1），不要求 rank==1。
    // 清零所有可达节点的 grad，种下输出梯度 1，再逆序累加各条路径的贡献。
    // 每次调用重新计算本图的梯度；不同 backward 调用之间不会累积梯度。
    void backward() const;
};

std::size_t numel(const Shape &shape);

// 两个输入须同形状，无广播；scale/relu 保留输入形状。
Tensor add(const Tensor &a, const Tensor &b);
Tensor scale(const Tensor &a, double factor);
Tensor relu(const Tensor &a);

// x[...,I], weight[O,I], bias[O] -> y[...,O]。任意前缀维合并为 rows。
Tensor linear(const Tensor &x, const Tensor &weight, const Tensor &bias);

// weight[V,D]，ids 是展平的 [B,L] -> y[B,L,D]；重复 ID 的梯度累加。
Tensor embedding(const Tensor &weight, const std::vector<std::size_t> &ids, std::size_t batch,
                 std::size_t length);

// x[...,D], gamma[D], beta[D] -> y[...,D]；仅归一化最后一维。
// 方差除以 D，eps 位于 sqrt 内部，且必须有限、为正。
Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta, double eps);

// Q[B,Lq,D], K/V[B,Lk,D] -> O[B,Lq,D]；Lq、Lk 可以不同。
// mask 是展平的 [B,Lq,Lk]，非零（True）=允许；所有 head 共享这个 mask。
// heads 须整除 D；头维 D/heads 连续存于最后一维；全屏蔽 query 行报错。
Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, std::size_t heads,
                 const std::vector<unsigned char> &mask);

// logits[B,T,V]，labels 是展平的 [B,T] -> loss[1]。
// 排除 label==pad_id 的行并对其余行取均值；全 PAD labels 报错。
Tensor cross_entropy(const Tensor &logits, const std::vector<std::size_t> &labels,
                     std::size_t pad_id);
} // namespace tlab
