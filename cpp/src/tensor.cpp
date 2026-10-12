#include "tlab/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace tlab {
namespace {
constexpr std::size_t max_elements = 10000000;

void require(bool condition, const char *message) {
    if (!condition)
        throw std::invalid_argument(message);
}

void same_shape(const Tensor &a, const Tensor &b) {
    require(a.shape() == b.shape(), "elementwise operation: shapes differ");
}

} // namespace

std::size_t numel(const Shape &shape) {
    require(!shape.empty() && shape.size() <= 4, "tensor rank must be 1..4");
    std::size_t result = 1;
    for (const auto dim : shape) {
        require(dim > 0, "tensor dimensions must be positive");
        require(dim <= max_elements / result, "tensor exceeds 10,000,000 element limit");
        result *= dim;
    }
    return result;
}

Tensor::Tensor(Shape shape, std::vector<double> data) : node(std::make_shared<Node>()) {
    require(numel(shape) == data.size(), "tensor data length does not match shape");
    for (const double value : data)
        require(std::isfinite(value), "tensor contains non-finite value");
    node->shape = std::move(shape);
    node->data = std::move(data);
    node->grad.assign(node->data.size(), 0.0);
}

Tensor::Tensor(Shape shape, double fill) : Tensor(shape, std::vector<double>(numel(shape), fill)) {}

void Tensor::backward() const {
    require(size() == 1, "backward requires a scalar tensor of shape [1]");
    // 1. 从输出收集可达子图；没有参与这个输出的节点不在本次 backward 中。
    // 后序 DFS 保证父节点在前。共享节点只访问一次，却可从多个子节点收到 +=。
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node *> seen;
    std::function<void(const std::shared_ptr<Node> &)> visit = [&](const auto &current) {
        if (!seen.insert(current.get()).second)
            return;
        for (const auto &parent : current->parents)
            visit(parent);
        order.push_back(current);
    };
    visit(node);

    // 2. 清零的是全部可达节点，而非只清零参数；每次 backward 都重新算梯度。
    for (auto &current : order)
        std::fill(current->grad.begin(), current->grad.end(), 0.0);
    node->grad[0] = 1.0;

    // 3. 子节点先执行。轮到父节点时，它已收到所有子路径累加的贡献。
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if ((*it)->propagate)
            (*it)->propagate(**it);
}

Tensor add(const Tensor &a, const Tensor &b) {
    same_shape(a, b);
    Tensor result(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i)
        result.data()[i] = a.data()[i] + b.data()[i];
    result.node->parents = {a.node, b.node};
    result.node->propagate = [a = a.node, b = b.node](Node &out) {
        for (std::size_t i = 0; i < out.data.size(); ++i) {
            a->grad[i] += out.grad[i];
            b->grad[i] += out.grad[i];
        }
    };
    return result;
}

Tensor scale(const Tensor &a, double factor) {
    require(std::isfinite(factor), "scale factor must be finite");
    Tensor result(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i)
        result.data()[i] = factor * a.data()[i];
    result.node->parents = {a.node};
    result.node->propagate = [a = a.node, factor](Node &out) {
        for (std::size_t i = 0; i < out.data.size(); ++i)
            a->grad[i] += factor * out.grad[i];
    };
    return result;
}

Tensor relu(const Tensor &a) {
    Tensor result(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i)
        result.data()[i] = std::max(a.data()[i], 0.0);
    result.node->parents = {a.node};
    result.node->propagate = [a = a.node](Node &out) {
        for (std::size_t i = 0; i < out.data.size(); ++i)
            if (a->data[i] > 0)
                a->grad[i] += out.grad[i];
    };
    return result;
}

Tensor linear(const Tensor &x, const Tensor &weight, const Tensor &bias) {
    require(weight.shape().size() == 2 && bias.shape().size() == 1, "linear expects W[O,I], b[O]");
    const std::size_t in = weight.shape()[1], out = weight.shape()[0];
    require(x.shape().back() == in && bias.size() == out, "linear dimensions do not match");
    const std::size_t rows = x.size() / in;
    Shape shape = x.shape();
    shape.back() = out;
    Tensor result(shape);
    // PyTorch Linear 存 W[O,I]；r 是展平的前缀行，o/i 是输出/输入通道。
    // 每行做 y_o = sum_i x_i W_oi + b_o。
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t o = 0; o < out; ++o) {
            double value = bias.data()[o];
            for (std::size_t i = 0; i < in; ++i)
                value += x.data()[r * in + i] * weight.data()[o * in + i];
            require(std::isfinite(value), "linear produced non-finite value");
            result.data()[r * out + o] = value;
        }
    result.node->parents = {x.node, weight.node, bias.node};
    result.node->propagate = [x = x.node, w = weight.node, b = bias.node, rows, in, out](Node &y) {
        // dx = dy W；dW = dy^T x；db = sum_rows dy。所有前缀维都归入 rows。
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t o = 0; o < out; ++o) {
                const double g = y.grad[r * out + o];
                b->grad[o] += g;
                for (std::size_t i = 0; i < in; ++i) {
                    x->grad[r * in + i] += g * w->data[o * in + i];
                    w->grad[o * in + i] += g * x->data[r * in + i];
                }
            }
    };
    return result;
}

Tensor embedding(const Tensor &weight, const std::vector<std::size_t> &ids, std::size_t batch,
                 std::size_t length) {
    require(weight.shape().size() == 2, "embedding weight must have shape [V,D]");
    const auto vocab = weight.shape()[0], dim = weight.shape()[1];
    require(batch > 0 && length > 0 && batch <= max_elements / length &&
                ids.size() == batch * length,
            "embedding ids must match [B,L]");
    Tensor result({batch, length, dim});
    for (std::size_t r = 0; r < ids.size(); ++r) {
        require(ids[r] < vocab, "embedding token ID out of range");
        for (std::size_t d = 0; d < dim; ++d)
            result.data()[r * dim + d] = weight.data()[ids[r] * dim + d];
    }
    result.node->parents = {weight.node};
    result.node->propagate = [w = weight.node, ids, dim](Node &out) {
        // lookup 的反向是 scatter-add。同一个词多次出现，不能覆盖它之前的梯度。
        for (std::size_t r = 0; r < ids.size(); ++r)
            for (std::size_t d = 0; d < dim; ++d)
                w->grad[ids[r] * dim + d] += out.grad[r * dim + d];
    };
    return result;
}

Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta, double eps) {
    const auto dim = x.shape().back(), rows = x.size() / dim;
    require(gamma.shape() == Shape{dim} && beta.shape() == Shape{dim},
            "LayerNorm gamma/beta must have shape [D]");
    require(std::isfinite(eps) && eps > 0, "LayerNorm epsilon must be finite and positive");
    std::vector<double> normalized(x.size()), inverse_std(rows);
    Tensor result(x.shape());
    // r 是一条前缀行，d 是其最后一维通道；每行独立求均值和总体方差。
    for (std::size_t r = 0; r < rows; ++r) {
        double mean = 0;
        for (std::size_t d = 0; d < dim; ++d)
            mean += x.data()[r * dim + d];
        mean /= static_cast<double>(dim);
        double variance = 0;
        for (std::size_t d = 0; d < dim; ++d) {
            const double centered = x.data()[r * dim + d] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(dim);
        inverse_std[r] = 1 / std::sqrt(variance + eps);
        require(std::isfinite(inverse_std[r]), "LayerNorm produced non-finite variance");
        for (std::size_t d = 0; d < dim; ++d) {
            normalized[r * dim + d] = (x.data()[r * dim + d] - mean) * inverse_std[r];
            result.data()[r * dim + d] = normalized[r * dim + d] * gamma.data()[d] + beta.data()[d];
        }
    }
    result.node->parents = {x.node, gamma.node, beta.node};
    result.node->propagate = [x = x.node, gamma = gamma.node, beta = beta.node,
                              normalized = std::move(normalized),
                              inverse_std = std::move(inverse_std), rows, dim](Node &out) {
        // xhat=(x-mu)/sqrt(var+eps)，var 除以 D（总体方差）。令 u=dy*gamma：
        // dx = inv_std * (u - mean(u) - xhat*mean(u*xhat))。
        // dgamma=sum_rows dy*xhat，dbeta=sum_rows dy。
        for (std::size_t r = 0; r < rows; ++r) {
            double sum_u = 0, sum_u_xhat = 0;
            // 1. affine 的参数梯度，并收集归一化反向所需的两个行内和。
            for (std::size_t d = 0; d < dim; ++d) {
                const auto i = r * dim + d;
                const double u = out.grad[i] * gamma->data[d];
                sum_u += u;
                sum_u_xhat += u * normalized[i];
                gamma->grad[d] += out.grad[i] * normalized[i];
                beta->grad[d] += out.grad[i];
            }
            // 2. 合并 x 的直接路径、均值路径和方差路径。
            for (std::size_t d = 0; d < dim; ++d) {
                const auto i = r * dim + d;
                x->grad[i] += inverse_std[r] *
                              (out.grad[i] * gamma->data[d] - sum_u / static_cast<double>(dim) -
                               normalized[i] * sum_u_xhat / static_cast<double>(dim));
            }
        }
    };
    return result;
}

Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, std::size_t heads,
                 const std::vector<unsigned char> &mask) {
    require(q.shape().size() == 3 && k.shape().size() == 3 && v.shape() == k.shape(),
            "attention expects Q[B,Q,D], K/V[B,K,D]");
    const auto batch = q.shape()[0], queries = q.shape()[1], dim = q.shape()[2],
               keys = k.shape()[1];
    require(k.shape()[0] == batch && k.shape()[2] == dim,
            "attention batch/model dimensions differ");
    require(heads > 0 && dim % heads == 0, "attention num_heads must divide D");
    require(mask.size() == numel({batch, queries, keys}), "attention mask must have shape [B,Q,K]");
    const auto head_dim = dim / heads;
    const double factor = 1 / std::sqrt(static_cast<double>(head_dim));
    std::vector<double> probabilities(numel({batch, heads, queries, keys}), 0.0);
    Tensor result(q.shape());
    // 物理布局仍是 [B,L,D]。把最后一维解释成 [H,d_h]，避免 transpose/copy。
    // b/h/query/key/d 分别是 batch、head、query 位置、key 位置、头内通道。
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t h = 0; h < heads; ++h)
            for (std::size_t query = 0; query < queries; ++query) {
                const auto probability_row = ((b * heads + h) * queries + query) * keys;

                // 1. S=QK^T/sqrt(d_h)：只计算允许的 key，同时找这一行最大值。
                // probabilities 暂存 score，后面在同一位置改写为概率 P。
                double maximum = -std::numeric_limits<double>::infinity();
                bool allowed = false;
                for (std::size_t key = 0; key < keys; ++key)
                    if (mask[(b * queries + query) * keys + key]) {
                        allowed = true;
                        double score = 0;
                        for (std::size_t d = 0; d < head_dim; ++d)
                            score += q.data()[(b * queries + query) * dim + h * head_dim + d] *
                                     k.data()[(b * keys + key) * dim + h * head_dim + d];
                        score *= factor;
                        require(std::isfinite(score), "attention score is non-finite");
                        probabilities[probability_row + key] = score;
                        maximum = std::max(maximum, score);
                    }
                require(allowed, "attention has an entirely masked row");

                // 2. mask 后的稳定 softmax：减最大值，求 exp 和分母。
                // 禁止位置保留初始的 0，不参与归一化。
                double denominator = 0;
                for (std::size_t key = 0; key < keys; ++key)
                    if (mask[(b * queries + query) * keys + key]) {
                        probabilities[probability_row + key] =
                            std::exp(probabilities[probability_row + key] - maximum);
                        denominator += probabilities[probability_row + key];
                    }

                // 3. 除分母得到 P，再以 P 加权 V，得到这一头的输出 O=PV。
                for (std::size_t key = 0; key < keys; ++key)
                    if (mask[(b * queries + query) * keys + key]) {
                        probabilities[probability_row + key] /= denominator;
                        for (std::size_t d = 0; d < head_dim; ++d)
                            result.data()[(b * queries + query) * dim + h * head_dim + d] +=
                                probabilities[probability_row + key] *
                                v.data()[(b * keys + key) * dim + h * head_dim + d];
                    }
            }
    result.node->parents = {q.node, k.node, v.node};
    result.node->propagate = [q = q.node, k = k.node, v = v.node, p = std::move(probabilities),
                              batch, heads, queries, keys, dim, head_dim, factor](Node &out) {
        // p 保存 forward 的 P[B,H,Lq,Lk]；每次只需一行 dP，不展开 softmax Jacobian。
        std::vector<double> probability_grad(keys);
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t h = 0; h < heads; ++h)
                for (std::size_t query = 0; query < queries; ++query) {
                    const auto probability_row = ((b * heads + h) * queries + query) * keys;
                    double mean_probability_grad = 0;

                    // 1. O=PV：dV=P^T dO，dP=dO V^T；同时求 sum_key P*dP。
                    for (std::size_t key = 0; key < keys; ++key) {
                        probability_grad[key] = 0;
                        for (std::size_t d = 0; d < head_dim; ++d) {
                            const auto query_offset =
                                           (b * queries + query) * dim + h * head_dim + d,
                                       key_offset = (b * keys + key) * dim + h * head_dim + d;
                            probability_grad[key] += out.grad[query_offset] * v->data[key_offset];
                            v->grad[key_offset] +=
                                p[probability_row + key] * out.grad[query_offset];
                        }
                        mean_probability_grad += p[probability_row + key] * probability_grad[key];
                    }

                    // 2. softmax：dS=P*(dP-sum_key P*dP)；P=0 的 mask 位置导数为 0。
                    // 3. 缩放点积：把 1/sqrt(d_h) 合入 dS，再累加 dQ=dS K、dK=dS^T Q。
                    for (std::size_t key = 0; key < keys; ++key) {
                        const double scaled_score_grad =
                            p[probability_row + key] *
                            (probability_grad[key] - mean_probability_grad) * factor;
                        for (std::size_t d = 0; d < head_dim; ++d) {
                            const auto query_offset =
                                           (b * queries + query) * dim + h * head_dim + d,
                                       key_offset = (b * keys + key) * dim + h * head_dim + d;
                            q->grad[query_offset] += scaled_score_grad * k->data[key_offset];
                            k->grad[key_offset] += scaled_score_grad * q->data[query_offset];
                        }
                    }
                }
    };
    return result;
}

Tensor cross_entropy(const Tensor &logits, const std::vector<std::size_t> &labels,
                     std::size_t pad_id) {
    require(logits.shape().size() == 3, "cross entropy logits must have shape [B,T,V]");
    const auto vocab = logits.shape().back(), rows = logits.size() / vocab;
    require(labels.size() == rows && pad_id < vocab, "cross entropy labels/PAD dimensions invalid");
    std::size_t valid_tokens = 0;
    for (const auto label : labels) {
        require(label < vocab, "cross entropy label out of range");
        if (label != pad_id)
            ++valid_tokens;
    }
    require(valid_tokens > 0, "cross entropy labels are all PAD");
    std::vector<double> probabilities(logits.size(), 0.0);
    double value = 0;
    for (std::size_t r = 0; r < rows; ++r)
        if (labels[r] != pad_id) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < vocab; ++c)
                maximum = std::max(maximum, logits.data()[r * vocab + c]);
            double denominator = 0;
            for (std::size_t c = 0; c < vocab; ++c) {
                const double e = std::exp(logits.data()[r * vocab + c] - maximum);
                probabilities[r * vocab + c] = e;
                denominator += e;
            }
            value += std::log(denominator) - (logits.data()[r * vocab + labels[r]] - maximum);
            for (std::size_t c = 0; c < vocab; ++c)
                probabilities[r * vocab + c] /= denominator;
        }
    Tensor result({1}, std::vector<double>{value / static_cast<double>(valid_tokens)});
    result.node->parents = {logits.node};
    result.node->propagate = [x = logits.node, p = std::move(probabilities), labels, pad_id,
                              valid_tokens, vocab, rows](Node &out) {
        // -log softmax 的导数是 p-onehot；先排除 PAD，再除有效 label 数，而非 B*T。
        const double factor = out.grad[0] / static_cast<double>(valid_tokens);
        for (std::size_t r = 0; r < rows; ++r)
            if (labels[r] != pad_id)
                for (std::size_t c = 0; c < vocab; ++c)
                    x->grad[r * vocab + c] +=
                        factor * (p[r * vocab + c] - (c == labels[r] ? 1.0 : 0.0));
    };
    return result;
}

} // namespace tlab
