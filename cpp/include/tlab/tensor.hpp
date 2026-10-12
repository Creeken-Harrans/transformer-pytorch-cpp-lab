#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace tlab {
using Shape = std::vector<std::size_t>;
struct Node {
    Shape shape;
    std::vector<double> data, grad;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(Node &)> propagate;
};
// shared_ptr 只指向父节点；回调从参数接收输出节点，因此没有所有权环。
struct Tensor {
    std::shared_ptr<Node> node;
    Tensor(Shape shape, std::vector<double> data);
    explicit Tensor(Shape shape, double fill = 0.0);
    const Shape &shape() const {
        return node->shape;
    }
    const std::vector<double> &data() const {
        return node->data;
    }
    std::vector<double> &data() {
        return node->data;
    }
    const std::vector<double> &grad() const {
        return node->grad;
    }
    std::size_t size() const {
        return node->data.size();
    }
    void backward() const;
};
std::size_t numel(const Shape &shape);
Tensor add(const Tensor &a, const Tensor &b);
Tensor scale(const Tensor &a, double factor);
Tensor relu(const Tensor &a);
Tensor linear(const Tensor &x, const Tensor &weight, const Tensor &bias);
Tensor embedding(const Tensor &weight, const std::vector<std::size_t> &ids, std::size_t batch,
                 std::size_t length);
Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta, double eps);
// mask [B,Q,K]：1=允许。头维 D/H 连续存储在最后一维。
Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, std::size_t heads,
                 const std::vector<unsigned char> &mask);
Tensor cross_entropy(const Tensor &logits, const std::vector<std::size_t> &labels,
                     std::size_t pad_id);
} // namespace tlab
