#include "tlab/tensor.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using tlab::Tensor;
namespace {
void near(double actual, double expected, double tolerance, const std::string &name) {
    if (std::abs(actual - expected) > tolerance || !std::isfinite(actual))
        throw std::runtime_error(name + ": expected " + std::to_string(expected) + ", got " +
                                 std::to_string(actual));
}
template <class F> void rejects(F operation, const std::string &name) {
    try {
        operation();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(name + ": expected error");
}
// 要让此测试失败，只需删去任一父节点梯度的 += 或错写一个解析导数。
template <class F> void gradient(Tensor &parameter, F objective, const std::string &name) {
    objective().backward();
    const auto analytic = parameter.grad();
    constexpr double h = 1e-6;
    for (std::size_t i = 0; i < parameter.size(); ++i) {
        const double old = parameter.data()[i];
        parameter.data()[i] = old + h;
        const double plus = objective().data()[0];
        parameter.data()[i] = old - h;
        const double minus = objective().data()[0];
        parameter.data()[i] = old;
        near(analytic[i], (plus - minus) / (2 * h), 2e-6,
             name + " grad[" + std::to_string(i) + "]");
    }
}
Tensor loss(const Tensor &logits) {
    return tlab::cross_entropy(logits, {1, 2}, 0);
}
void test_linear() {
    Tensor x({1, 2}, {2.0, 3.0}), w({1, 2}, {4.0, 5.0}), b({1}, {1.0});
    auto y = tlab::linear(x, w, b);
    near(y.data()[0], 24.0, 1e-12, "linear forward");
    y.backward();
    near(x.grad()[0], 4, 1e-12, "linear dx");
    near(w.grad()[1], 3, 1e-12, "linear dw");
    Tensor xx({1, 2, 2}, {0.2, -0.7, 0.3, 0.6});
    Tensor ww({3, 2}, {0.4, -0.2, 0.1, 0.6, -0.3, 0.7}), bb({3}, {0.2, -0.1, 0.05});
    auto objective = [&] { return loss(tlab::linear(xx, ww, bb)); };
    gradient(xx, objective, "linear x");
    gradient(ww, objective, "linear w");
    gradient(bb, objective, "linear b");
}
void test_shared_dag() {
    Tensor x({1}, {3.0});
    auto a = tlab::scale(x, 2);
    auto y = tlab::add(a, a);
    y.backward();
    near(x.grad()[0], 4, 1e-12, "shared node accumulation");
    y.backward();
    near(x.grad()[0], 4, 1e-12, "backward clears prior graph gradients");
    auto z = tlab::add(tlab::relu(x), tlab::scale(x, 3));
    z.backward();
    near(x.grad()[0], 4, 1e-12, "residual");
    Tensor negative({1}, {-2.0});
    tlab::relu(negative).backward();
    near(negative.grad()[0], 0, 0, "negative ReLU");
    std::weak_ptr<tlab::Node> weak;
    {
        auto transient = tlab::scale(x, 2);
        weak = transient.node;
    }
    if (!weak.expired())
        throw std::runtime_error("DAG retains an ownership cycle");
}
void test_embedding() {
    Tensor w({3, 3}, {0.1, 0.2, 0.3, -0.4, 0.2, 0.7, 0.5, -0.1, 0.6});
    auto objective = [&] { return loss(tlab::embedding(w, {1, 1}, 1, 2)); };
    gradient(w, objective, "embedding repeated token");
    rejects([&] { tlab::embedding(w, {3}, 1, 1); }, "embedding bad id");
}
void test_norm() {
    Tensor x({1, 2, 3}, {0.2, -0.4, 0.8, 1.2, 0.7, -0.1});
    Tensor gamma({3}, {0.8, 1.1, 1.3}), beta({3}, {0.1, -0.2, 0.4});
    auto objective = [&] { return loss(tlab::layer_norm(x, gamma, beta, 1e-5)); };
    gradient(x, objective, "norm x");
    gradient(gamma, objective, "norm gamma");
    gradient(beta, objective, "norm beta");
    Tensor identical({1, 3}, 2.0);
    auto y = tlab::layer_norm(identical, gamma, beta, 1e-5);
    for (std::size_t i = 0; i < 3; ++i)
        near(y.data()[i], beta.data()[i], 1e-12, "constant norm");
    rejects([&] { tlab::layer_norm(x, gamma, beta, 0); }, "norm eps");
}
void test_attention() {
    const double p = std::exp(1.0) / (1 + std::exp(1.0));
    auto reference =
        tlab::attention(Tensor({1, 1, 2}, {std::sqrt(2.0), 0}), Tensor({1, 2, 2}, {1, 0, 0, 0}),
                        Tensor({1, 2, 2}, {2, 4, 6, 8}), 1, {1, 1});
    near(reference.data()[0], 6 - 4 * p, 1e-12, "scaled dot-product analytic formula");
    near(reference.data()[1], 8 - 4 * p, 1e-12, "scaled dot-product weighted values");
    Tensor q({1, 2, 4}, {0.1, 0.3, -0.2, 0.4, 0.7, -0.1, 0.3, 0.2});
    Tensor k({1, 3, 4}, {0.4, 0.2, -0.1, 0.5, 0.7, 0.3, 0.2, -0.6, 0.2, -0.3, 0.5, 0.1});
    Tensor v({1, 3, 4}, {0.3, -0.4, 0.2, 0.7, 0.1, 0.5, -0.3, 0.6, 0.8, -0.2, 0.1, 0.4});
    const std::vector<unsigned char> mask{1, 0, 0, 1, 1, 0};
    auto objective = [&] { return loss(tlab::attention(q, k, v, 2, mask)); };
    auto y = tlab::attention(q, k, v, 2, mask);
    for (std::size_t i = 0; i < 4; ++i)
        near(y.data()[i], v.data()[i], 1e-12, "single allowed key");
    gradient(q, objective, "attention q");
    gradient(k, objective, "attention k");
    gradient(v, objective, "attention v");
    const double saved = k.data()[8];
    k.data()[8] = 100;
    auto yy = tlab::attention(q, k, v, 2, mask);
    k.data()[8] = saved;
    for (std::size_t i = 0; i < y.size(); ++i)
        near(y.data()[i], yy.data()[i], 1e-12, "masked key independence");
    rejects([&] { tlab::attention(q, k, v, 2, {0, 0, 0, 1, 1, 1}); }, "all masked row");
    rejects([&] { tlab::attention(q, k, v, 3, mask); }, "head divisibility");
}
void test_ce() {
    Tensor x({1, 2, 3}, {1000, 1001, 999, -1000, -999, -998});
    gradient(
        x,
        [&] {
            return tlab::cross_entropy(x, {1, 0}, 0);
        },
        "CE stable and PAD");
    auto y = tlab::cross_entropy(x, {1, 0}, 0);
    near(y.data()[0], std::log(1 + std::exp(-1.0) + std::exp(-2.0)), 1e-12, "CE stable");
    rejects([&] { tlab::cross_entropy(x, {0, 0}, 0); }, "all PAD labels");
    rejects([&] { tlab::cross_entropy(x, {1, 3}, 0); }, "label range");
}
} // namespace
int main() {
    try {
        test_linear();
        test_shared_dag();
        test_embedding();
        test_norm();
        test_attention();
        test_ce();
        rejects([] { Tensor({0, 2}); }, "zero shape");
        rejects([] { Tensor({2}, std::vector<double>{1.0}); }, "data size");
        std::cout << "all tensor forward / gradient / mask checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
