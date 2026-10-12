#include "tlab/format.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
void check(bool okay, const std::string &message) {
    if (!okay)
        throw std::runtime_error(message);
}
template <class F> void rejects(F action, const std::string &name) {
    try {
        action();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(name + ": expected rejection");
}
void test_model() {
    tlab::Config config;
    config.vocab_size = 8;
    config.d_model = 6;
    config.num_heads = 2;
    config.d_ff = 12;
    config.num_encoder_layers = 2;
    config.num_decoder_layers = 2;
    config.max_len = 10;
    tlab::Transformer model(config, 17);
    check(model.parameters.size() == 88, "2-layer model has 88 independent parameter tensors");
    const std::vector<std::size_t> src{3, 4, 0, 5, 6, 7}, tgt{1, 3, 4, 1, 5, 6},
        labels{3, 4, 2, 5, 6, 2};
    auto output = model.forward(src, 2, 3, tgt, 3);
    check(output.logits.shape() == tlab::Shape{2, 3, 8}, "model logits shape");
    check(output.traces.size() == 6, "all four layer boundaries plus final outputs");
    auto changed_target = tgt;
    changed_target[2] = 7;
    const auto causal = model.forward(src, 2, 3, changed_target, 3);
    for (std::size_t i = 0; i < 8; ++i)
        check(std::abs(causal.logits.data()[i] - output.logits.data()[i]) < 1e-12,
              "future target cannot affect first logits");
    auto &source_weights = model.parameters.at("src_embedding.weight").data();
    const double pad_weight = source_weights[0];
    source_weights[0] += 100;
    const auto masked = model.forward(src, 2, 3, tgt, 3);
    source_weights[0] = pad_weight;
    for (std::size_t i = 0; i < output.logits.size(); ++i)
        check(std::abs(masked.logits.data()[i] - output.logits.data()[i]) < 1e-12,
              "PAD key does not affect decoder output");
    auto loss = tlab::cross_entropy(output.logits, labels, 0);
    const double initial = loss.data()[0];
    loss.backward();
    for (const auto &pair : model.parameters)
        for (const double gradient : pair.second.grad())
            check(std::isfinite(gradient), "model parameter gradient is finite");
    for (int i = 0; i < 70; ++i) {
        auto train = model.forward(src, 2, 3, tgt, 3);
        tlab::cross_entropy(train.logits, labels, 0).backward();
        model.sgd(0.1);
    }
    const double final =
        tlab::cross_entropy(model.forward(src, 2, 3, tgt, 3).logits, labels, 0).data()[0];
    check(final < initial * 0.35, "SGD substantially reduces teacher-forcing loss");
    // 强制 EOS 为唯一最大 logit。若 finished 分支错误，长度/第二个 token 会改变。
    auto &bias = model.parameters.at("generator.bias").data();
    std::fill(bias.begin(), bias.end(), -1000);
    bias[2] = 1000;
    auto generated = model.generate(src, 2, 3, 1, 2, 5);
    check(generated == std::vector<std::vector<std::size_t>>{{1, 2}, {1, 2}},
          "EOS ends generation");
    check(model.generate(src, 2, 3, 1, 2, 0) == std::vector<std::vector<std::size_t>>{{1}, {1}},
          "zero budget emits BOS");
    rejects([&] { model.forward({0, 0, 0}, 1, 3, {1}, 1); }, "all PAD source");
    rejects([&] { model.forward({3}, 1, 1, {0}, 1); }, "initial target PAD");
    rejects([&] { model.forward({8}, 1, 1, {1}, 1); }, "source range");
    rejects([&] { model.generate({3}, 1, 1, 1, 2, 10); }, "position budget");
    rejects([&] { model.generate({3}, 1, 1, 1, 1, 3); }, "BOS/EOS must differ");
    config.num_heads = 4;
    rejects([&] { tlab::Transformer bad(config); }, "head divisibility");
    config.num_heads = 2;
    config.eps = 0;
    rejects([&] { tlab::Transformer bad(config); }, "config epsilon");
    config = tlab::Config{};
    config.vocab_size = 500000;
    rejects([&] { config.validate(); }, "total parameter limit before allocation");
}
void test_format() {
    tlab::Bundle bundle;
    bundle.emplace("test", tlab::Tensor({2, 2}, {0.1, 0.2, 0.3, 0.4}));
    std::stringstream text;
    tlab::write_bundle(text, bundle);
    auto loaded = tlab::read_bundle(text);
    check(loaded.at("test").data() == bundle.at("test").data(),
          "17-digit serialization round trip");
    const std::vector<std::string> invalid{"BAD 0",
                                           "TLAB1 -1",
                                           "TLAB1 1 x 0",
                                           "TLAB1 1 x 1 0",
                                           "TLAB1 1 x 1 2 1",
                                           "TLAB1 1 x 1 1 nan",
                                           "TLAB1 1 x 1 1 inf",
                                           "TLAB1 0 tail",
                                           "TLAB1 2 x 1 1 1 x 1 1 2",
                                           "TLAB1 1 x 1 10000001",
                                           "TLAB1 1 x 1 18446744073709551616",
                                           "TLAB1 1 x 1 1 1junk"};
    for (const auto &bad : invalid)
        rejects(
            [&] {
                std::stringstream in(bad);
                tlab::read_bundle(in);
            },
            "malformed TLAB1");
    tlab::Transformer model(tlab::Config{}, 11);
    auto fixture = tlab::make_case(model, tlab::Tensor({1, 3}, {3, 4, 0}),
                                   tlab::Tensor({1, 2}, {1, 3}), tlab::Tensor({1, 2}, {3, 2}), 0.1);
    const double snapshot = fixture.at("generator.weight").data()[0];
    model.parameters.at("generator.weight").data()[0] += 1;
    check(fixture.at("generator.weight").data()[0] == snapshot,
          "make_case snapshots model parameters");
    model.parameters.at("generator.weight").data()[0] = snapshot;
    auto restored = tlab::load_case(fixture);
    check(restored.src == std::vector<std::size_t>{3, 4, 0}, "case keeps data");
    check(restored.model.parameters.at("generator.weight").data() ==
              model.parameters.at("generator.weight").data(),
          "case keeps parameters");
    fixture.at("src").data()[0] = 3.5;
    rejects([&] { tlab::load_case(fixture); }, "fractional token");
    fixture.at("src").data()[0] = 3;
    fixture.at("config").data()[0] = 16.5;
    rejects([&] { tlab::load_case(fixture); }, "fractional config");
    fixture.at("config").data()[0] = 16;
    fixture.erase("generator.weight");
    rejects([&] { tlab::load_case(fixture); }, "missing parameter");
}
} // namespace
int main() {
    try {
        test_model();
        test_format();
        std::cout << "model/training/generation/format checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
