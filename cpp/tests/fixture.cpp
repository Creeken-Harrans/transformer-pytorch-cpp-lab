#include "tlab/format.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

void check(bool okay, const std::string &message) {
    if (!okay)
        throw std::runtime_error(message);
}
double loss(const tlab::Case &fixture) {
    auto prediction = fixture.model.forward(fixture.src, fixture.batch, fixture.source_length,
                                            fixture.tgt, fixture.target_length);
    return tlab::cross_entropy(prediction.logits, fixture.labels, fixture.model.config.pad_id)
        .data()[0];
}
int main(int argc, char **argv) {
    try {
        check(argc >= 3, "fixture arguments");
        const std::string mode = argv[1];
        if (mode == "create" || mode == "eos") {
            tlab::Transformer model(tlab::Config{}, 19);
            if (mode == "eos") {
                auto &bias = model.parameters.at("generator.bias").data();
                std::fill(bias.begin(), bias.end(), -1000);
                bias[2] = 1000;
            }
            auto fixture = tlab::make_case(model, tlab::Tensor({2, 3}, {3, 4, 0, 5, 6, 7}),
                                           tlab::Tensor({2, 3}, {1, 3, 4, 1, 5, 6}),
                                           tlab::Tensor({2, 3}, {3, 4, 2, 5, 6, 2}), 0.1);
            tlab::write_bundle(argv[2], fixture);
            return 0;
        }
        check(argc == 4, "fixture comparison arguments");
        const auto source = tlab::read_bundle(std::string(argv[2]));
        const auto output = tlab::read_bundle(std::string(argv[3]));
        auto fixture = tlab::load_case(source);
        if (mode == "inspect") {
            check(output.at("logits").shape() == tlab::Shape{2, 3, 16}, "inspect logits shape");
            check(output.size() == 2 * fixture.model.parameters.size() + 6,
                  "inspect complete gradient/update/trace record set");
            check(std::abs(output.at("loss").data()[0] - loss(fixture)) < 1e-12,
                  "inspect original loss");
            check(output.at("grad.generator.bias").data().size() == 16, "inspect bias gradient");
            bool changed = false;
            for (const auto &pair : fixture.model.parameters) {
                const auto &grad = output.at("grad." + pair.first);
                const auto &updated = output.at("updated." + pair.first);
                check(grad.shape() == pair.second.shape() && updated.shape() == pair.second.shape(),
                      "gradient/update shape");
                for (std::size_t i = 0; i < pair.second.size(); ++i) {
                    check(std::abs(updated.data()[i] - (pair.second.data()[i] -
                                                        fixture.learning_rate * grad.data()[i])) <
                              1e-12,
                          "inspect SGD relation");
                    changed = changed || updated.data()[i] != pair.second.data()[i];
                }
            }
            check(changed, "inspect makes a real SGD update");
        } else if (mode == "same") {
            check(output.size() == source.size(), "checkpoint record count");
            for (const auto &pair : source)
                check(output.at(pair.first).data() == pair.second.data(),
                      "zero-step checkpoint preserves every record");
        } else if (mode == "trained") {
            auto after = tlab::load_case(output);
            check(loss(after) < loss(fixture), "CLI training decreases loss");
            for (const auto *key : {"config", "learning_rate", "src", "tgt", "labels"})
                check(output.at(key).data() == source.at(key).data(),
                      "training preserves batch/configuration");
        } else
            throw std::runtime_error("unknown fixture mode");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
