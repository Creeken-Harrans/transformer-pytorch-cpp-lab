#include "tlab/format.hpp"
#include <charconv>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {
void usage(std::ostream &out) {
    out << "Usage:\n"
        << "  transformer_cpp inspect CASE.tlab RESULT.tlab\n"
        << "  transformer_cpp train CASE.tlab CHECKPOINT.tlab --steps N\n"
        << "  transformer_cpp generate CASE.tlab --bos ID --eos ID --max-new-tokens N\n";
}

std::size_t unsigned_option(const std::string &text, const std::string &name) {
    if (text.empty())
        throw std::invalid_argument(name + " requires a nonnegative integer");
    for (const char c : text)
        if (c < '0' || c > '9')
            throw std::invalid_argument(name + " requires a nonnegative integer");
    std::size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument(name + " integer is out of range");
    return value;
}

std::map<std::string, std::size_t> options(int argc, char **argv, int first) {
    if ((argc - first) % 2 != 0)
        throw std::invalid_argument("every option needs a value");
    std::map<std::string, std::size_t> result;
    for (int i = first; i < argc; i += 2) {
        const std::string name = argv[i];
        if (!result.emplace(name, unsigned_option(argv[i + 1], name)).second)
            throw std::invalid_argument("duplicate option: " + name);
    }
    return result;
}

void inspect(tlab::Case &fixture, const std::string &path) {
    auto prediction = fixture.model.forward(fixture.src, fixture.batch, fixture.source_length,
                                            fixture.tgt, fixture.target_length);
    auto loss = tlab::cross_entropy(prediction.logits, fixture.labels, fixture.model.config.pad_id);
    loss.backward();
    tlab::Bundle output = prediction.traces;
    output.emplace("logits", prediction.logits);
    output.emplace("loss", loss);
    // 显式新建 Tensor，导出本次梯度数值；普通 Tensor 拷贝只会共享参数节点。
    for (const auto &pair : fixture.model.parameters)
        output.emplace("grad." + pair.first, tlab::Tensor(pair.second.shape(), pair.second.grad()));
    fixture.model.sgd(fixture.learning_rate);
    for (const auto &pair : fixture.model.parameters)
        output.emplace("updated." + pair.first,
                       tlab::Tensor(pair.second.shape(), pair.second.data()));
    tlab::write_bundle(path, output);
}

double objective(const tlab::Case &fixture) {
    auto prediction = fixture.model.forward(fixture.src, fixture.batch, fixture.source_length,
                                            fixture.tgt, fixture.target_length);
    return tlab::cross_entropy(prediction.logits, fixture.labels, fixture.model.config.pad_id)
        .data()[0];
}

void train(tlab::Case &fixture, const std::string &path, std::size_t steps) {
    const double initial = objective(fixture);
    for (std::size_t step = 0; step < steps; ++step) {
        // 每步创建新 DAG；forward/loss 局部变量在本轮结束销毁，只保留参数叶节点。
        auto prediction = fixture.model.forward(fixture.src, fixture.batch, fixture.source_length,
                                                fixture.tgt, fixture.target_length);
        auto loss =
            tlab::cross_entropy(prediction.logits, fixture.labels, fixture.model.config.pad_id);
        loss.backward();
        fixture.model.sgd(fixture.learning_rate);
    }
    const double final = objective(fixture);
    // 原 bundle 保留 config、学习率和全部 batch；只替换权重。
    for (const auto &pair : fixture.model.parameters)
        fixture.original.at(pair.first) = pair.second;
    tlab::write_bundle(path, fixture.original);
    std::cout << std::setprecision(17) << "steps=" << steps << " initial_loss=" << initial
              << " final_loss=" << final << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            usage(std::cout);
            return 0;
        }
        if (argc < 3) {
            usage(std::cerr);
            return 1;
        }
        const std::string command = argv[1];
        if (command != "inspect" && command != "train" && command != "generate")
            throw std::invalid_argument("unknown command: " + command);
        auto fixture = tlab::load_case(tlab::read_bundle(std::string(argv[2])));
        if (command == "inspect") {
            if (argc != 4)
                throw std::invalid_argument("inspect requires CASE and RESULT paths");
            inspect(fixture, argv[3]);
        } else if (command == "train") {
            if (argc != 4 && argc != 6)
                throw std::invalid_argument(
                    "train requires CASE, CHECKPOINT and optional --steps N");
            const auto opts = options(argc, argv, 4);
            std::size_t steps = 100;
            if (!opts.empty()) {
                if (opts.size() != 1 || !opts.count("--steps"))
                    throw std::invalid_argument("train accepts only --steps N");
                steps = opts.at("--steps");
            }
            train(fixture, argv[3], steps);
        } else {
            const auto opts = options(argc, argv, 3);
            for (const auto &pair : opts)
                if (pair.first != "--bos" && pair.first != "--eos" &&
                    pair.first != "--max-new-tokens")
                    throw std::invalid_argument("unknown generate option: " + pair.first);
            const auto bos = opts.count("--bos") ? opts.at("--bos") : 1;
            const auto eos = opts.count("--eos") ? opts.at("--eos") : 2;
            const auto budget = opts.count("--max-new-tokens") ? opts.at("--max-new-tokens") : 8;
            const auto rows = fixture.model.generate(fixture.src, fixture.batch,
                                                     fixture.source_length, bos, eos, budget);
            for (const auto &row : rows) {
                for (std::size_t i = 0; i < row.size(); ++i) {
                    if (i)
                        std::cout << ' ';
                    std::cout << row[i];
                }
                std::cout << '\n';
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
