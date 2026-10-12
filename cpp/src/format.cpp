#include "tlab/format.hpp"
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tlab {
namespace {
constexpr std::size_t max_values = 10000000, max_records = 10000, max_bytes = 256 * 1024 * 1024;

void require(bool okay, const std::string &message) {
    if (!okay)
        throw std::invalid_argument(message);
}

std::string token(std::istream &input, const std::string &description) {
    std::string result;
    require(static_cast<bool>(input >> result), "TLAB1 truncated while reading " + description);
    require(result.size() <= 1024, "TLAB1 token exceeds length limit");
    return result;
}

std::size_t integer_token(const std::string &text, const std::string &description) {
    require(!text.empty(), "TLAB1 empty " + description);
    for (const auto c : text)
        require(c >= '0' && c <= '9',
                "TLAB1 " + description + " must be an unsigned decimal integer");
    std::size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    require(result.ec == std::errc{} && result.ptr == text.data() + text.size(),
            "TLAB1 " + description + " overflows integer range");
    return value;
}

double real_token(const std::string &text) {
    std::istringstream input(text);
    input.imbue(std::locale::classic());
    double value = 0;
    input >> std::noskipws >> value;
    require(input && input.eof() && std::isfinite(value),
            "TLAB1 value is not a finite decimal number: " + text);
    return value;
}

void validate_bundle(const Bundle &bundle) {
    require(bundle.size() <= max_records, "TLAB1 exceeds 10,000 record limit");
    std::size_t total = 0;
    for (const auto &pair : bundle) {
        require(!pair.first.empty() && pair.first.size() <= 1024, "TLAB1 invalid record name");
        for (const unsigned char c : pair.first)
            require(!std::isspace(c), "TLAB1 record name contains whitespace");
        require(numel(pair.second.shape()) == pair.second.size(),
                "TLAB1 tensor shape/data mismatch");
        require(pair.second.size() <= max_values - total,
                "TLAB1 exceeds 10,000,000 total value limit");
        total += pair.second.size();
        for (const double value : pair.second.data())
            require(std::isfinite(value), "TLAB1 cannot write non-finite value");
    }
}

const Tensor &record(const Bundle &bundle, const std::string &name) {
    const auto found = bundle.find(name);
    require(found != bundle.end(), "case missing record: " + name);
    return found->second;
}

std::size_t exact_integer(double value, const std::string &description) {
    require(std::isfinite(value) && value >= 0 && value <= max_values && std::floor(value) == value,
            "case " + description + " must be an exact nonnegative integer within size limit");
    return static_cast<std::size_t>(value);
}

std::vector<std::size_t> tokens(const Tensor &tensor, std::size_t vocab, const std::string &name) {
    require(tensor.shape().size() == 2, "case " + name + " must have shape [B,L]");
    std::vector<std::size_t> result;
    result.reserve(tensor.size());
    for (const double value : tensor.data()) {
        const auto id = exact_integer(value, name + " token");
        require(id < vocab, "case " + name + " token is outside vocabulary");
        result.push_back(id);
    }
    return result;
}

} // namespace

Bundle read_bundle(std::istream &input) {
    input.imbue(std::locale::classic());
    require(token(input, "magic") == "TLAB1", "file magic must be TLAB1");
    const auto count = integer_token(token(input, "record count"), "record count");
    require(count <= max_records, "TLAB1 exceeds 10,000 record limit");
    Bundle bundle;
    std::size_t total = 0;
    for (std::size_t r = 0; r < count; ++r) {
        const auto name = token(input, "record name");
        require(bundle.find(name) == bundle.end(), "TLAB1 duplicate record: " + name);
        const auto rank = integer_token(token(input, "rank"), "rank");
        require(rank >= 1 && rank <= 4, "TLAB1 rank must be 1..4");
        Shape shape;
        shape.reserve(rank);
        for (std::size_t d = 0; d < rank; ++d)
            shape.push_back(integer_token(token(input, "dimension"), "dimension"));
        const auto size = numel(shape);
        require(size <= max_values - total, "TLAB1 exceeds 10,000,000 total value limit");
        total += size;
        std::vector<double> data;
        data.reserve(size);
        for (std::size_t i = 0; i < size; ++i)
            data.push_back(real_token(token(input, "value for " + name)));
        bundle.emplace(name, Tensor(std::move(shape), std::move(data)));
    }
    std::string extra;
    require(!(input >> extra), "TLAB1 has an extra trailing token");
    require(input.eof(), "TLAB1 failed while reading input stream");
    return bundle;
}

Bundle read_bundle(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("cannot open input file: " + path);
    const auto size = input.tellg();
    require(size >= 0 && size <= static_cast<std::streamoff>(max_bytes),
            "TLAB1 file exceeds 256 MiB limit");
    input.seekg(0);
    return read_bundle(input);
}

void write_bundle(std::ostream &output, const Bundle &bundle) {
    validate_bundle(bundle);
    output.imbue(std::locale::classic());
    output << std::setprecision(17);
    output << "TLAB1\n" << bundle.size() << '\n';
    for (const auto &pair : bundle) {
        output << pair.first << ' ' << pair.second.shape().size();
        for (const auto dim : pair.second.shape())
            output << ' ' << dim;
        output << '\n';
        for (std::size_t i = 0; i < pair.second.size(); ++i) {
            if (i)
                output << ' ';
            output << pair.second.data()[i];
        }
        output << '\n';
    }
    if (!output)
        throw std::runtime_error("failed writing TLAB1 stream");
}

void write_bundle(const std::string &path, const Bundle &bundle) {
    validate_bundle(bundle);
    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("cannot open output file: " + path);
    write_bundle(output, bundle);
    output.close();
    if (!output)
        throw std::runtime_error("failed closing output file: " + path);
}

Bundle make_case(const Transformer &model, const Tensor &src, const Tensor &tgt,
                 const Tensor &labels, double learning_rate) {
    Bundle bundle;
    for (const auto &pair : model.parameters)
        bundle.emplace(pair.first, Tensor(pair.second.shape(), pair.second.data()));
    const auto &c = model.config;
    bundle.emplace(
        "config",
        Tensor({9}, std::vector<double>{
                        static_cast<double>(c.vocab_size), static_cast<double>(c.d_model),
                        static_cast<double>(c.num_heads), static_cast<double>(c.d_ff),
                        static_cast<double>(c.num_encoder_layers),
                        static_cast<double>(c.num_decoder_layers), static_cast<double>(c.max_len),
                        static_cast<double>(c.pad_id), c.eps}));
    bundle.emplace("learning_rate", Tensor({1}, std::vector<double>{learning_rate}));
    bundle.emplace("src", Tensor(src.shape(), src.data()));
    bundle.emplace("tgt", Tensor(tgt.shape(), tgt.data()));
    bundle.emplace("labels", Tensor(labels.shape(), labels.data()));
    // 写出之前采用和读取相同的契约，避免生成无法再载入的 fixture。
    (void)load_case(bundle);
    return bundle;
}

Case load_case(const Bundle &bundle) {
    validate_bundle(bundle);
    const auto &configuration = record(bundle, "config");
    require(configuration.shape() == Shape{9}, "case config must have shape [9]");
    const auto &c = configuration.data();
    Config config;
    config.vocab_size = exact_integer(c[0], "vocab_size");
    config.d_model = exact_integer(c[1], "d_model");
    config.num_heads = exact_integer(c[2], "num_heads");
    config.d_ff = exact_integer(c[3], "d_ff");
    config.num_encoder_layers = exact_integer(c[4], "num_encoder_layers");
    config.num_decoder_layers = exact_integer(c[5], "num_decoder_layers");
    config.max_len = exact_integer(c[6], "max_len");
    config.pad_id = exact_integer(c[7], "pad_id");
    config.eps = c[8];
    config.validate();
    Transformer model(config);
    require(bundle.size() == model.parameters.size() + 5, "case has missing or unexpected records");
    for (auto &pair : model.parameters) {
        const auto &source = record(bundle, pair.first);
        require(source.shape() == pair.second.shape(), "case wrong parameter shape: " + pair.first);
        // 新建叶节点，不能带入调用者可能保留的 autograd 父节点。
        pair.second = Tensor(source.shape(), source.data());
    }
    const auto &rate = record(bundle, "learning_rate");
    require(rate.shape() == Shape{1}, "case learning_rate must have shape [1]");
    require(rate.data()[0] > 0 && std::isfinite(rate.data()[0]),
            "case learning_rate must be finite and positive");
    const auto &source = record(bundle, "src");
    const auto &target = record(bundle, "tgt");
    const auto &label = record(bundle, "labels");
    auto src = tokens(source, config.vocab_size, "src"),
         tgt = tokens(target, config.vocab_size, "tgt"),
         labels = tokens(label, config.vocab_size, "labels");
    require(source.shape()[0] == target.shape()[0] && target.shape() == label.shape(),
            "case batch/target/label shapes differ");
    const auto batch = source.shape()[0], source_length = source.shape()[1],
               target_length = target.shape()[1];
    require(source_length <= config.max_len && target_length <= config.max_len,
            "case sequence exceeds config max_len");
    for (std::size_t b = 0; b < batch; ++b) {
        bool any = false;
        for (std::size_t t = 0; t < source_length; ++t)
            if (src[b * source_length + t] != config.pad_id)
                any = true;
        require(any, "case source contains all-PAD sequence");
        require(tgt[b * target_length] != config.pad_id, "case target first token is PAD");
    }
    bool any_label = false;
    for (const auto id : labels)
        if (id != config.pad_id)
            any_label = true;
    require(any_label, "case labels are all PAD");
    return {std::move(model),  std::move(src), std::move(tgt),
            std::move(labels), batch,          source_length,
            target_length,     rate.data()[0], bundle};
}

} // namespace tlab
