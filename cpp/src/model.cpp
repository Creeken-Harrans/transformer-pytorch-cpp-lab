#include "tlab/model.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace tlab {
namespace {
void require(bool okay, const std::string &message) {
    if (!okay)
        throw std::invalid_argument(message);
}

} // namespace

void Config::validate() const {
    require(vocab_size > 0 && d_model > 0 && num_heads > 0 && d_ff > 0 && num_encoder_layers > 0 &&
                num_decoder_layers > 0 && max_len > 0,
            "config dimensions/layer counts must be positive");
    require(d_model % num_heads == 0, "config num_heads must divide d_model");
    require(pad_id < vocab_size, "config pad_id is outside vocabulary");
    require(std::isfinite(eps) && eps > 0, "config epsilon must be finite and positive");
    // 在分配之前校验尺寸。这个教学实现限制单个张量 1000 万元素、层数 1000。
    numel({vocab_size, d_model});
    numel({d_model, d_model});
    numel({d_ff, d_model});
    numel({max_len, d_model});
    require(num_encoder_layers <= 1000 && num_decoder_layers <= 1000,
            "config layer count exceeds 1000");
    const long double D = static_cast<long double>(d_model), F = static_cast<long double>(d_ff);
    const long double total = 3 * static_cast<long double>(vocab_size) * D + vocab_size +
                              num_encoder_layers * (4 * D * D + 2 * F * D + 9 * D + F) +
                              num_decoder_layers * (8 * D * D + 2 * F * D + 15 * D + F);
    require(total <= 10000000, "config exceeds 10,000,000 total parameter elements");
}

Transformer::Transformer(Config configuration, std::uint32_t seed) : config(configuration) {
    config.validate();
    std::mt19937 generator(seed);
    auto zeros = [&](const std::string &name, Shape shape, double value = 0) {
        parameters.emplace(name, Tensor(std::move(shape), value));
    };
    auto weight = [&](const std::string &name, Shape shape) {
        Tensor tensor(shape);
        const double limit = std::sqrt(6.0 / static_cast<double>(shape[0] + shape[1]));
        std::uniform_real_distribution<double> distribution(-limit, limit);
        for (double &value : tensor.data())
            value = distribution(generator);
        parameters.emplace(name, std::move(tensor));
    };
    auto linear_parameters = [&](const std::string &prefix, std::size_t in, std::size_t out) {
        weight(prefix + ".weight", {out, in});
        zeros(prefix + ".bias", {out});
    };
    auto norm_parameters = [&](const std::string &prefix) {
        zeros(prefix + ".weight", {config.d_model}, 1);
        zeros(prefix + ".bias", {config.d_model});
    };
    auto attention_parameters = [&](const std::string &prefix) {
        for (const auto *projection : {"q", "k", "v", "out"})
            linear_parameters(prefix + "." + projection + "_proj", config.d_model, config.d_model);
    };
    auto feed_forward_parameters = [&](const std::string &prefix) {
        linear_parameters(prefix + ".linear1", config.d_model, config.d_ff);
        linear_parameters(prefix + ".linear2", config.d_ff, config.d_model);
    };
    weight("src_embedding.weight", {config.vocab_size, config.d_model});
    weight("tgt_embedding.weight", {config.vocab_size, config.d_model});
    for (std::size_t i = 0; i < config.num_encoder_layers; ++i) {
        const auto prefix = "encoder." + std::to_string(i);
        attention_parameters(prefix + ".self_attn");
        norm_parameters(prefix + ".norm1");
        norm_parameters(prefix + ".norm2");
        feed_forward_parameters(prefix + ".ffn");
    }
    for (std::size_t i = 0; i < config.num_decoder_layers; ++i) {
        const auto prefix = "decoder." + std::to_string(i);
        attention_parameters(prefix + ".self_attn");
        attention_parameters(prefix + ".cross_attn");
        norm_parameters(prefix + ".norm1");
        norm_parameters(prefix + ".norm2");
        norm_parameters(prefix + ".norm3");
        feed_forward_parameters(prefix + ".ffn");
    }
    linear_parameters("generator", config.d_model, config.vocab_size);
}

void Transformer::validate_tokens(const std::vector<std::size_t> &ids, std::size_t batch,
                                  std::size_t length, bool source) const {
    require(length <= config.max_len, "sequence length exceeds config max_len");
    require(ids.size() == numel({batch, length}), "token count does not match [B,L]");
    for (const auto id : ids)
        require(id < config.vocab_size, "token ID is outside vocabulary");
    for (std::size_t b = 0; b < batch; ++b) {
        if (source) {
            bool any = false;
            for (std::size_t t = 0; t < length; ++t)
                if (ids[b * length + t] != config.pad_id)
                    any = true;
            require(any, "source contains an all-PAD sequence");
        } else
            require(ids[b * length] != config.pad_id, "target first token must not be PAD");
    }
}

Tensor Transformer::embed(const std::vector<std::size_t> &ids, std::size_t batch,
                          std::size_t length, const std::string &prefix) const {
    auto embeddings = scale(embedding(parameters.at(prefix + ".weight"), ids, batch, length),
                            std::sqrt(static_cast<double>(config.d_model)));
    Tensor positions({batch, length, config.d_model});
    // PE(pos,2i)=sin(pos/10000^(2i/D))，PE(pos,2i+1)=cos(...)。
    // 奇数 D 的最后一列只取 sin；这里 d/2 是整数除法。
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t pos = 0; pos < length; ++pos)
            for (std::size_t d = 0; d < config.d_model; ++d) {
                const double angle =
                    static_cast<double>(pos) *
                    std::exp(-std::log(10000.0) * static_cast<double>(2 * (d / 2)) /
                             static_cast<double>(config.d_model));
                positions.data()[(b * length + pos) * config.d_model + d] =
                    (d % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
    return add(embeddings, positions);
}

Tensor Transformer::multi_head(const Tensor &query, const Tensor &memory,
                               const std::vector<unsigned char> &mask,
                               const std::string &prefix) const {
    auto projection = [&](const Tensor &x, const std::string &name) {
        return linear(x, parameters.at(prefix + "." + name + "_proj.weight"),
                      parameters.at(prefix + "." + name + "_proj.bias"));
    };
    // self-attention 中 query/memory 共享节点；cross-attention 中它们的长度可以不同。
    auto q = projection(query, "q"), k = projection(memory, "k"), v = projection(memory, "v");
    return projection(attention(q, k, v, config.num_heads, mask), "out");
}

Tensor Transformer::norm(const Tensor &x, const std::string &prefix) const {
    return layer_norm(x, parameters.at(prefix + ".weight"), parameters.at(prefix + ".bias"),
                      config.eps);
}

Tensor Transformer::ffn(const Tensor &x, const std::string &prefix) const {
    auto hidden = relu(linear(x, parameters.at(prefix + ".linear1.weight"),
                              parameters.at(prefix + ".linear1.bias")));
    return linear(hidden, parameters.at(prefix + ".linear2.weight"),
                  parameters.at(prefix + ".linear2.bias"));
}

Forward Transformer::forward(const std::vector<std::size_t> &src, std::size_t batch,
                             std::size_t source_length, const std::vector<std::size_t> &tgt,
                             std::size_t target_length) const {
    validate_tokens(src, batch, source_length, true);
    validate_tokens(tgt, batch, target_length, false);

    // 1. mask[B,Lq,Lk] 的最后一维指向 key；PAD query 仍允许产生输出。
    // encoder/cross 屏蔽源 PAD key；decoder 再屏蔽未来的目标 key。
    std::vector<unsigned char> encoder_mask(numel({batch, source_length, source_length}));
    std::vector<unsigned char> decoder_mask(numel({batch, target_length, target_length}));
    std::vector<unsigned char> cross_mask(numel({batch, target_length, source_length}));
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t q = 0; q < source_length; ++q)
            for (std::size_t k = 0; k < source_length; ++k)
                encoder_mask[(b * source_length + q) * source_length + k] =
                    (src[b * source_length + k] != config.pad_id);
        for (std::size_t q = 0; q < target_length; ++q) {
            for (std::size_t k = 0; k < target_length; ++k)
                decoder_mask[(b * target_length + q) * target_length + k] =
                    (k <= q && tgt[b * target_length + k] != config.pad_id);
            for (std::size_t k = 0; k < source_length; ++k)
                cross_mask[(b * target_length + q) * source_length + k] =
                    (src[b * source_length + k] != config.pad_id);
        }
    }

    // 2. encoder：源 embedding+PE -> self-attention -> FFN，两个 sublayer 均 Post-LN。
    std::map<std::string, Tensor> traces;
    auto memory = embed(src, batch, source_length, "src_embedding");
    for (std::size_t i = 0; i < config.num_encoder_layers; ++i) {
        const auto prefix = "encoder." + std::to_string(i);
        // Post-LN 的两个残差边都保留原节点，反向必须将它们的贡献相加。
        memory = norm(add(memory, multi_head(memory, memory, encoder_mask, prefix + ".self_attn")),
                      prefix + ".norm1");
        memory = norm(add(memory, ffn(memory, prefix + ".ffn")), prefix + ".norm2");
        traces.emplace(prefix + ".output", memory);
    }
    traces.emplace("encoder_output", memory);

    // 3. decoder：目标 embedding+PE -> causal self-attention -> cross-attention -> FFN。
    // 每层复用同一 encoder memory，反向把所有 K/V 路径的梯度累加到它。
    auto output = embed(tgt, batch, target_length, "tgt_embedding");
    for (std::size_t i = 0; i < config.num_decoder_layers; ++i) {
        const auto prefix = "decoder." + std::to_string(i);
        output = norm(add(output, multi_head(output, output, decoder_mask, prefix + ".self_attn")),
                      prefix + ".norm1");
        // cross-attention 的 Q 来自 decoder，K/V 来自同一 encoder memory。
        output = norm(add(output, multi_head(output, memory, cross_mask, prefix + ".cross_attn")),
                      prefix + ".norm2");
        output = norm(add(output, ffn(output, prefix + ".ffn")), prefix + ".norm3");
        traces.emplace(prefix + ".output", output);
    }
    traces.emplace("decoder_output", output);

    // 4. 逐 token 投影到词表：[B,T,D] -> [B,T,V]；traces 保留各层输出节点。
    return {linear(output, parameters.at("generator.weight"), parameters.at("generator.bias")),
            std::move(traces)};
}

void Transformer::sgd(double learning_rate) {
    require(std::isfinite(learning_rate) && learning_rate > 0,
            "SGD learning rate must be finite and positive");
    // 先检查所有候选值，再原地更新，以免发现非有限值时留下一半更新的模型。
    for (const auto &pair : parameters)
        for (std::size_t i = 0; i < pair.second.size(); ++i)
            require(std::isfinite(pair.second.data()[i] - learning_rate * pair.second.grad()[i]),
                    "SGD update produced non-finite parameter");
    for (auto &pair : parameters)
        for (std::size_t i = 0; i < pair.second.size(); ++i)
            pair.second.data()[i] -= learning_rate * pair.second.grad()[i];
    // grad 保留供 inspect 等调用者读取；旧图缓存与更新后的参数已不再匹配。
}

std::vector<std::vector<std::size_t>> Transformer::generate(const std::vector<std::size_t> &src,
                                                            std::size_t batch,
                                                            std::size_t source_length,
                                                            std::size_t bos_id, std::size_t eos_id,
                                                            std::size_t max_new_tokens) const {
    validate_tokens(src, batch, source_length, true);
    require(bos_id < config.vocab_size && eos_id < config.vocab_size,
            "BOS/EOS token ID out of range");
    require(bos_id != config.pad_id && eos_id != config.pad_id, "BOS/EOS must not be PAD");
    require(bos_id != eos_id, "BOS and EOS must be different tokens");
    require(max_new_tokens < config.max_len, "BOS plus new tokens exceeds config max_len");
    std::vector<std::vector<std::size_t>> result(batch, std::vector<std::size_t>{bos_id});
    std::vector<bool> finished(batch, false);
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        const auto length = result.front().size();
        std::vector<std::size_t> tokens;
        tokens.reserve(batch * length);
        for (const auto &row : result)
            tokens.insert(tokens.end(), row.begin(), row.end());
        // 每步重算完整 prefix；只用最后一个位置的 logits 选下一个 token。
        auto prediction = forward(src, batch, source_length, tokens, length);
        bool all_finished = true;
        for (std::size_t b = 0; b < batch; ++b) {
            std::size_t next = config.pad_id;
            if (!finished[b]) {
                const auto offset = (b * length + length - 1) * config.vocab_size;
                next = 0;
                for (std::size_t v = 1; v < config.vocab_size; ++v)
                    if (prediction.logits.data()[offset + v] >
                        prediction.logits.data()[offset + next])
                        next = v;
                if (next == eos_id)
                    finished[b] = true;
            }
            result[b].push_back(next);
            all_finished = all_finished && finished[b];
        }
        if (all_finished)
            break;
    }
    return result;
}

} // namespace tlab
