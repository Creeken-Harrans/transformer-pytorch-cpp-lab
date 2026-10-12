#pragma once
#include "tlab/tensor.hpp"
#include <cstdint>
#include <map>
#include <string>

namespace tlab {
struct Config {
    std::size_t vocab_size = 16, d_model = 8, num_heads = 2, d_ff = 16;
    std::size_t num_encoder_layers = 1, num_decoder_layers = 1, max_len = 32, pad_id = 0;
    double eps = 1e-5;
    void validate() const;
};
struct Forward {
    Tensor logits;
    std::map<std::string, Tensor> traces;
};
class Transformer {
  public:
    Config config;
    std::map<std::string, Tensor> parameters;
    explicit Transformer(Config config, std::uint32_t seed = 42);
    Forward forward(const std::vector<std::size_t> &src, std::size_t batch,
                    std::size_t source_length, const std::vector<std::size_t> &tgt,
                    std::size_t target_length) const;
    void sgd(double learning_rate);
    std::vector<std::vector<std::size_t>> generate(const std::vector<std::size_t> &src,
                                                   std::size_t batch, std::size_t source_length,
                                                   std::size_t bos_id, std::size_t eos_id,
                                                   std::size_t max_new_tokens) const;

  private:
    Tensor embed(const std::vector<std::size_t> &ids, std::size_t batch, std::size_t length,
                 const std::string &prefix) const;
    Tensor multi_head(const Tensor &query, const Tensor &memory,
                      const std::vector<unsigned char> &mask, const std::string &prefix) const;
    Tensor norm(const Tensor &x, const std::string &prefix) const;
    Tensor ffn(const Tensor &x, const std::string &prefix) const;
    void validate_tokens(const std::vector<std::size_t> &ids, std::size_t batch, std::size_t length,
                         bool source) const;
};
} // namespace tlab
