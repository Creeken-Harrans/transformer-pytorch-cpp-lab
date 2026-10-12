#pragma once
#include "tlab/tensor.hpp"
#include <cstdint>
#include <map>
#include <string>

namespace tlab {
// 参数名、shape 与 Python state_dict 相同；模型采用 double、Post-LN 和 dropout=0。
struct Config {
    std::size_t vocab_size = 16, d_model = 8, num_heads = 2, d_ff = 16;
    std::size_t num_encoder_layers = 1, num_decoder_layers = 1, max_len = 32, pad_id = 0;
    double eps = 1e-5;
    void validate() const;
};

struct Forward {
    // [B,T,V]
    Tensor logits;
    // 各层 output，以及 encoder/decoder_output；共享 DAG。
    std::map<std::string, Tensor> traces;
};

class Transformer {
  public:
    Config config;
    std::map<std::string, Tensor> parameters;

    explicit Transformer(Config config, std::uint32_t seed = 42);

    // src/tgt 是 row-major 展平的 [B,S]/[B,T]；返回 logits[B,T,V]。
    // 每条 src 至少有一个非 PAD；每条 tgt 的首 token 非 PAD；长度 <= max_len。
    // 返回图共享 parameters 的叶节点；forward 到 backward 之间不能更新参数。
    Forward forward(const std::vector<std::size_t> &src, std::size_t batch,
                    std::size_t source_length, const std::vector<std::size_t> &tgt,
                    std::size_t target_length) const;

    // theta -= learning_rate * grad：使用现有梯度，不清零、不重新 backward。
    // 更新后旧 forward 图已过期；下一训练步重新 forward/backward。
    void sgd(double learning_rate);

    // src[B,S] -> 每条序列含 BOS；EOS 之后补 PAD，全部结束时提前停止。
    // BOS/EOS 在词表内、彼此不同且非 PAD；1+max_new_tokens <= max_len。
    std::vector<std::vector<std::size_t>> generate(const std::vector<std::size_t> &src,
                                                   std::size_t batch, std::size_t source_length,
                                                   std::size_t bos_id, std::size_t eos_id,
                                                   std::size_t max_new_tokens) const;

  private:
    // lookup * sqrt(D) + 正弦位置编码 -> [B,L,D]。
    Tensor embed(const std::vector<std::size_t> &ids, std::size_t batch, std::size_t length,
                 const std::string &prefix) const;

    // query[B,Lq,D] 提供 Q；memory[B,Lk,D] 提供 K/V；四个 projection 均独立。
    Tensor multi_head(const Tensor &query, const Tensor &memory,
                      const std::vector<unsigned char> &mask, const std::string &prefix) const;

    Tensor norm(const Tensor &x, const std::string &prefix) const;

    // [B,L,D] -> linear1[B,L,F] -> ReLU -> linear2[B,L,D]。
    Tensor ffn(const Tensor &x, const std::string &prefix) const;

    void validate_tokens(const std::vector<std::size_t> &ids, std::size_t batch, std::size_t length,
                         bool source) const;
};
} // namespace tlab
