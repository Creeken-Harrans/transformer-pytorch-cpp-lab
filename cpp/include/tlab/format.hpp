#pragma once
#include "tlab/model.hpp"
#include <iosfwd>
#include <map>
#include <string>

namespace tlab {
using Bundle = std::map<std::string, Tensor>;

// TLAB1：具名 row-major double 张量；这里只读写数值，不保存 grad 或 DAG。
// 通用 bundle 的文件约束见 DESIGN.md；模型/batch 约束由 load_case 检查。
Bundle read_bundle(std::istream &input);
Bundle read_bundle(const std::string &path);
void write_bundle(std::ostream &output, const Bundle &bundle);
void write_bundle(const std::string &path, const Bundle &bundle);

struct Case {
    Transformer model;
    std::vector<std::size_t> src, tgt, labels;
    std::size_t batch, source_length, target_length;
    double learning_rate;
    Bundle original; // 保留全部原始记录，供 CLI 保存配置与 batch；Tensor 仍为共享句柄。
};

// src[B,S]、tgt/labels[B,T]，token 用精确整数 double 表示。
// 深拷贝参数和 batch 数值，返回独立快照，并以 load_case 的契约校验。
Bundle make_case(const Transformer &model, const Tensor &src, const Tensor &tgt,
                 const Tensor &labels, double learning_rate);

// 校验完整 case，为模型参数创建独立叶节点；original 则共享输入 bundle 的节点。
Case load_case(const Bundle &bundle);
} // namespace tlab
