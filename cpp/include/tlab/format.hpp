#pragma once
#include "tlab/model.hpp"
#include <iosfwd>
#include <map>
#include <string>

namespace tlab {
using Bundle = std::map<std::string, Tensor>;
Bundle read_bundle(std::istream &input);
Bundle read_bundle(const std::string &path);
void write_bundle(std::ostream &output, const Bundle &bundle);
void write_bundle(const std::string &path, const Bundle &bundle);
struct Case {
    Transformer model;
    std::vector<std::size_t> src, tgt, labels;
    std::size_t batch, source_length, target_length;
    double learning_rate;
    Bundle original;
};
Bundle make_case(const Transformer &model, const Tensor &src, const Tensor &tgt,
                 const Tensor &labels, double learning_rate);
Case load_case(const Bundle &bundle);
} // namespace tlab
