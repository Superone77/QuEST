#pragma once
#include <torch/extension.h>

namespace QUIK::symmetric {

torch::Tensor int4FusedDequantizeCUDA(const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      const torch::Tensor &scale_row,
                                      const torch::Tensor &scale_col,
                                      const torch::Tensor &vec_a_add,
                                      const torch::Tensor &vec_b_add,
                                      const torch::Tensor &y);

}  // namespace QUIK::symmetric
