#include "quantized_linear.h"

namespace gomlx {

using namespace mlx::core;

QuantizedLinear::QuantizedLinear(
    array weight,
    array scales,
    array biases,
    int group_size,
    int bits)
    : weight_(std::move(weight)),
      scales_(std::move(scales)),
      biases_(std::move(biases)),
      group_size_(group_size),
      bits_(bits),
      has_biases_(true) {}

array QuantizedLinear::forward(const array& input) const {
    // Use MLX's quantized_matmul for efficient quantized inference.
    // quantized_matmul(x, w, scales, biases, transpose, group_size, bits)
    return quantized_matmul(
        input,
        weight_,
        scales_,
        biases_,
        /*transpose=*/true,
        group_size_,
        bits_);
}

} // namespace gomlx
