#pragma once

#include "mlx/mlx.h"

namespace gomlx {

// Quantized linear layer that uses MLX's quantized_matmul for efficient
// inference with quantized weights (typically 4-bit).
// Weights are stored in quantized form with scales and optional biases.
class QuantizedLinear {
public:
    // Construct from pre-quantized weight, scales, biases.
    // weight: quantized weight matrix
    // scales: per-group scale factors
    // biases: per-group bias values (optional, can be empty)
    // group_size: quantization group size (e.g. 64, 128)
    // bits: number of bits per weight (e.g. 4, 8)
    QuantizedLinear(
        mlx::core::array weight,
        mlx::core::array scales,
        mlx::core::array biases,
        int group_size = 64,
        int bits = 4);

    // Forward pass: input @ dequantized(weight).T
    // input shape: [..., in_features]
    // output shape: [..., out_features]
    mlx::core::array forward(const mlx::core::array& input) const;

    int group_size() const { return group_size_; }
    int bits() const { return bits_; }

    const mlx::core::array& weight() const { return weight_; }
    const mlx::core::array& scales() const { return scales_; }
    const mlx::core::array& biases() const { return biases_; }

private:
    mlx::core::array weight_;
    mlx::core::array scales_;
    mlx::core::array biases_;
    int group_size_;
    int bits_;
    bool has_biases_;
};

} // namespace gomlx
