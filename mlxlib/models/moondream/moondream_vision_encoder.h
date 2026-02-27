#pragma once

#include "config.h"
#include "mlx/mlx.h"

#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

namespace gomlx {

struct SigLIPBlock {
    // LayerNorm with bias (pre-norm)
    std::optional<mlx::core::array> norm1_weight, norm1_bias;
    std::optional<mlx::core::array> norm2_weight, norm2_bias;

    // Fused QKV split at load time
    std::optional<mlx::core::array> q_proj_weight, k_proj_weight, v_proj_weight;
    std::optional<mlx::core::array> q_proj_bias, k_proj_bias, v_proj_bias;

    // Output projection
    std::optional<mlx::core::array> out_proj_weight, out_proj_bias;

    // MLP with GELU (fc1/fc2 with bias)
    std::optional<mlx::core::array> fc1_weight, fc1_bias;
    std::optional<mlx::core::array> fc2_weight, fc2_bias;
};

class MoondreamVisionEncoder {
public:
    explicit MoondreamVisionEncoder(const MoondreamConfig& config);

    // pixel_values: [1, 3, 378, 378]
    // Returns: [1, 729, 1152]
    mlx::core::array forward(const mlx::core::array& pixel_values);

    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);

private:
    MoondreamConfig config_;
    int head_dim_;
    int num_patches_;

    // Patch embedding (linear, not conv)
    std::optional<mlx::core::array> patch_embed_weight_;  // [588, 1152] -> transposed for matmul
    std::optional<mlx::core::array> patch_embed_bias_;    // [1152]

    // Position embedding
    std::optional<mlx::core::array> position_embedding_;  // [729, 1152]

    // 27 SigLIP blocks
    std::vector<SigLIPBlock> blocks_;

    // Post-LayerNorm
    std::optional<mlx::core::array> post_ln_weight_, post_ln_bias_;
};

} // namespace gomlx
