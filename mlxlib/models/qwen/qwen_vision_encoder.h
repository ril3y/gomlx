#pragma once

#include "config.h"
#include "mlx/mlx.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

namespace gomlx {

struct QwenVisionBlock {
    // RMSNorm (no bias)
    std::optional<mlx::core::array> norm1_weight;
    std::optional<mlx::core::array> norm2_weight;

    // Combined QKV attention: [3*hidden, hidden]
    std::optional<mlx::core::array> qkv_weight;  // [3840, 1280]
    std::optional<mlx::core::array> qkv_bias;     // [3840]

    // Output projection
    std::optional<mlx::core::array> proj_weight;   // [1280, 1280]
    std::optional<mlx::core::array> proj_bias;     // [1280]

    // SwiGLU MLP (all with bias)
    std::optional<mlx::core::array> gate_proj_weight, gate_proj_bias;  // [3420, 1280] + [3420]
    std::optional<mlx::core::array> up_proj_weight, up_proj_bias;      // [3420, 1280] + [3420]
    std::optional<mlx::core::array> down_proj_weight, down_proj_bias;  // [1280, 3420] + [1280]
};

class QwenVisionEncoder {
public:
    explicit QwenVisionEncoder(const Qwen2_5VisionConfig& config);

    // pixel_values: [1, 3, 2, H, W]
    // Returns: [1, num_vision_tokens, out_hidden_size]
    mlx::core::array forward(const mlx::core::array& pixel_values, int grid_h, int grid_w);

    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);

    int num_vision_tokens(int grid_h, int grid_w) const {
        int merge = config_.spatial_merge_size;
        return (grid_h / merge) * (grid_w / merge);
    }

private:
    Qwen2_5VisionConfig config_;
    int head_dim_;
    std::set<int> fullatt_set_;

    // Patch embedding (Conv3D weight folded to Conv2D)
    std::optional<mlx::core::array> patch_embed_weight_;  // [1280, 14, 14, 3] after folding

    // 32 ViT blocks
    std::vector<QwenVisionBlock> blocks_;

    // PatchMerger
    std::optional<mlx::core::array> merger_ln_weight_;     // RMSNorm [1280]
    std::optional<mlx::core::array> merger_mlp0_weight_, merger_mlp0_bias_;  // [5120, 5120]
    std::optional<mlx::core::array> merger_mlp2_weight_, merger_mlp2_bias_;  // [3584, 5120]

    // Helpers
    std::pair<mlx::core::array, mlx::core::array> compute_2d_rope(int grid_h, int grid_w);
    mlx::core::array run_block(QwenVisionBlock& block, const mlx::core::array& x,
                                const mlx::core::array& rope_cos, const mlx::core::array& rope_sin,
                                bool full_attention, int grid_h, int grid_w);
};

} // namespace gomlx
