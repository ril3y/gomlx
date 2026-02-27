#pragma once

#include "base_model.h"
#include "config.h"
#include "quantized_linear.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

namespace gomlx {

struct VisionEncoderLayer {
    // LayerNorm (vision uses standard LayerNorm with bias, NOT RMSNorm)
    std::optional<mlx::core::array> input_layernorm_weight, input_layernorm_bias;
    std::optional<mlx::core::array> post_attention_layernorm_weight, post_attention_layernorm_bias;

    // Self-attention Q/K/V/O
    std::optional<mlx::core::array> q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight;
    std::unique_ptr<QuantizedLinear> q_proj_q, k_proj_q, v_proj_q, o_proj_q;

    // MLP: fc1, fc2 (with actual linear bias, plus GELU activation)
    std::optional<mlx::core::array> fc1_weight, fc1_bias, fc2_weight, fc2_bias;
    std::unique_ptr<QuantizedLinear> fc1_q, fc2_q;

    // Gating (global layers only)
    bool is_gated = false;
    std::optional<mlx::core::array> gate_attn, gate_ffn;  // scalar [1] gates

    bool is_quantized = false;
};

class VisionEncoder {
public:
    explicit VisionEncoder(const VisionModelConfig& config);

    // pixel_values: [num_tiles, 3, 560, 560], returns [1, num_tiles*(num_patches+1), 7680]
    mlx::core::array forward(const mlx::core::array& pixel_values, int aspect_ratio_id, int num_tiles);

    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights,
                      int group_size, int bits);

private:
    // Patch embedding
    std::optional<mlx::core::array> patch_embedding_weight_;    // Conv2D: [1280, 14, 14, 3] (OHWI for MLX)
    std::optional<mlx::core::array> class_embedding_;           // [1280]
    std::optional<mlx::core::array> position_embedding_weight_; // [num_patches+1, 1280]
    std::optional<mlx::core::array> position_gate_;             // [1]

    // Tile embeddings
    std::optional<mlx::core::array> tile_embedding_weight_;     // [max_ar_id+1, max_tiles, num_patches+1, 1280]
    std::optional<mlx::core::array> pre_tile_embedding_weight_, pre_tile_gate_;
    std::optional<mlx::core::array> post_tile_embedding_weight_, post_tile_gate_;

    // Pre/post norms
    std::optional<mlx::core::array> layernorm_pre_weight_, layernorm_pre_bias_;
    std::optional<mlx::core::array> layernorm_post_weight_, layernorm_post_bias_;

    // Transformer layers
    std::vector<VisionEncoderLayer> local_layers_;   // 32 layers
    std::vector<VisionEncoderLayer> global_layers_;  // 8 layers, gated

    VisionModelConfig config_;

    // Helper methods
    mlx::core::array layer_norm(const mlx::core::array& x,
                                const mlx::core::array& weight,
                                const mlx::core::array& bias, float eps);
    mlx::core::array run_encoder_layer(VisionEncoderLayer& layer,
                                       const mlx::core::array& x,
                                       int num_heads, int head_dim);
    mlx::core::array linear_forward(const mlx::core::array& input,
                                    const std::optional<mlx::core::array>& weight,
                                    const QuantizedLinear* q_linear,
                                    const std::string& debug_name = "") const;
    void load_linear(const std::string& prefix,
                     const std::unordered_map<std::string, mlx::core::array>& weights,
                     std::optional<mlx::core::array>& dense_weight,
                     std::unique_ptr<QuantizedLinear>& q_linear,
                     bool& is_quantized, int group_size, int bits);
};

} // namespace gomlx
