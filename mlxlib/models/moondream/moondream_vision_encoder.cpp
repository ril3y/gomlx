#include "models/moondream/moondream_vision_encoder.h"

#include <cmath>
#include <stdexcept>

namespace gomlx {

using namespace mlx::core;

// Exact GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
static array gelu_exact(const array& x) {
    auto cdf = multiply(
        array(0.5f, float32),
        add(array(1.0f, float32),
            erf(divide(x, array(std::sqrt(2.0f), float32)))));
    return multiply(x, cdf);
}

MoondreamVisionEncoder::MoondreamVisionEncoder(const MoondreamConfig& config)
    : config_(config) {
    head_dim_ = config_.vis_hidden_size / config_.vis_num_heads;  // 1152/16 = 72
    num_patches_ = config_.vis_num_patches;  // 729
    blocks_.resize(config_.vis_num_layers);  // 27
}

array MoondreamVisionEncoder::forward(const array& pixel_values) {
    int hidden = config_.vis_hidden_size;  // 1152
    int num_heads = config_.vis_num_heads;  // 16
    int patch_size = config_.vis_patch_size;  // 14
    int image_size = config_.vis_image_size;  // 378
    int patches_per_side = image_size / patch_size;  // 27
    int num_patches = patches_per_side * patches_per_side;  // 729
    float eps = config_.vis_layer_norm_eps;

    // --- 1. Patch embedding (linear, not conv2d) ---
    // Input: [1, 3, 378, 378] (NCHW)
    // Reshape to patches: [1, 729, 3*14*14] = [1, 729, 588]
    // Transpose to NHWC first: [1, 378, 378, 3]
    auto x = transpose(pixel_values, {0, 2, 3, 1});
    // Reshape to [1, 27, 14, 27, 14, 3]
    x = reshape(x, {1, patches_per_side, patch_size, patches_per_side, patch_size, 3});
    // Transpose to [1, 27, 27, 14, 14, 3]
    x = transpose(x, {0, 1, 3, 2, 4, 5});
    // Flatten patches: [1, 729, 588]
    x = reshape(x, {1, num_patches, patch_size * patch_size * 3});

    // Linear projection: [1, 729, 588] x [588, 1152]^T -> [1, 729, 1152]
    x = add(matmul(x, transpose(*patch_embed_weight_)), *patch_embed_bias_);

    // --- 2. Add position embedding ---
    // position_embedding_: [729, 1152] -> broadcast to [1, 729, 1152]
    x = add(x, reshape(*position_embedding_, {1, num_patches, hidden}));

    // --- 3. Run 27 SigLIP blocks ---
    for (int i = 0; i < config_.vis_num_layers; i++) {
        auto& block = blocks_[i];
        int batch = x.shape(0);
        int seq_len = x.shape(1);

        // Pre-norm 1 (LayerNorm with bias)
        auto normed = fast::layer_norm(x, *block.norm1_weight, *block.norm1_bias, eps);

        // Q, K, V projections (already split at load time)
        auto q = add(matmul(normed, transpose(*block.q_proj_weight)), *block.q_proj_bias);
        auto k = add(matmul(normed, transpose(*block.k_proj_weight)), *block.k_proj_bias);
        auto v = add(matmul(normed, transpose(*block.v_proj_weight)), *block.v_proj_bias);

        // Reshape: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        q = transpose(reshape(q, {batch, seq_len, num_heads, head_dim_}), {0, 2, 1, 3});
        k = transpose(reshape(k, {batch, seq_len, num_heads, head_dim_}), {0, 2, 1, 3});
        v = transpose(reshape(v, {batch, seq_len, num_heads, head_dim_}), {0, 2, 1, 3});

        // Fully bidirectional attention (no mask)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        auto attn_out = fast::scaled_dot_product_attention(q, k, v, scale, "");

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        attn_out = transpose(attn_out, {0, 2, 1, 3});
        attn_out = reshape(attn_out, {batch, seq_len, hidden});

        // Output projection + residual
        attn_out = add(matmul(attn_out, transpose(*block.out_proj_weight)), *block.out_proj_bias);
        x = add(x, attn_out);

        // Pre-norm 2 (LayerNorm with bias)
        normed = fast::layer_norm(x, *block.norm2_weight, *block.norm2_bias, eps);

        // MLP: fc1 -> GELU -> fc2
        auto mlp_out = add(matmul(normed, transpose(*block.fc1_weight)), *block.fc1_bias);
        mlp_out = gelu_exact(mlp_out);
        mlp_out = add(matmul(mlp_out, transpose(*block.fc2_weight)), *block.fc2_bias);

        // MLP residual
        x = add(x, mlp_out);
    }

    // --- 4. Post-LayerNorm ---
    x = fast::layer_norm(x, *post_ln_weight_, *post_ln_bias_, eps);

    // Output: [1, 729, 1152]
    return x;
}

void MoondreamVisionEncoder::load_weights(
    const std::unordered_map<std::string, array>& weights) {

    auto find_weight = [&](const std::string& name) -> const array& {
        auto it = weights.find(name);
        if (it == weights.end()) {
            throw std::runtime_error("Missing Moondream vision weight: " + name);
        }
        return it->second;
    };

    // --- Patch embedding (linear) ---
    patch_embed_weight_ = find_weight(
        "vision_encoder.encoder.model.visual.patch_embedding.linear.weight");
    patch_embed_bias_ = find_weight(
        "vision_encoder.encoder.model.visual.patch_embedding.linear.bias");

    // --- Position embedding ---
    position_embedding_ = find_weight(
        "vision_encoder.encoder.model.visual.patch_embedding.position_embedding.weight");

    // --- Transformer blocks ---
    for (int i = 0; i < config_.vis_num_layers; i++) {
        std::string prefix = "vision_encoder.encoder.model.visual.encoder.layers." + std::to_string(i);
        auto& block = blocks_[i];

        // LayerNorm weights (with bias)
        block.norm1_weight = find_weight(prefix + ".layer_norm1.weight");
        block.norm1_bias = find_weight(prefix + ".layer_norm1.bias");
        block.norm2_weight = find_weight(prefix + ".layer_norm2.weight");
        block.norm2_bias = find_weight(prefix + ".layer_norm2.bias");

        // Fused QKV: [3*1152, 1152] = [3456, 1152] -> split into Q/K/V
        auto qkv_weight = find_weight(prefix + ".self_attn.qkv_proj.weight");
        auto qkv_bias = find_weight(prefix + ".self_attn.qkv_proj.bias");

        int h = config_.vis_hidden_size;  // 1152
        block.q_proj_weight = slice(qkv_weight, {0, 0}, {h, h});
        block.k_proj_weight = slice(qkv_weight, {h, 0}, {2 * h, h});
        block.v_proj_weight = slice(qkv_weight, {2 * h, 0}, {3 * h, h});

        block.q_proj_bias = slice(qkv_bias, {0}, {h});
        block.k_proj_bias = slice(qkv_bias, {h}, {2 * h});
        block.v_proj_bias = slice(qkv_bias, {2 * h}, {3 * h});

        // Output projection
        block.out_proj_weight = find_weight(prefix + ".self_attn.out_proj.weight");
        block.out_proj_bias = find_weight(prefix + ".self_attn.out_proj.bias");

        // MLP
        block.fc1_weight = find_weight(prefix + ".mlp.fc1.weight");
        block.fc1_bias = find_weight(prefix + ".mlp.fc1.bias");
        block.fc2_weight = find_weight(prefix + ".mlp.fc2.weight");
        block.fc2_bias = find_weight(prefix + ".mlp.fc2.bias");
    }

    // --- Post-LayerNorm ---
    post_ln_weight_ = find_weight("vision_encoder.encoder.model.visual.post_layernorm.weight");
    post_ln_bias_ = find_weight("vision_encoder.encoder.model.visual.post_layernorm.bias");
}

} // namespace gomlx
