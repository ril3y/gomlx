#include "models/llama/vision_encoder.h"

#include <cmath>
#include <stdexcept>

namespace gomlx {

using namespace mlx::core;

// GELU activation (exact form)
static array gelu(const array& x) {
    auto cdf = multiply(
        array(0.5f, float32),
        add(array(1.0f, float32),
            erf(divide(x, array(std::sqrt(2.0f), float32)))));
    return multiply(x, cdf);
}

// Check if conv weight is already in MLX format [O, kH, kW, I]
static bool check_array_shape(const array& arr) {
    if (arr.ndim() != 4) return false;
    int out_ch = arr.shape(0), kH = arr.shape(1), kW = arr.shape(2);
    return (out_ch >= kH) && (out_ch >= kW) && (kH == kW);
}

VisionEncoder::VisionEncoder(const VisionModelConfig& config)
    : config_(config) {
    local_layers_.resize(config_.num_hidden_layers);
    global_layers_.resize(config_.num_global_layers);
}

array VisionEncoder::layer_norm(const array& x, const array& weight,
                                const array& bias, float eps) {
    auto mean_val = mean(x, -1, true);
    auto centered = subtract(x, mean_val);
    auto var_val = mean(square(centered), -1, true);
    auto normalized = divide(centered, sqrt(add(var_val, array(eps, float32))));
    return add(multiply(normalized, weight), bias);
}

array VisionEncoder::linear_forward(const array& input,
                                    const std::optional<array>& weight,
                                    const QuantizedLinear* q_linear,
                                    const std::string& debug_name) const {
    if (q_linear) {
        return q_linear->forward(input);
    }
    if (!weight.has_value()) {
        throw std::runtime_error("linear_forward: no weight for " + debug_name);
    }
    return matmul(input, transpose(*weight));
}

void VisionEncoder::load_linear(
    const std::string& prefix,
    const std::unordered_map<std::string, array>& weights,
    std::optional<array>& dense_weight,
    std::unique_ptr<QuantizedLinear>& q_linear,
    bool& is_quantized, int group_size, int bits) {

    auto w_it = weights.find(prefix + ".weight");
    auto s_it = weights.find(prefix + ".scales");

    if (s_it != weights.end() && w_it != weights.end()) {
        is_quantized = true;
        auto b_it = weights.find(prefix + ".biases");
        if (b_it != weights.end()) {
            q_linear = std::make_unique<QuantizedLinear>(
                w_it->second, s_it->second, b_it->second, group_size, bits);
        } else {
            auto zero_biases = zeros(s_it->second.shape(), s_it->second.dtype());
            q_linear = std::make_unique<QuantizedLinear>(
                w_it->second, s_it->second, zero_biases, group_size, bits);
        }
    } else if (w_it != weights.end()) {
        dense_weight = w_it->second;
    }
}

array VisionEncoder::run_encoder_layer(VisionEncoderLayer& layer,
                                       const array& x,
                                       int num_heads, int head_dim) {
    int batch = x.shape(0);
    int seq_len = x.shape(1);

    // Pre-attention LayerNorm
    array normed = layer_norm(x, *layer.input_layernorm_weight,
                              *layer.input_layernorm_bias, config_.norm_eps);

    // Self-attention: Q, K, V projections
    array q = linear_forward(normed, layer.q_proj_weight, layer.q_proj_q.get(), "q_proj");
    array k = linear_forward(normed, layer.k_proj_weight, layer.k_proj_q.get(), "k_proj");
    array v = linear_forward(normed, layer.v_proj_weight, layer.v_proj_q.get(), "v_proj");

    // Reshape to [batch, seq_len, num_heads, head_dim] then transpose
    q = transpose(reshape(q, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});
    k = transpose(reshape(k, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});
    v = transpose(reshape(v, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});

    // Scaled dot-product attention (NO causal mask for vision)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    array attn_out = fast::scaled_dot_product_attention(q, k, v, scale, "");

    // Reshape back
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch, seq_len, num_heads * head_dim});

    // Output projection
    attn_out = linear_forward(attn_out, layer.o_proj_weight, layer.o_proj_q.get(), "o_proj");

    // Gated attention residual (global layers) or plain residual
    array h = (layer.is_gated && layer.gate_attn.has_value())
        ? add(x, multiply(tanh(*layer.gate_attn), attn_out))
        : add(x, attn_out);

    // Post-attention LayerNorm
    normed = layer_norm(h, *layer.post_attention_layernorm_weight,
                        *layer.post_attention_layernorm_bias, config_.norm_eps);

    // MLP: fc1 -> GELU -> fc2
    array mlp_out = linear_forward(normed, layer.fc1_weight, layer.fc1_q.get(), "fc1");
    if (layer.fc1_bias.has_value()) {
        mlp_out = add(mlp_out, *layer.fc1_bias);
    }
    mlp_out = gelu(mlp_out);
    mlp_out = linear_forward(mlp_out, layer.fc2_weight, layer.fc2_q.get(), "fc2");
    if (layer.fc2_bias.has_value()) {
        mlp_out = add(mlp_out, *layer.fc2_bias);
    }

    // Gated MLP residual (global layers) or plain residual
    if (layer.is_gated && layer.gate_ffn.has_value()) {
        return add(h, multiply(tanh(*layer.gate_ffn), mlp_out));
    }
    return add(h, mlp_out);
}

array VisionEncoder::forward(const array& pixel_values, int aspect_ratio_id, int num_tiles) {
    int num_patches = config_.num_patches_per_tile();  // 1600
    int hidden_size = config_.hidden_size;              // 1280
    int num_heads = config_.num_attention_heads;        // 16
    int head_dim = hidden_size / num_heads;             // 80
    int max_num_tiles = config_.max_num_tiles;          // 4
    int num_patches_with_cls = num_patches + 1;         // 1601

    // --- 1. Patch embedding (Conv2D with stride=14) ---
    // Input pixel_values: [num_tiles, 3, 560, 560] (NCHW)
    // MLX conv2d expects NHWC input
    array x = transpose(pixel_values, {0, 2, 3, 1});  // -> [num_tiles, 560, 560, 3]
    // patch_embedding_weight_ is in OHWI format: [1280, 14, 14, 3]
    x = conv2d(x, *patch_embedding_weight_, /*stride=*/{14, 14});
    // Result: [num_tiles, 40, 40, 1280]
    // Transpose to [num_tiles, 1280, 40, 40] then reshape to [num_tiles, 1600, 1280]
    // Following reference: moveaxis(3,1) then reshape
    x = transpose(x, {0, 3, 1, 2});  // [num_tiles, 1280, 40, 40]
    x = reshape(x, {x.shape(0), x.shape(1), -1});  // [num_tiles, 1280, 1600]
    x = transpose(x, {0, 2, 1});  // [num_tiles, 1600, 1280]

    // --- 2. Pre-tile positional embedding ---
    // Reshape to [1, num_tiles, num_patches, hidden_size] for tile embedding
    x = reshape(x, {1, num_tiles, num_patches, hidden_size});

    if (pre_tile_embedding_weight_.has_value() && pre_tile_gate_.has_value()) {
        // pre_tile_embedding is nn.Embedding: [max_ar_id+1, max_tiles * hidden_size]
        // Lookup by aspect_ratio_id -> [1, max_tiles * hidden_size]
        // Reshape to [1, max_tiles, 1, hidden_size]
        array ar_ids = array({aspect_ratio_id}, {1}, int32);
        array emb = take(*pre_tile_embedding_weight_, ar_ids, 0);  // [1, max_tiles * hidden_size]
        emb = reshape(emb, {1, max_num_tiles, 1, hidden_size});
        // Slice to actual num_tiles and broadcast-add with gate
        emb = slice(emb, {0, 0, 0, 0}, {1, num_tiles, 1, hidden_size});
        x = add(x, multiply(tanh(*pre_tile_gate_), emb));
    }

    // --- 3. Prepend CLS token ---
    // Reshape back to [num_tiles, num_patches, hidden_size]
    x = reshape(x, {num_tiles, num_patches, hidden_size});
    // class_embedding_: [hidden_size] -> [num_tiles, 1, hidden_size]
    array cls = reshape(*class_embedding_, {1, 1, hidden_size});
    cls = broadcast_to(cls, {num_tiles, 1, hidden_size});
    x = concatenate({cls, x}, 1);
    // Result: [num_tiles, 1601, 1280]

    // --- 4. Gated positional + tile embedding ---
    // Reshape to [1, num_tiles, num_patches+1, hidden_size]
    x = reshape(x, {1, num_tiles, num_patches_with_cls, hidden_size});

    if (position_embedding_weight_.has_value() && position_gate_.has_value()) {
        // position_embedding: [1601, 1280]
        // Gated: (1 - tanh(gate)) * pos_emb
        array pos_emb = reshape(*position_embedding_weight_,
                                {1, 1, num_patches_with_cls, hidden_size});
        array gated_pos = multiply(
            subtract(array(1.0f, float32), tanh(*position_gate_)),
            pos_emb);
        x = add(x, gated_pos);

        // tile_embedding: nn.Embedding [max_ar_id+1, max_tiles * num_patches+1 * hidden_size]
        // Lookup by aspect_ratio_id, reshape to [1, max_tiles, num_patches+1, hidden_size]
        array ar_ids = array({aspect_ratio_id}, {1}, int32);
        array tile_emb = take(*tile_embedding_weight_, ar_ids, 0);
        tile_emb = reshape(tile_emb, {1, max_num_tiles, num_patches_with_cls, hidden_size});
        tile_emb = slice(tile_emb, {0, 0, 0, 0}, {1, num_tiles, num_patches_with_cls, hidden_size});
        array gated_tile = multiply(tanh(*position_gate_), tile_emb);
        x = add(x, gated_tile);
    }

    // --- 5. Pre-LayerNorm ---
    x = layer_norm(x, *layernorm_pre_weight_, *layernorm_pre_bias_, config_.norm_eps);

    // Pad sequence to multiple of 8
    int seq_after_cls = num_patches_with_cls;
    int num_padding_patches = (8 - (seq_after_cls % 8)) % 8;
    if (num_padding_patches > 0) {
        // Pad along axis 2 (seq dim) in shape [1, num_tiles, seq, hidden]
        x = pad(x, {{0, 0}, {0, 0}, {0, num_padding_patches}, {0, 0}});
    }
    int padded_seq = seq_after_cls + num_padding_patches;

    // Flatten to [1, num_tiles * padded_seq, hidden_size] for transformer
    x = reshape(x, {1, num_tiles * padded_seq, hidden_size});

    // --- 6. Local transformer layers (32 layers) ---
    // Collect ALL hidden states for intermediate extraction
    std::vector<array> all_encoder_states;
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        x = run_encoder_layer(local_layers_[i], x, num_heads, head_dim);
        all_encoder_states.push_back(x);
    }

    // --- 7. Post-LayerNorm ---
    x = layer_norm(x, *layernorm_post_weight_, *layernorm_post_bias_, config_.norm_eps);

    // --- 8. Post-tile positional embedding ---
    // Reshape to [1, num_tiles, padded_seq, hidden_size]
    x = reshape(x, {1, num_tiles, padded_seq, hidden_size});

    if (post_tile_embedding_weight_.has_value() && post_tile_gate_.has_value()) {
        array ar_ids = array({aspect_ratio_id}, {1}, int32);
        array emb = take(*post_tile_embedding_weight_, ar_ids, 0);
        emb = reshape(emb, {1, max_num_tiles, 1, hidden_size});
        emb = slice(emb, {0, 0, 0, 0}, {1, num_tiles, 1, hidden_size});
        x = add(x, multiply(tanh(*post_tile_gate_), emb));
    }

    // Flatten for global transformer
    x = reshape(x, {1, num_tiles * padded_seq, hidden_size});

    // --- 9. Global transformer layers (8 layers, gated) ---
    for (int i = 0; i < config_.num_global_layers; i++) {
        x = run_encoder_layer(global_layers_[i], x, num_heads, head_dim);
    }

    // --- 10. Remove padding and reshape ---
    // Reshape global output to [1, num_tiles, padded_seq, hidden_size]
    x = reshape(x, {1, num_tiles, padded_seq, hidden_size});
    // Remove padding
    if (num_padding_patches > 0) {
        x = slice(x, {0, 0, 0, 0}, {1, num_tiles, seq_after_cls, hidden_size});
    }
    // Final shape: [1, num_tiles, num_patches+1, hidden_size]

    // --- 11. Collect intermediate hidden states ---
    // Stack all encoder states along last dim, then index
    // Each state: [1, num_tiles * padded_seq, hidden_size]
    // Stack along axis=-1 -> [1, num_tiles*padded_seq, hidden_size, num_layers]
    array stacked = stack(all_encoder_states, -1);
    // Select intermediate layers
    std::vector<array> selected;
    for (int idx : config_.intermediate_layers_indices) {
        // Slice the last dimension at index idx
        selected.push_back(
            slice(stacked, {0, 0, 0, idx}, {1, num_tiles * padded_seq, hidden_size, idx + 1}));
    }
    // Concatenate selected along last dim
    array intermediate = concatenate(selected, -1);
    // intermediate shape: [1, num_tiles*padded_seq, hidden_size * num_selected]

    // Reshape and remove padding
    int num_selected = (int)config_.intermediate_layers_indices.size();
    intermediate = reshape(intermediate,
                           {1, num_tiles, padded_seq, hidden_size * num_selected});
    if (num_padding_patches > 0) {
        intermediate = slice(intermediate, {0, 0, 0, 0},
                             {1, num_tiles, seq_after_cls, hidden_size * num_selected});
    }

    // Reshape both to [1, num_tiles, num_patches+1, dim]
    // Concatenate [global_output, intermediate] along last dim
    // -> [1, num_tiles, num_patches+1, hidden_size * (1 + num_selected)]
    array result = concatenate({x, intermediate}, -1);

    // Reshape to [1, num_tiles * (num_patches+1), vision_output_dim]
    result = reshape(result, {1, num_tiles * num_patches_with_cls, config_.vision_output_dim});

    return result;
}

void VisionEncoder::load_weights(
    const std::unordered_map<std::string, array>& weights,
    int group_size, int bits) {

    auto find_weight = [&](const std::string& name) -> const array& {
        auto it = weights.find(name);
        if (it == weights.end()) {
            throw std::runtime_error("Missing vision weight: " + name);
        }
        return it->second;
    };

    auto find_optional = [&](const std::string& name) -> std::optional<array> {
        auto it = weights.find(name);
        if (it != weights.end()) return it->second;
        return std::nullopt;
    };

    // Helper to dequantize an embedding weight for lookup
    auto dequantize_embedding = [&](const std::string& prefix) -> array {
        auto w = find_weight(prefix + ".weight");
        auto s_it = weights.find(prefix + ".scales");
        if (s_it != weights.end()) {
            auto b_it = weights.find(prefix + ".biases");
            std::optional<array> biases;
            if (b_it != weights.end()) biases = b_it->second;
            auto deq = dequantize(w, s_it->second, biases, group_size, bits);
            eval(deq);
            return deq;
        }
        return w;
    };

    // --- Patch embedding ---
    {
        auto patch_w = find_weight("vision_tower.patch_embedding.weight");
        if (check_array_shape(patch_w)) {
            patch_embedding_weight_ = patch_w;  // Already in OHWI
        } else {
            patch_embedding_weight_ = transpose(patch_w, {0, 2, 3, 1});  // OIHW -> OHWI
        }
    }

    // Class embedding
    class_embedding_ = find_weight("vision_tower.class_embedding");

    // Gated positional embedding
    position_embedding_weight_ = find_weight(
        "vision_tower.gated_positional_embedding.embedding");
    position_gate_ = find_weight(
        "vision_tower.gated_positional_embedding.gate");
    tile_embedding_weight_ = dequantize_embedding(
        "vision_tower.gated_positional_embedding.tile_embedding");

    // Pre-tile positional embedding
    pre_tile_embedding_weight_ = dequantize_embedding(
        "vision_tower.pre_tile_positional_embedding.embedding");
    pre_tile_gate_ = find_weight(
        "vision_tower.pre_tile_positional_embedding.gate");

    // Post-tile positional embedding
    post_tile_embedding_weight_ = dequantize_embedding(
        "vision_tower.post_tile_positional_embedding.embedding");
    post_tile_gate_ = find_weight(
        "vision_tower.post_tile_positional_embedding.gate");

    // Pre/post LayerNorm
    layernorm_pre_weight_ = find_weight("vision_tower.layernorm_pre.weight");
    layernorm_pre_bias_ = find_weight("vision_tower.layernorm_pre.bias");
    layernorm_post_weight_ = find_weight("vision_tower.layernorm_post.weight");
    layernorm_post_bias_ = find_weight("vision_tower.layernorm_post.bias");

    // --- Local transformer layers ---
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string prefix = "vision_tower.transformer.layers." + std::to_string(i);
        auto& layer = local_layers_[i];

        layer.input_layernorm_weight = find_weight(prefix + ".input_layernorm.weight");
        layer.input_layernorm_bias = find_weight(prefix + ".input_layernorm.bias");
        layer.post_attention_layernorm_weight = find_weight(
            prefix + ".post_attention_layernorm.weight");
        layer.post_attention_layernorm_bias = find_weight(
            prefix + ".post_attention_layernorm.bias");

        load_linear(prefix + ".self_attn.q_proj", weights,
                    layer.q_proj_weight, layer.q_proj_q, layer.is_quantized,
                    group_size, bits);
        load_linear(prefix + ".self_attn.k_proj", weights,
                    layer.k_proj_weight, layer.k_proj_q, layer.is_quantized,
                    group_size, bits);
        load_linear(prefix + ".self_attn.v_proj", weights,
                    layer.v_proj_weight, layer.v_proj_q, layer.is_quantized,
                    group_size, bits);
        load_linear(prefix + ".self_attn.o_proj", weights,
                    layer.o_proj_weight, layer.o_proj_q, layer.is_quantized,
                    group_size, bits);

        load_linear(prefix + ".mlp.fc1", weights,
                    layer.fc1_weight, layer.fc1_q, layer.is_quantized,
                    group_size, bits);
        load_linear(prefix + ".mlp.fc2", weights,
                    layer.fc2_weight, layer.fc2_q, layer.is_quantized,
                    group_size, bits);

        layer.fc1_bias = find_optional(prefix + ".mlp.fc1.bias");
        layer.fc2_bias = find_optional(prefix + ".mlp.fc2.bias");
    }

    // --- Global transformer layers ---
    for (int i = 0; i < config_.num_global_layers; i++) {
        std::string prefix = "vision_tower.global_transformer.layers." + std::to_string(i);
        auto& layer = global_layers_[i];
        layer.is_gated = true;

        layer.input_layernorm_weight = find_weight(prefix + ".input_layernorm.weight");
        layer.input_layernorm_bias = find_weight(prefix + ".input_layernorm.bias");
        layer.post_attention_layernorm_weight = find_weight(
            prefix + ".post_attention_layernorm.weight");
        layer.post_attention_layernorm_bias = find_weight(
            prefix + ".post_attention_layernorm.bias");

        load_linear(prefix + ".self_attn.q_proj", weights,
                    layer.q_proj_weight, layer.q_proj_q, layer.is_quantized,
                    group_size, bits);
        load_linear(prefix + ".self_attn.k_proj", weights,
                    layer.k_proj_weight, layer.k_proj_q, layer.is_quantized,
                    group_size, bits);
        load_linear(prefix + ".self_attn.v_proj", weights,
                    layer.v_proj_weight, layer.v_proj_q, layer.is_quantized,
                    group_size, bits);
        load_linear(prefix + ".self_attn.o_proj", weights,
                    layer.o_proj_weight, layer.o_proj_q, layer.is_quantized,
                    group_size, bits);

        load_linear(prefix + ".mlp.fc1", weights,
                    layer.fc1_weight, layer.fc1_q, layer.is_quantized,
                    group_size, bits);
        load_linear(prefix + ".mlp.fc2", weights,
                    layer.fc2_weight, layer.fc2_q, layer.is_quantized,
                    group_size, bits);

        layer.fc1_bias = find_optional(prefix + ".mlp.fc1.bias");
        layer.fc2_bias = find_optional(prefix + ".mlp.fc2.bias");

        layer.gate_attn = find_optional(prefix + ".gate_attn");
        layer.gate_ffn = find_optional(prefix + ".gate_ffn");
    }
}

} // namespace gomlx
