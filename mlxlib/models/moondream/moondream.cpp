#include "models/moondream/moondream.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "third_party/json.hpp"

namespace gomlx {

using namespace mlx::core;

// GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static array gelu_tanh(const array& x) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    auto x3 = multiply(multiply(x, x), x);
    auto inner = multiply(array(sqrt_2_over_pi, float32),
                          add(x, multiply(array(0.044715f, float32), x3)));
    return multiply(multiply(array(0.5f, float32), x),
                    add(array(1.0f, float32), tanh(inner)));
}

// Exact GELU for vision projection (same as encoder)
static array gelu_exact(const array& x) {
    auto cdf = multiply(
        array(0.5f, float32),
        add(array(1.0f, float32),
            erf(divide(x, array(std::sqrt(2.0f), float32)))));
    return multiply(x, cdf);
}

MoondreamModel::MoondreamModel(const ModelConfig& config)
    : config_(config), md_config_(config.moondream_config) {
    layers_.resize(md_config_.text_num_layers);  // 24
    kv_cache_ = std::make_unique<KVCache>(md_config_.text_num_layers);
    vision_encoder_ = std::make_unique<MoondreamVisionEncoder>(md_config_);
    image_processor_ = std::make_unique<MoondreamImageProcessor>();
}

// --- Vision projection ---
// Vision encoder output [1, 729, 1152] -> project to text dim [1, N, 2048]
array MoondreamModel::project_vision(const array& vision_out) {
    // vision_out: [1, num_patches, vis_hidden_size] = [1, 729, 1152]
    int num_patches = vision_out.shape(1);
    int vis_dim = vision_out.shape(2);

    // Check if we need to concatenate pairs (proj_input_dim = 2 * vis_dim)
    array proj_input = vision_out;
    int proj_tokens = num_patches;

    if (md_config_.proj_input_dim == 2 * vis_dim) {
        // Concatenate pairs of adjacent tokens
        // 729 is odd, so pad to 730 with zeros
        if (num_patches % 2 != 0) {
            auto pad_token = zeros({1, 1, vis_dim}, float32);
            proj_input = concatenate({proj_input, pad_token}, 1);
            num_patches = num_patches + 1;
        }
        // Reshape [1, 730, 1152] -> [1, 365, 2304]
        proj_tokens = num_patches / 2;
        proj_input = reshape(proj_input, {1, proj_tokens, 2 * vis_dim});
    }

    // fc1 -> GELU -> fc2
    auto x = add(matmul(proj_input, transpose(*proj_fc1_weight_)), *proj_fc1_bias_);
    x = gelu_exact(x);
    x = add(matmul(x, transpose(*proj_fc2_weight_)), *proj_fc2_bias_);

    // Output: [1, proj_tokens, 2048]
    return x;
}

// --- Bidirectional prefix mask ---
// For positions 0..prefix_len-1: fully bidirectional (0)
// For positions prefix_len+: causal (0 for j<=i, -inf for j>i)
array MoondreamModel::build_prefix_mask(int prefix_len, int total_seq) {
    float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> mask_data(total_seq * total_seq, 0.0f);

    for (int i = 0; i < total_seq; i++) {
        for (int j = 0; j < total_seq; j++) {
            if (i >= prefix_len) {
                // Causal row: can attend to positions <= i, mask future
                if (j > i) {
                    mask_data[i * total_seq + j] = neg_inf;
                }
            }
            // Rows < prefix_len: all zeros (fully bidirectional)
        }
    }

    return array(mask_data.data(), {1, 1, total_seq, total_seq}, float32);
}

// --- Self-attention with partial RoPE ---
array MoondreamModel::self_attention(int layer_idx, const array& x,
                                      const std::optional<array>& mask_arr, int offset) {
    auto& layer = layers_[layer_idx];
    int batch = x.shape(0);
    int seq_len = x.shape(1);
    int num_heads = md_config_.text_num_heads;  // 32
    int head_dim = md_config_.text_head_dim;    // 64

    // Q, K, V projections
    auto q = add(matmul(x, transpose(*layer.q_proj_weight)), *layer.q_proj_bias);
    auto k = add(matmul(x, transpose(*layer.k_proj_weight)), *layer.k_proj_bias);
    auto v = add(matmul(x, transpose(*layer.v_proj_weight)), *layer.v_proj_bias);

    // Reshape: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    q = transpose(reshape(q, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});
    k = transpose(reshape(k, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});
    v = transpose(reshape(v, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});

    // Partial RoPE: only 32 of 64 head dims rotated
    q = fast::rope(q, md_config_.text_rope_partial_dims, false,
                   md_config_.text_rope_theta, 1.0f, offset);
    k = fast::rope(k, md_config_.text_rope_partial_dims, false,
                   md_config_.text_rope_theta, 1.0f, offset);

    // Update KV cache
    auto kv_pair = kv_cache_->update(layer_idx, k, v);
    auto& cached_k = kv_pair.first;
    auto& cached_v = kv_pair.second;

    // Scaled dot product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto sdpa = [&]() -> array {
        if (mask_arr.has_value()) {
            // Custom mask (bidirectional prefix during vision prefill)
            return fast::scaled_dot_product_attention(q, cached_k, cached_v, scale, "", mask_arr);
        } else if (seq_len > 1) {
            // Standard causal for text-only prefill
            return fast::scaled_dot_product_attention(q, cached_k, cached_v, scale, "causal");
        } else {
            // Single-token autoregressive: no mask needed
            return fast::scaled_dot_product_attention(q, cached_k, cached_v, scale, "");
        }
    };
    auto attn_out = sdpa();

    // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch, seq_len, num_heads * head_dim});

    // Output projection
    return add(matmul(attn_out, transpose(*layer.out_proj_weight)), *layer.out_proj_bias);
}

// --- MLP (GELU, not SwiGLU) ---
array MoondreamModel::mlp(int layer_idx, const array& x) {
    auto& layer = layers_[layer_idx];

    // fc1 -> GELU(tanh approx) -> fc2
    auto h = add(matmul(x, transpose(*layer.fc1_weight)), *layer.fc1_bias);
    h = gelu_tanh(h);
    return add(matmul(h, transpose(*layer.fc2_weight)), *layer.fc2_bias);
}

// --- Forward pass ---
array MoondreamModel::forward(const array& tokens, int offset) {
    float eps = md_config_.text_layer_norm_eps;

    // 1. Embed text tokens
    array input_tokens = tokens;
    if (input_tokens.ndim() == 1) {
        input_tokens = reshape(input_tokens, {1, static_cast<int>(input_tokens.shape(0))});
    }
    int batch = input_tokens.shape(0);
    int text_seq_len = input_tokens.shape(1);

    auto flat_tokens = reshape(input_tokens, {-1});
    auto text_embs = take(*wte_, flat_tokens, 0);
    text_embs = reshape(text_embs, {batch, text_seq_len, md_config_.text_hidden_size});

    // 2. If image pending: encode, project, prepend [BOS_emb, vision_embs, text_embs]
    array h = text_embs;
    std::optional<array> mask_arr;  // nullopt = no custom mask
    int actual_offset = offset;

    if (pending_image_.has_value()) {
        // Encode image through vision encoder
        auto vision_out = vision_encoder_->forward(pending_image_->pixel_values);

        // Project vision features to text dimension
        auto vision_projected = project_vision(vision_out);
        int vision_tokens = vision_projected.shape(1);

        // Get BOS embedding (token 0 is typically BOS for moondream)
        auto bos_emb = take(*wte_, array({0}, int32), 0);
        bos_emb = reshape(bos_emb, {1, 1, md_config_.text_hidden_size});

        // Prepend: [BOS_emb, vision_embs, text_embs]
        h = concatenate({bos_emb, vision_projected, text_embs}, 1);

        // Build bidirectional prefix mask
        // prefix_len = 1 (BOS) + vision_tokens
        int prefix_len = 1 + vision_tokens;
        int total_seq = h.shape(1);
        mask_arr = build_prefix_mask(prefix_len, total_seq);

        has_done_vision_prefill_ = true;
        pending_image_ = std::nullopt;
    }

    int seq_len = h.shape(1);

    // 3. Run 24 Phi-style blocks (parallel attn+MLP from same LayerNorm output)
    for (int i = 0; i < md_config_.text_num_layers; i++) {
        auto normed = fast::layer_norm(h, *layers_[i].ln_weight, *layers_[i].ln_bias, eps);
        auto attn_out = self_attention(i, normed, mask_arr, actual_offset);
        auto mlp_out = mlp(i, normed);  // SAME normed input (Phi-style parallel)
        h = add(add(h, attn_out), mlp_out);  // Both residuals added to original x
    }

    // 4. Final LayerNorm (part of lm_head) -> lm_head linear
    h = fast::layer_norm(h, *lm_head_ln_weight_, *lm_head_ln_bias_, eps);
    h = add(matmul(h, transpose(*lm_head_weight_)), *lm_head_bias_);

    return h;
}

// --- Weight loading ---
void MoondreamModel::load_weights(const std::string& path) {
    namespace fs = std::filesystem;

    // Collect safetensor files
    std::vector<std::string> safetensor_files;

    std::string index_file = path + "/model.safetensors.index.json";
    std::ifstream idx_stream(index_file);
    if (idx_stream.is_open()) {
        nlohmann::json idx_json;
        idx_stream >> idx_json;
        idx_stream.close();

        if (idx_json.contains("weight_map")) {
            std::unordered_map<std::string, bool> seen;
            for (auto& [key, val] : idx_json["weight_map"].items()) {
                std::string filename = val.get<std::string>();
                if (!seen.count(filename)) {
                    seen[filename] = true;
                    safetensor_files.push_back(path + "/" + filename);
                }
            }
            std::sort(safetensor_files.begin(), safetensor_files.end());
        }
    }

    if (safetensor_files.empty()) {
        std::string single = path + "/model.safetensors";
        if (fs::exists(single)) {
            safetensor_files.push_back(single);
        } else {
            for (const auto& entry : fs::directory_iterator(path)) {
                std::string fname = entry.path().filename().string();
                if (fname.find(".safetensors") != std::string::npos &&
                    fname.find("model-") == 0) {
                    safetensor_files.push_back(entry.path().string());
                }
            }
            std::sort(safetensor_files.begin(), safetensor_files.end());
        }
    }

    if (safetensor_files.empty()) {
        throw std::runtime_error("No safetensors files found in: " + path);
    }

    // Load all weights into a single map
    std::unordered_map<std::string, array> all_weights;
    for (const auto& file : safetensor_files) {
        auto [weights, metadata] = load_safetensors(file);
        for (auto& [name, arr] : weights) {
            all_weights.insert_or_assign(name, std::move(arr));
        }
    }

    auto find_weight = [&](const std::string& name) -> const array& {
        auto it = all_weights.find(name);
        if (it == all_weights.end()) {
            throw std::runtime_error("Missing Moondream weight: " + name);
        }
        return it->second;
    };

    auto find_weight_optional = [&](const std::string& name) -> std::optional<array> {
        auto it = all_weights.find(name);
        if (it == all_weights.end()) return std::nullopt;
        return it->second;
    };

    // --- Pass vision weights to VisionEncoder ---
    std::unordered_map<std::string, array> vision_weights;
    for (auto& [name, arr] : all_weights) {
        if (name.find("vision_encoder.encoder.") == 0) {
            vision_weights.insert_or_assign(name, arr);
        }
    }
    vision_encoder_->load_weights(vision_weights);

    // --- Vision projection MLP ---
    proj_fc1_weight_ = find_weight("vision_encoder.projection.mlp.fc1.weight");
    proj_fc1_bias_ = find_weight("vision_encoder.projection.mlp.fc1.bias");
    proj_fc2_weight_ = find_weight("vision_encoder.projection.mlp.fc2.weight");
    proj_fc2_bias_ = find_weight("vision_encoder.projection.mlp.fc2.bias");

    // --- Token embedding ---
    wte_ = find_weight("text_model.transformer.embd.wte.weight");

    // --- Decoder layers ---
    for (int i = 0; i < md_config_.text_num_layers; i++) {
        std::string prefix = "text_model.transformer.h." + std::to_string(i);
        auto& layer = layers_[i];

        // LayerNorm (weight + bias)
        layer.ln_weight = find_weight(prefix + ".ln.weight");
        layer.ln_bias = find_weight(prefix + ".ln.bias");

        // Fused QKV: [6144, 2048] -> split into Q/K/V each [2048, 2048]
        auto wqkv = find_weight(prefix + ".mixer.Wqkv.weight");
        int h = md_config_.text_hidden_size;  // 2048
        layer.q_proj_weight = slice(wqkv, {0, 0}, {h, h});
        layer.k_proj_weight = slice(wqkv, {h, 0}, {2 * h, h});
        layer.v_proj_weight = slice(wqkv, {2 * h, 0}, {3 * h, h});

        // Fused QKV bias: [6144] -> split into Q/K/V each [2048]
        auto bqkv = find_weight(prefix + ".mixer.Wqkv.bias");
        layer.q_proj_bias = slice(bqkv, {0}, {h});
        layer.k_proj_bias = slice(bqkv, {h}, {2 * h});
        layer.v_proj_bias = slice(bqkv, {2 * h}, {3 * h});

        // Output projection
        layer.out_proj_weight = find_weight(prefix + ".mixer.out_proj.weight");
        layer.out_proj_bias = find_weight(prefix + ".mixer.out_proj.bias");

        // MLP
        layer.fc1_weight = find_weight(prefix + ".mlp.fc1.weight");
        layer.fc1_bias = find_weight(prefix + ".mlp.fc1.bias");
        layer.fc2_weight = find_weight(prefix + ".mlp.fc2.weight");
        layer.fc2_bias = find_weight(prefix + ".mlp.fc2.bias");
    }

    // --- LM head LayerNorm ---
    lm_head_ln_weight_ = find_weight("text_model.lm_head.ln.weight");
    lm_head_ln_bias_ = find_weight("text_model.lm_head.ln.bias");

    // --- LM head linear ---
    lm_head_weight_ = find_weight("text_model.lm_head.linear.weight");
    lm_head_bias_ = find_weight("text_model.lm_head.linear.bias");
}

void MoondreamModel::reset_cache() {
    kv_cache_->reset();
    pending_image_ = std::nullopt;
    has_done_vision_prefill_ = false;
}

} // namespace gomlx
