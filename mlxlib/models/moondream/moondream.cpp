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
    // traditional=false: llama-style interleaved pairs (matching starmie RoPE output)
    // offset is computed once before the layer loop to avoid reading stale kv_cache length
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

    // Compute RoPE offset ONCE before the layer loop.
    // kv_cache_->sequence_length() returns the number of tokens already cached.
    // All layers must use the same offset (reading inside the loop would give
    // wrong values after layer 0's cache update).
    int rope_offset = kv_cache_->sequence_length();

    if (pending_image_.has_value()) {
        // Encode image through vision encoder
        auto vision_out = vision_encoder_->forward(pending_image_->pixel_values);

        // Project vision features to text dimension
        auto vision_projected = project_vision(vision_out);
        int vision_tokens = vision_projected.shape(1);

        // Get BOS embedding (token 0 = <|endoftext|> used as BOS in starmie-v1 tokenizer)
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
        auto attn_out = self_attention(i, normed, mask_arr, rope_offset);
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
        if (name.find("model.vision.") == 0) {
            vision_weights.insert_or_assign(name, arr);
        }
    }
    vision_encoder_->load_weights(vision_weights);

    // --- Vision projection MLP ---
    proj_fc1_weight_ = find_weight("model.vision.proj_mlp.fc1.weight");
    proj_fc1_bias_ = find_weight("model.vision.proj_mlp.fc1.bias");
    proj_fc2_weight_ = find_weight("model.vision.proj_mlp.fc2.weight");
    proj_fc2_bias_ = find_weight("model.vision.proj_mlp.fc2.bias");

    // --- Token embedding (note: key is "model.text.wte" with no .weight suffix) ---
    wte_ = find_weight("model.text.wte");

    // --- Decoder layers ---
    for (int i = 0; i < md_config_.text_num_layers; i++) {
        std::string prefix = "model.text.blocks." + std::to_string(i);
        auto& layer = layers_[i];

        // LayerNorm (weight + bias)
        layer.ln_weight = find_weight(prefix + ".ln.weight");
        layer.ln_bias = find_weight(prefix + ".ln.bias");

        // Fused QKV: [6144, 2048] -> split into Q/K/V each [2048, 2048]
        auto wqkv = find_weight(prefix + ".attn.qkv.weight");
        int h = md_config_.text_hidden_size;  // 2048
        layer.q_proj_weight = slice(wqkv, {0, 0}, {h, h});
        layer.k_proj_weight = slice(wqkv, {h, 0}, {2 * h, h});
        layer.v_proj_weight = slice(wqkv, {2 * h, 0}, {3 * h, h});

        // Fused QKV bias: [6144] -> split into Q/K/V each [2048]
        auto bqkv = find_weight(prefix + ".attn.qkv.bias");
        layer.q_proj_bias = slice(bqkv, {0}, {h});
        layer.k_proj_bias = slice(bqkv, {h}, {2 * h});
        layer.v_proj_bias = slice(bqkv, {2 * h}, {3 * h});

        // Output projection
        layer.out_proj_weight = find_weight(prefix + ".attn.proj.weight");
        layer.out_proj_bias = find_weight(prefix + ".attn.proj.bias");

        // MLP
        layer.fc1_weight = find_weight(prefix + ".mlp.fc1.weight");
        layer.fc1_bias = find_weight(prefix + ".mlp.fc1.bias");
        layer.fc2_weight = find_weight(prefix + ".mlp.fc2.weight");
        layer.fc2_bias = find_weight(prefix + ".mlp.fc2.bias");
    }

    // --- Post LayerNorm (separate from lm_head) ---
    lm_head_ln_weight_ = find_weight("model.text.post_ln.weight");
    lm_head_ln_bias_ = find_weight("model.text.post_ln.bias");

    // --- LM head (no nested .linear) ---
    lm_head_weight_ = find_weight("model.text.lm_head.weight");
    lm_head_bias_ = find_weight("model.text.lm_head.bias");

    // --- Region model weights (optional — older models may not have these) ---
    auto coord_feat = find_weight_optional("model.region.coord_features");
    if (coord_feat.has_value()) {
        coord_features_ = *coord_feat;  // already [1, 128]

        coord_encoder_weight_ = find_weight("model.region.coord_encoder.weight");
        coord_encoder_bias_ = find_weight("model.region.coord_encoder.bias");

        coord_dec_fc1_weight_ = find_weight("model.region.coord_decoder.fc1.weight");
        coord_dec_fc1_bias_ = find_weight("model.region.coord_decoder.fc1.bias");
        coord_dec_fc2_weight_ = find_weight("model.region.coord_decoder.fc2.weight");
        coord_dec_fc2_bias_ = find_weight("model.region.coord_decoder.fc2.bias");

        size_features_ = find_weight("model.region.size_features");  // already [2, 256]

        size_encoder_weight_ = find_weight("model.region.size_encoder.weight");
        size_encoder_bias_ = find_weight("model.region.size_encoder.bias");

        size_dec_fc1_weight_ = find_weight("model.region.size_decoder.fc1.weight");
        size_dec_fc1_bias_ = find_weight("model.region.size_decoder.fc1.bias");
        size_dec_fc2_weight_ = find_weight("model.region.size_decoder.fc2.weight");
        size_dec_fc2_bias_ = find_weight("model.region.size_decoder.fc2.bias");
    }
}

// --- forward_with_hidden: returns {logits, hidden_state} ---
std::pair<array, array> MoondreamModel::forward_with_hidden(const array& tokens, int offset) {
    float eps = md_config_.text_layer_norm_eps;

    array input_tokens = tokens;
    if (input_tokens.ndim() == 1) {
        input_tokens = reshape(input_tokens, {1, static_cast<int>(input_tokens.shape(0))});
    }
    int batch = input_tokens.shape(0);
    int text_seq_len = input_tokens.shape(1);

    auto flat_tokens = reshape(input_tokens, {-1});
    auto text_embs = take(*wte_, flat_tokens, 0);
    text_embs = reshape(text_embs, {batch, text_seq_len, md_config_.text_hidden_size});

    array h = text_embs;
    std::optional<array> mask_arr;
    int rope_offset = kv_cache_->sequence_length();

    if (pending_image_.has_value()) {
        auto vision_out = vision_encoder_->forward(pending_image_->pixel_values);
        auto vision_projected = project_vision(vision_out);
        int vision_tokens = vision_projected.shape(1);
        auto bos_emb = take(*wte_, array({0}, int32), 0);
        bos_emb = reshape(bos_emb, {1, 1, md_config_.text_hidden_size});
        h = concatenate({bos_emb, vision_projected, text_embs}, 1);
        int prefix_len = 1 + vision_tokens;
        int total_seq = h.shape(1);
        mask_arr = build_prefix_mask(prefix_len, total_seq);
        has_done_vision_prefill_ = true;
        pending_image_ = std::nullopt;
    }

    for (int i = 0; i < md_config_.text_num_layers; i++) {
        auto normed = fast::layer_norm(h, *layers_[i].ln_weight, *layers_[i].ln_bias, eps);
        auto attn_out = self_attention(i, normed, mask_arr, rope_offset);
        auto mlp_out = mlp(i, normed);
        h = add(add(h, attn_out), mlp_out);
    }

    // Final LayerNorm — this is the hidden state before lm_head
    auto hidden = fast::layer_norm(h, *lm_head_ln_weight_, *lm_head_ln_bias_, eps);
    auto logits = add(matmul(hidden, transpose(*lm_head_weight_)), *lm_head_bias_);

    return {logits, hidden};
}

// --- Fourier feature encoding ---
// x: (1, D_in), w: (D_in, D_out) → f = 2π * (x @ w) → concat [cos(f), sin(f)]
array MoondreamModel::fourier_features(const array& x, const array& w) {
    auto f = multiply(array(2.0f * M_PI, float32), matmul(x, w));
    return concatenate({cos(f), sin(f)}, -1);
}

// --- Coordinate encode/decode ---
array MoondreamModel::encode_coordinate(float coord) {
    auto feat = fourier_features(array({coord}, {1, 1}, float32), *coord_features_);
    return add(matmul(feat, transpose(*coord_encoder_weight_)), *coord_encoder_bias_);
}

array MoondreamModel::decode_coordinate(const array& hidden) {
    auto h = gelu_tanh(add(matmul(hidden, transpose(*coord_dec_fc1_weight_)), *coord_dec_fc1_bias_));
    return add(matmul(h, transpose(*coord_dec_fc2_weight_)), *coord_dec_fc2_bias_);
}

// --- Size encode/decode ---
array MoondreamModel::encode_size(float w, float h) {
    auto feat = fourier_features(array({w, h}, {1, 2}, float32), *size_features_);
    return add(matmul(feat, transpose(*size_encoder_weight_)), *size_encoder_bias_);
}

array MoondreamModel::decode_size(const array& hidden) {
    auto h = gelu_tanh(add(matmul(hidden, transpose(*size_dec_fc1_weight_)), *size_dec_fc1_bias_));
    auto out = add(matmul(h, transpose(*size_dec_fc2_weight_)), *size_dec_fc2_bias_);
    return reshape(out, {2, 1024});
}

// --- Core region generation loop ---
std::vector<std::pair<float,float>> MoondreamModel::generate_points(
    array hidden, int next_token, int pos, bool include_size, int max_objects) {

    std::vector<std::pair<float,float>> results;

    // hidden is the last hidden state from prefill, shape [1, seq, 2048]
    // Take the last position
    int last_pos = hidden.shape(1) - 1;
    auto h = slice(hidden, {0, last_pos, 0}, {1, last_pos + 1, hidden.shape(2)});

    while (next_token != 0 && static_cast<int>(results.size()) < max_objects) {
        // Decode x-coordinate from hidden state
        auto x_logits = decode_coordinate(h);
        eval(x_logits);
        int x_bin = argmax(x_logits, -1).item<int>();
        float x_coord = static_cast<float>(x_bin) / 1024.0f;

        // Encode x and feed through text model
        auto x_emb = encode_coordinate(x_coord);
        x_emb = reshape(x_emb, {1, 1, md_config_.text_hidden_size});

        // Feed x embedding as a token embedding directly through transformer layers
        {
            float eps = md_config_.text_layer_norm_eps;
            int rope_offset = kv_cache_->sequence_length();
            array cur = x_emb;
            for (int i = 0; i < md_config_.text_num_layers; i++) {
                auto normed = fast::layer_norm(cur, *layers_[i].ln_weight, *layers_[i].ln_bias, eps);
                auto attn_out = self_attention(i, normed, std::nullopt, rope_offset);
                auto mlp_out = mlp(i, normed);
                cur = add(add(cur, attn_out), mlp_out);
            }
            h = fast::layer_norm(cur, *lm_head_ln_weight_, *lm_head_ln_bias_, eps);
        }

        // Decode y-coordinate
        auto y_logits = decode_coordinate(h);
        eval(y_logits);
        int y_bin = argmax(y_logits, -1).item<int>();
        float y_coord = static_cast<float>(y_bin) / 1024.0f;

        if (include_size) {
            // Encode y and feed through text model
            auto y_emb = encode_coordinate(y_coord);
            y_emb = reshape(y_emb, {1, 1, md_config_.text_hidden_size});

            {
                float eps = md_config_.text_layer_norm_eps;
                int rope_offset = kv_cache_->sequence_length();
                array cur = y_emb;
                for (int i = 0; i < md_config_.text_num_layers; i++) {
                    auto normed = fast::layer_norm(cur, *layers_[i].ln_weight, *layers_[i].ln_bias, eps);
                    auto attn_out = self_attention(i, normed, std::nullopt, rope_offset);
                    auto mlp_out = mlp(i, normed);
                    cur = add(add(cur, attn_out), mlp_out);
                }
                h = fast::layer_norm(cur, *lm_head_ln_weight_, *lm_head_ln_bias_, eps);
            }

            // Decode size (width, height)
            auto size_logits = decode_size(h);
            eval(size_logits);
            auto w_logits = slice(size_logits, {0, 0}, {1, 1024});
            auto h_logits = slice(size_logits, {1, 0}, {2, 1024});
            int w_bin = argmax(w_logits, -1).item<int>();
            int h_bin = argmax(h_logits, -1).item<int>();
            float w_size = std::pow(2.0f, (static_cast<float>(w_bin) / 1023.0f) * 10.0f - 10.0f);
            float h_size = std::pow(2.0f, (static_cast<float>(h_bin) / 1023.0f) * 10.0f - 10.0f);

            // Store as bounding box center + size → min/max will be computed in detect_objects
            // For now store x_center, y_center, width, height
            // We'll encode as two pairs: (x_coord, y_coord) and (w_size, h_size)
            results.push_back({x_coord, y_coord});
            results.push_back({w_size, h_size});

            // Encode size and feed through text model to get next token
            auto size_emb = encode_size(w_size, h_size);
            size_emb = reshape(size_emb, {1, 1, md_config_.text_hidden_size});

            {
                float eps = md_config_.text_layer_norm_eps;
                int rope_offset = kv_cache_->sequence_length();
                array cur = size_emb;
                for (int i = 0; i < md_config_.text_num_layers; i++) {
                    auto normed = fast::layer_norm(cur, *layers_[i].ln_weight, *layers_[i].ln_bias, eps);
                    auto attn_out = self_attention(i, normed, std::nullopt, rope_offset);
                    auto mlp_out = mlp(i, normed);
                    cur = add(add(cur, attn_out), mlp_out);
                }
                h = fast::layer_norm(cur, *lm_head_ln_weight_, *lm_head_ln_bias_, eps);
            }
        } else {
            // Point mode: store just x,y
            results.push_back({x_coord, y_coord});

            // Encode y and feed through text model to get next token
            auto y_emb = encode_coordinate(y_coord);
            y_emb = reshape(y_emb, {1, 1, md_config_.text_hidden_size});

            {
                float eps = md_config_.text_layer_norm_eps;
                int rope_offset = kv_cache_->sequence_length();
                array cur = y_emb;
                for (int i = 0; i < md_config_.text_num_layers; i++) {
                    auto normed = fast::layer_norm(cur, *layers_[i].ln_weight, *layers_[i].ln_bias, eps);
                    auto attn_out = self_attention(i, normed, std::nullopt, rope_offset);
                    auto mlp_out = mlp(i, normed);
                    cur = add(add(cur, attn_out), mlp_out);
                }
                h = fast::layer_norm(cur, *lm_head_ln_weight_, *lm_head_ln_bias_, eps);
            }
        }

        // Get next token from logits
        auto logits = add(matmul(h, transpose(*lm_head_weight_)), *lm_head_bias_);
        eval(logits);
        next_token = argmax(reshape(logits, {-1})).item<int>();
    }

    return results;
}

// --- detect_points: point detection ---
std::vector<std::pair<float, float>> MoondreamModel::detect_points(
    const std::vector<int>& prompt_tokens, int max_objects) {

    if (!coord_features_.has_value()) {
        throw std::runtime_error("Region model weights not loaded — point detection unavailable");
    }

    // Create token array
    array token_arr(prompt_tokens.data(), {1, static_cast<int>(prompt_tokens.size())}, int32);

    // Forward with hidden to get both logits and hidden state
    auto [logits, hidden] = forward_with_hidden(token_arr, 0);
    eval(logits);
    eval(hidden);

    // Get first predicted token (greedy)
    int last_pos = logits.shape(1) - 1;
    auto last_logits = slice(logits, {0, last_pos, 0}, {1, last_pos + 1, logits.shape(2)});
    int next_token = argmax(reshape(last_logits, {-1})).item<int>();

    // Generate points (no size)
    return generate_points(hidden, next_token, static_cast<int>(prompt_tokens.size()), false, max_objects);
}

// --- detect_objects: bounding box detection ---
std::vector<std::array<float, 4>> MoondreamModel::detect_objects(
    const std::vector<int>& prompt_tokens, int max_objects) {

    if (!coord_features_.has_value() || !size_features_.has_value()) {
        throw std::runtime_error("Region model weights not loaded — object detection unavailable");
    }

    array token_arr(prompt_tokens.data(), {1, static_cast<int>(prompt_tokens.size())}, int32);

    auto [logits, hidden] = forward_with_hidden(token_arr, 0);
    eval(logits);
    eval(hidden);

    int last_pos = logits.shape(1) - 1;
    auto last_logits = slice(logits, {0, last_pos, 0}, {1, last_pos + 1, logits.shape(2)});
    int next_token = argmax(reshape(last_logits, {-1})).item<int>();

    // Generate points with size info (pairs: center, size)
    auto raw = generate_points(hidden, next_token, static_cast<int>(prompt_tokens.size()), true, max_objects);

    // Convert center+size pairs to bounding boxes
    std::vector<std::array<float, 4>> boxes;
    for (size_t i = 0; i + 1 < raw.size(); i += 2) {
        float cx = raw[i].first;
        float cy = raw[i].second;
        float w = raw[i + 1].first;
        float bh = raw[i + 1].second;

        float x_min = std::max(0.0f, cx - w / 2.0f);
        float y_min = std::max(0.0f, cy - bh / 2.0f);
        float x_max = std::min(1.0f, cx + w / 2.0f);
        float y_max = std::min(1.0f, cy + bh / 2.0f);
        boxes.push_back({x_min, y_min, x_max, y_max});
    }

    return boxes;
}

void MoondreamModel::reset_cache() {
    kv_cache_->reset();
    pending_image_ = std::nullopt;
    has_done_vision_prefill_ = false;
}

} // namespace gomlx
