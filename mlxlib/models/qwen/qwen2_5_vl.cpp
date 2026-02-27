#include "models/qwen/qwen2_5_vl.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "third_party/json.hpp"

namespace gomlx {

using namespace mlx::core;

// SiLU activation: x * sigmoid(x)
static array silu(const array& x) {
    return multiply(x, sigmoid(x));
}

Qwen2_5VLModel::Qwen2_5VLModel(const ModelConfig& config)
    : config_(config) {
    head_dim_ = config_.get_head_dim();  // 3584/28 = 128
    num_kv_groups_ = config_.num_attention_heads / config_.num_key_value_heads;  // 28/4 = 7
    layers_.resize(config_.num_hidden_layers);  // 28
    kv_cache_ = std::make_unique<KVCache>(config_.num_hidden_layers);
    vision_encoder_ = std::make_unique<QwenVisionEncoder>(config_.qwen_vision_config);
    image_processor_ = std::make_unique<QwenImageProcessor>(config_.qwen_vision_config);
}

void Qwen2_5VLModel::set_image(const QwenImageProcessorResult& result) {
    pending_image_ = result;
}

int Qwen2_5VLModel::pending_vision_tokens() const {
    if (pending_image_.has_value()) return pending_image_->num_vision_tokens;
    return 0;
}

void Qwen2_5VLModel::load_linear(
    const std::string& prefix,
    const std::unordered_map<std::string, array>& weights,
    std::optional<array>& dense_weight,
    std::unique_ptr<QuantizedLinear>& q_linear,
    bool& is_quantized) {

    auto w_it = weights.find(prefix + ".weight");
    auto s_it = weights.find(prefix + ".scales");

    if (s_it != weights.end() && w_it != weights.end()) {
        is_quantized = true;
        auto b_it = weights.find(prefix + ".biases");
        if (b_it != weights.end()) {
            q_linear = std::make_unique<QuantizedLinear>(
                w_it->second, s_it->second, b_it->second, group_size_, bits_);
        } else {
            auto zero_biases = zeros(s_it->second.shape(), s_it->second.dtype());
            q_linear = std::make_unique<QuantizedLinear>(
                w_it->second, s_it->second, zero_biases, group_size_, bits_);
        }
    } else if (w_it != weights.end()) {
        dense_weight = w_it->second;
    }
}

array Qwen2_5VLModel::linear_forward(
    const array& input,
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

// --- mRoPE position computation ---

array Qwen2_5VLModel::compute_mrope_positions(int seq_len, int grid_h, int grid_w,
                                                const array& tokens) {
    // positions: [3, seq_len] -- one row per mRoPE section
    // During autoregressive (single token), all sections same position
    if (seq_len == 1) {
        std::vector<int> pos_data = {mrope_next_pos_, mrope_next_pos_, mrope_next_pos_};
        mrope_next_pos_++;
        return array(pos_data.data(), {3, 1}, int32);
    }

    // Prefill: need to assign positions
    auto flat = reshape(tokens, {-1});
    eval(flat);
    std::vector<int> token_vec(flat.data<int>(), flat.data<int>() + flat.size());

    // Find image_pad token positions
    std::vector<int> image_positions;
    for (int i = 0; i < seq_len; i++) {
        if (token_vec[i] == config_.image_token_index) {
            image_positions.push_back(i);
        }
    }

    // Build position arrays for 3 sections
    std::vector<int> pos_t(seq_len), pos_h(seq_len), pos_w(seq_len);

    if (image_positions.empty() || grid_h == 0) {
        // Text-only prefill: sequential positions
        for (int i = 0; i < seq_len; i++) {
            pos_t[i] = pos_h[i] = pos_w[i] = i;
        }
        mrope_next_pos_ = seq_len;
    } else {
        // Vision + text prefill
        int merge = config_.qwen_vision_config.spatial_merge_size;  // 2
        int merged_h = grid_h / merge;
        int merged_w = grid_w / merge;

        int text_pos = 0;
        int first_img = image_positions[0];

        // Positions before image: sequential text positions
        for (int i = 0; i < first_img; i++) {
            pos_t[i] = pos_h[i] = pos_w[i] = text_pos++;
        }

        // Image positions: 3D grid positions
        // temporal is always the same (single image), but offset by text_pos
        for (int i = 0; i < (int)image_positions.size(); i++) {
            int p = image_positions[i];
            int row = i / merged_w;
            int col = i % merged_w;
            pos_t[p] = text_pos;       // temporal: all same (single image)
            pos_h[p] = text_pos + row;  // height position
            pos_w[p] = text_pos + col;  // width position
        }

        // Max position used by vision across all dims
        int max_vision_pos = text_pos + std::max(merged_h - 1, merged_w - 1);

        // Positions after image: resume from max_vision_pos + 1
        int last_img = image_positions.back();
        text_pos = max_vision_pos + 1;
        for (int i = last_img + 1; i < seq_len; i++) {
            pos_t[i] = pos_h[i] = pos_w[i] = text_pos++;
        }

        mrope_next_pos_ = text_pos;
    }

    // Stack into [3, seq_len]
    std::vector<int> all_pos;
    all_pos.insert(all_pos.end(), pos_t.begin(), pos_t.end());
    all_pos.insert(all_pos.end(), pos_h.begin(), pos_h.end());
    all_pos.insert(all_pos.end(), pos_w.begin(), pos_w.end());

    return array(all_pos.data(), {3, seq_len}, int32);
}

// --- mRoPE application ---

array Qwen2_5VLModel::apply_mrope(const array& x, const array& positions,
                                    const std::vector<int>& sections) {
    // x: [batch, num_heads, seq, head_dim] (head_dim=128)
    // positions: [3, seq] — one row per mRoPE section (temporal, height, width)
    // sections: [16, 24, 24] -> dims [32, 48, 48] = 128
    //
    // Key: frequencies are always computed from the FULL head_dim (128),
    // not the individual section sizes. Each section selects different positions
    // but uses the global frequency table.

    eval(positions);
    int seq_len = positions.shape(1);
    int full_dim = head_dim_;   // 128
    int half_dim = full_dim / 2; // 64

    // Read all position data: [3 * seq_len]
    auto pos_data = std::vector<int>(positions.data<int>(),
                                      positions.data<int>() + 3 * seq_len);

    // Check if all 3 position dims are identical AND sequential (text-only fast path)
    bool all_same = true;
    for (int i = 0; i < seq_len && all_same; i++) {
        if (pos_data[i] != pos_data[seq_len + i] ||
            pos_data[i] != pos_data[2 * seq_len + i]) {
            all_same = false;
        }
    }
    bool is_sequential = all_same;
    if (is_sequential) {
        for (int i = 1; i < seq_len; i++) {
            if (pos_data[i] != pos_data[i - 1] + 1) {
                is_sequential = false;
                break;
            }
        }
    }

    if (is_sequential && seq_len > 0) {
        // All sections same sequential positions → standard RoPE on full head_dim
        int offset = pos_data[0];
        return fast::rope(x, full_dim, false, config_.rope_theta, 1.0f, offset);
    }

    // Manual path: build cos/sin for full head_dim with section-specific positions
    // Global frequency table: inv_freq[i] = 1/theta^(2i/128)
    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; i++) {
        inv_freq[i] = 1.0f / std::pow(config_.rope_theta, (2.0f * i) / static_cast<float>(full_dim));
    }

    // Map each of the 128 head dims to its mRoPE section (0=temporal, 1=height, 2=width)
    std::vector<int> dim_to_section(full_dim);
    int dim_offset = 0;
    for (int s = 0; s < 3; s++) {
        int sec_dims = sections[s] * 2;
        for (int d = dim_offset; d < dim_offset + sec_dims; d++) {
            dim_to_section[d] = s;
        }
        dim_offset += sec_dims;
    }

    // Build cos/sin tables [seq, 128]
    std::vector<float> cos_data(seq_len * full_dim);
    std::vector<float> sin_data(seq_len * full_dim);

    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < full_dim; d++) {
            int sec = dim_to_section[d];
            int pos = pos_data[sec * seq_len + t];
            int freq_idx = d % half_dim;  // freq index wraps (due to emb = cat(freqs, freqs))
            float angle = static_cast<float>(pos) * inv_freq[freq_idx];
            cos_data[t * full_dim + d] = std::cos(angle);
            sin_data[t * full_dim + d] = std::sin(angle);
        }
    }

    auto cos_arr = array(cos_data.data(), {1, 1, seq_len, full_dim}, float32);
    auto sin_arr = array(sin_data.data(), {1, 1, seq_len, full_dim}, float32);

    // Standard half-rotation: rotate_half(x) = [-x_second_half, x_first_half]
    auto x_first = slice(x, {0, 0, 0, 0},
                          {x.shape(0), x.shape(1), x.shape(2), half_dim});
    auto x_second = slice(x, {0, 0, 0, half_dim},
                           {x.shape(0), x.shape(1), x.shape(2), full_dim});
    auto x_rotated = concatenate({negative(x_second), x_first}, -1);

    return add(multiply(x, cos_arr), multiply(x_rotated, sin_arr));
}

// --- Self-attention ---

array Qwen2_5VLModel::self_attention(int layer_idx, const array& x,
                                      const array& position_ids) {
    auto& layer = layers_[layer_idx];
    int batch = x.shape(0);
    int seq_len = x.shape(1);
    std::string ln = "layer" + std::to_string(layer_idx);

    // Q, K, V projections
    array q = linear_forward(x, layer.q_proj_weight, layer.q_proj_q.get(), ln + ".q");
    if (layer.q_proj_bias.has_value()) q = add(q, *layer.q_proj_bias);

    array k = linear_forward(x, layer.k_proj_weight, layer.k_proj_q.get(), ln + ".k");
    if (layer.k_proj_bias.has_value()) k = add(k, *layer.k_proj_bias);

    array v = linear_forward(x, layer.v_proj_weight, layer.v_proj_q.get(), ln + ".v");
    if (layer.v_proj_bias.has_value()) v = add(v, *layer.v_proj_bias);

    // Reshape: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    q = reshape(q, {batch, seq_len, config_.num_attention_heads, head_dim_});
    q = transpose(q, {0, 2, 1, 3});

    k = reshape(k, {batch, seq_len, config_.num_key_value_heads, head_dim_});
    k = transpose(k, {0, 2, 1, 3});

    v = reshape(v, {batch, seq_len, config_.num_key_value_heads, head_dim_});
    v = transpose(v, {0, 2, 1, 3});

    // Apply mRoPE to Q and K
    q = apply_mrope(q, position_ids, config_.mrope_sections);
    k = apply_mrope(k, position_ids, config_.mrope_sections);

    // Update KV cache
    auto [cached_k, cached_v] = kv_cache_->update(layer_idx, k, v);

    // Scaled dot product attention with causal mask
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    array attn_out = fast::scaled_dot_product_attention(q, cached_k, cached_v, scale, "causal");

    // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch, seq_len, config_.num_attention_heads * head_dim_});

    // Output projection (NO bias for o_proj)
    return linear_forward(attn_out, layer.o_proj_weight, layer.o_proj_q.get(), ln + ".o");
}

// --- MLP ---

array Qwen2_5VLModel::mlp(int layer_idx, const array& x) {
    auto& layer = layers_[layer_idx];
    std::string ln = "layer" + std::to_string(layer_idx);

    // SwiGLU: silu(gate_proj(x)) * up_proj(x)
    array gate = linear_forward(x, layer.gate_proj_weight, layer.gate_proj_q.get(), ln + ".gate");
    array up = linear_forward(x, layer.up_proj_weight, layer.up_proj_q.get(), ln + ".up");
    array activated = multiply(silu(gate), up);

    return linear_forward(activated, layer.down_proj_weight, layer.down_proj_q.get(), ln + ".down");
}

// --- Forward pass ---

array Qwen2_5VLModel::forward(const array& tokens, int offset) {
    // 1. Embed tokens
    array input_tokens = tokens;
    if (input_tokens.ndim() == 1) {
        input_tokens = reshape(input_tokens, {1, static_cast<int>(input_tokens.shape(0))});
    }
    int batch = input_tokens.shape(0);
    int seq_len = input_tokens.shape(1);

    array flat_tokens = reshape(input_tokens, {-1});
    array h = take(*embed_tokens_, flat_tokens, 0);
    h = reshape(h, {batch, seq_len, config_.hidden_size});

    // 2. If pending image, run vision encoder and replace image_pad token embeddings
    int grid_h = 0, grid_w = 0;
    if (pending_image_.has_value()) {
        auto& img = *pending_image_;
        auto vision_features = vision_encoder_->forward(img.pixel_values, img.grid_h, img.grid_w);
        grid_h = img.grid_h;
        grid_w = img.grid_w;

        // Find positions of image_token_id in the input tokens and replace embeddings
        auto flat = reshape(input_tokens, {-1});
        eval(flat);
        std::vector<int> flat_vec(flat.data<int>(), flat.data<int>() + flat.size());

        eval(vision_features);
        eval(h);

        // Build list of image token positions
        std::vector<int> image_positions;
        for (int i = 0; i < (int)flat_vec.size(); i++) {
            if (flat_vec[i] == config_.image_token_index) {
                image_positions.push_back(i);
            }
        }

        if (!image_positions.empty() && (int)image_positions.size() == img.num_vision_tokens) {
            // Replace embeddings at image_pad positions with vision features
            auto h_flat = reshape(h, {seq_len, config_.hidden_size});
            auto vf_flat = reshape(vision_features, {img.num_vision_tokens, config_.hidden_size});

            // Build index mapping: for each position, either take from vision or text
            std::vector<int> src_indices(seq_len);
            std::vector<int> is_vision(seq_len, 0);
            int v_idx = 0;
            for (int i = 0; i < seq_len; i++) {
                if (v_idx < (int)image_positions.size() && i == image_positions[v_idx]) {
                    src_indices[i] = v_idx;
                    is_vision[i] = 1;
                    v_idx++;
                } else {
                    src_indices[i] = i;
                    is_vision[i] = 0;
                }
            }

            // Gather from vision_features for all positions (clamped)
            auto vision_indices = array(src_indices.data(), {seq_len}, int32);
            auto vision_gathered = take(vf_flat, vision_indices, 0);

            // Select: vision features at image positions, text embeddings elsewhere
            auto is_vis_arr = array(is_vision.data(), {seq_len, 1}, int32);
            is_vis_arr = astype(is_vis_arr, float32);

            h_flat = add(
                multiply(vision_gathered, is_vis_arr),
                multiply(h_flat, subtract(array(1.0f, float32), is_vis_arr))
            );
            h = reshape(h_flat, {batch, seq_len, config_.hidden_size});
        }

        has_done_vision_prefill_ = true;
    }

    // 3. Compute position IDs for mRoPE
    array position_ids = compute_mrope_positions(seq_len, grid_h, grid_w, input_tokens);

    // 4. Clear pending image after processing
    if (pending_image_.has_value()) {
        pending_image_ = std::nullopt;
    }

    // 5. Decoder layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        array normed = fast::rms_norm(h, *layers_[i].input_layernorm_weight, config_.rms_norm_eps);
        array attn_out = self_attention(i, normed, position_ids);
        h = add(h, attn_out);

        normed = fast::rms_norm(h, *layers_[i].post_attention_layernorm_weight, config_.rms_norm_eps);
        array mlp_out = mlp(i, normed);
        h = add(h, mlp_out);
    }

    // 6. Final norm + LM head
    h = fast::rms_norm(h, *final_norm_weight_, config_.rms_norm_eps);
    if (has_lm_head_) {
        h = linear_forward(h, lm_head_weight_, lm_head_q_.get(), "lm_head");
    } else {
        h = matmul(h, transpose(*embed_tokens_));
    }

    return h;
}

// --- Weight loading ---

void Qwen2_5VLModel::load_weights(const std::string& path) {
    namespace fs = std::filesystem;

    // Detect quantization parameters from config.json
    {
        std::string config_file = path + "/config.json";
        std::ifstream cfg_stream(config_file);
        if (cfg_stream.is_open()) {
            nlohmann::json cfg;
            cfg_stream >> cfg;
            if (cfg.contains("quantization")) {
                auto& qc = cfg["quantization"];
                if (qc.contains("group_size")) group_size_ = qc["group_size"].get<int>();
                if (qc.contains("bits")) bits_ = qc["bits"].get<int>();
            }
        }
    }

    // Collect safetensor files to load
    std::vector<std::string> safetensor_files;

    // Prefer index.json for sharded models
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

    // Detect quantization from first layer's q_proj
    if (all_weights.count("language_model.model.layers.0.self_attn.q_proj.scales")) {
        quantized_ = true;
    }

    // --- Pass vision_tower weights to VisionEncoder ---
    std::unordered_map<std::string, array> vision_weights;
    for (auto& [name, arr] : all_weights) {
        if (name.find("vision_tower.") == 0 || name.find("visual.") == 0) {
            vision_weights.insert_or_assign(name, arr);
        }
    }
    vision_encoder_->load_weights(vision_weights);

    // --- Load embedding ---
    // Dequantize if quantized (needed for take() lookup)
    auto emb_it = all_weights.find("language_model.model.embed_tokens.weight");
    if (emb_it == all_weights.end()) {
        throw std::runtime_error("Missing language_model.model.embed_tokens.weight");
    }
    auto emb_scales_it = all_weights.find("language_model.model.embed_tokens.scales");
    if (emb_scales_it != all_weights.end()) {
        auto emb_biases_it = all_weights.find("language_model.model.embed_tokens.biases");
        std::optional<array> emb_biases;
        if (emb_biases_it != all_weights.end()) {
            emb_biases = emb_biases_it->second;
        }
        auto bias_arr = emb_biases.has_value() ? *emb_biases
            : zeros(emb_scales_it->second.shape(), emb_scales_it->second.dtype());
        embed_tokens_q_ = std::make_unique<QuantizedLinear>(
            emb_it->second, emb_scales_it->second, bias_arr, group_size_, bits_);
        auto dequantized = dequantize(
            emb_it->second, emb_scales_it->second, emb_biases,
            group_size_, bits_);
        eval(dequantized);
        embed_tokens_ = dequantized;
    } else {
        embed_tokens_ = emb_it->second;
    }

    // --- Load decoder layers ---
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string prefix = "language_model.model.layers." + std::to_string(i);
        auto& layer = layers_[i];

        // Norms
        auto in_it = all_weights.find(prefix + ".input_layernorm.weight");
        if (in_it == all_weights.end()) {
            throw std::runtime_error("Missing " + prefix + ".input_layernorm.weight");
        }
        layer.input_layernorm_weight = in_it->second;

        auto pn_it = all_weights.find(prefix + ".post_attention_layernorm.weight");
        if (pn_it == all_weights.end()) {
            throw std::runtime_error("Missing " + prefix + ".post_attention_layernorm.weight");
        }
        layer.post_attention_layernorm_weight = pn_it->second;

        // Self-attention projections
        load_linear(prefix + ".self_attn.q_proj", all_weights,
                    layer.q_proj_weight, layer.q_proj_q, layer.is_quantized);
        load_linear(prefix + ".self_attn.k_proj", all_weights,
                    layer.k_proj_weight, layer.k_proj_q, layer.is_quantized);
        load_linear(prefix + ".self_attn.v_proj", all_weights,
                    layer.v_proj_weight, layer.v_proj_q, layer.is_quantized);
        load_linear(prefix + ".self_attn.o_proj", all_weights,
                    layer.o_proj_weight, layer.o_proj_q, layer.is_quantized);

        // Q/K/V linear biases (separate from quantization .biases)
        auto qb_it = all_weights.find(prefix + ".self_attn.q_proj.bias");
        if (qb_it != all_weights.end()) {
            layer.q_proj_bias = qb_it->second;
        }
        auto kb_it = all_weights.find(prefix + ".self_attn.k_proj.bias");
        if (kb_it != all_weights.end()) {
            layer.k_proj_bias = kb_it->second;
        }
        auto vb_it = all_weights.find(prefix + ".self_attn.v_proj.bias");
        if (vb_it != all_weights.end()) {
            layer.v_proj_bias = vb_it->second;
        }
        // o_proj has NO linear bias

        // MLP projections (no linear bias)
        load_linear(prefix + ".mlp.gate_proj", all_weights,
                    layer.gate_proj_weight, layer.gate_proj_q, layer.is_quantized);
        load_linear(prefix + ".mlp.up_proj", all_weights,
                    layer.up_proj_weight, layer.up_proj_q, layer.is_quantized);
        load_linear(prefix + ".mlp.down_proj", all_weights,
                    layer.down_proj_weight, layer.down_proj_q, layer.is_quantized);
    }

    // --- Final norm ---
    auto fn_it = all_weights.find("language_model.model.norm.weight");
    if (fn_it == all_weights.end()) {
        throw std::runtime_error("Missing language_model.model.norm.weight");
    }
    final_norm_weight_ = fn_it->second;

    // --- LM head ---
    bool lm_head_quantized = false;
    load_linear("language_model.lm_head", all_weights,
                lm_head_weight_, lm_head_q_, lm_head_quantized);
    if (lm_head_quantized || all_weights.count("language_model.lm_head.weight")) {
        has_lm_head_ = true;
    } else if (config_.tie_word_embeddings) {
        if (embed_tokens_q_) {
            lm_head_q_ = std::make_unique<QuantizedLinear>(
                embed_tokens_q_->weight(), embed_tokens_q_->scales(),
                embed_tokens_q_->biases(), group_size_, bits_);
        } else {
            lm_head_weight_ = *embed_tokens_;
        }
        has_lm_head_ = true;
    }
}

void Qwen2_5VLModel::reset_cache() {
    kv_cache_->reset();
    pending_image_ = std::nullopt;
    mrope_next_pos_ = 0;
    has_done_vision_prefill_ = false;
}

} // namespace gomlx
