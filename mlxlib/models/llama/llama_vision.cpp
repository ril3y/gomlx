#include "models/llama/llama_vision.h"

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

LlamaVisionModel::LlamaVisionModel(const ModelConfig& config)
    : config_(config) {
    head_dim_ = config_.get_head_dim();
    num_kv_groups_ = config_.num_attention_heads / config_.num_key_value_heads;
    layers_.resize(config_.num_hidden_layers);

    // Mark cross-attention layers
    for (int idx : config_.text_config.cross_attention_layers) {
        cross_attn_layer_set_.insert(idx);
        if (idx < config_.num_hidden_layers) {
            layers_[idx].has_cross_attention = true;
        }
    }

    kv_cache_ = std::make_unique<KVCache>(config_.num_hidden_layers);
    vision_encoder_ = std::make_unique<VisionEncoder>(config_.vision_config);
    image_processor_ = std::make_unique<ImageProcessor>(config_.vision_config);
}

void LlamaVisionModel::set_image(const ImageProcessorResult& result) {
    pending_pixel_values_ = result.pixel_values;
    pending_aspect_ratio_id_ = result.aspect_ratio_id;
    pending_num_tiles_ = result.num_tiles;
    cross_attention_states_ = std::nullopt;
    cross_kv_cache_.clear();
}

void LlamaVisionModel::load_linear(
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

array LlamaVisionModel::linear_forward(
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

array LlamaVisionModel::self_attention(int layer_idx, const array& x, int offset) {
    auto& layer = layers_[layer_idx];

    int batch = x.shape(0);
    int seq_len = x.shape(1);

    // Project Q, K, V
    std::string ln = "layer" + std::to_string(layer_idx);
    array q = linear_forward(x, layer.q_proj_weight, layer.q_proj_q.get(), ln + ".sa.q_proj");
    array k = linear_forward(x, layer.k_proj_weight, layer.k_proj_q.get(), ln + ".sa.k_proj");
    array v = linear_forward(x, layer.v_proj_weight, layer.v_proj_q.get(), ln + ".sa.v_proj");

    // Reshape to [batch, seq_len, num_heads, head_dim] then transpose
    q = reshape(q, {batch, seq_len, config_.num_attention_heads, head_dim_});
    q = transpose(q, {0, 2, 1, 3});

    k = reshape(k, {batch, seq_len, config_.num_key_value_heads, head_dim_});
    k = transpose(k, {0, 2, 1, 3});

    v = reshape(v, {batch, seq_len, config_.num_key_value_heads, head_dim_});
    v = transpose(v, {0, 2, 1, 3});

    // Apply RoPE with offset = current cache length
    int rope_offset = kv_cache_->sequence_length();
    q = fast::rope(q, head_dim_, false, config_.rope_theta, 1.0f, rope_offset);
    k = fast::rope(k, head_dim_, false, config_.rope_theta, 1.0f, rope_offset);

    // Update KV cache
    auto [cached_k, cached_v] = kv_cache_->update(layer_idx, k, v);

    // Scaled dot product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    array attn_out = fast::scaled_dot_product_attention(
        q, cached_k, cached_v, scale, "causal");

    // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch, seq_len, config_.num_attention_heads * head_dim_});

    // Output projection
    return linear_forward(attn_out, layer.o_proj_weight, layer.o_proj_q.get(), ln + ".sa.o_proj");
}

array LlamaVisionModel::cross_attention(int layer_idx, const array& x) {
    auto& layer = layers_[layer_idx];
    int batch = x.shape(0);
    int seq_len = x.shape(1);
    std::string ln = "layer" + std::to_string(layer_idx);

    // Q from text hidden states
    array q = linear_forward(x, layer.cross_q_proj_weight, layer.cross_q_proj_q.get(), ln + ".xa.q_proj");
    q = reshape(q, {batch, seq_len, config_.num_attention_heads, head_dim_});
    q = transpose(q, {0, 2, 1, 3});

    // Per-head RMSNorm on Q
    q = fast::rms_norm(q, *layer.cross_q_norm_weight, config_.rms_norm_eps);

    if (cross_attention_states_.has_value()) {
        // First call with vision: compute K/V from vision features and cache
        array k = linear_forward(*cross_attention_states_,
                                 layer.cross_k_proj_weight, layer.cross_k_proj_q.get(), ln + ".xa.k_proj");
        array v = linear_forward(*cross_attention_states_,
                                 layer.cross_v_proj_weight, layer.cross_v_proj_q.get(), ln + ".xa.v_proj");

        int vision_seq = cross_attention_states_->shape(1);
        k = reshape(k, {batch, vision_seq, config_.num_key_value_heads, head_dim_});
        k = transpose(k, {0, 2, 1, 3});
        v = reshape(v, {batch, vision_seq, config_.num_key_value_heads, head_dim_});
        v = transpose(v, {0, 2, 1, 3});

        // Per-head RMSNorm on K
        k = fast::rms_norm(k, *layer.cross_k_norm_weight, config_.rms_norm_eps);

        // Cache for subsequent tokens
        cross_kv_cache_[layer_idx] = {k, v};
    } else if (cross_kv_cache_.find(layer_idx) == cross_kv_cache_.end()) {
        // No vision states, no cache: return zeros (layer effectively skipped via gate)
        return zeros({batch, seq_len, config_.num_attention_heads * head_dim_}, float32);
    }

    auto& cached = cross_kv_cache_[layer_idx];

    // NO RoPE for cross-attention
    // NO causal mask -- all text tokens attend to all vision tokens
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    array attn_out = fast::scaled_dot_product_attention(
        q, *cached.keys, *cached.values, scale, "");

    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch, seq_len, config_.num_attention_heads * head_dim_});

    return linear_forward(attn_out, layer.cross_o_proj_weight, layer.cross_o_proj_q.get(), ln + ".xa.o_proj");
}

array LlamaVisionModel::mlp(int layer_idx, const array& x) {
    auto& layer = layers_[layer_idx];
    std::string ln = "layer" + std::to_string(layer_idx);

    // SwiGLU: silu(gate_proj(x)) * up_proj(x)
    array gate = linear_forward(x, layer.gate_proj_weight, layer.gate_proj_q.get(), ln + ".mlp.gate");
    array up = linear_forward(x, layer.up_proj_weight, layer.up_proj_q.get(), ln + ".mlp.up");
    array activated = multiply(silu(gate), up);

    // Down projection
    return linear_forward(activated, layer.down_proj_weight, layer.down_proj_q.get(), ln + ".mlp.down");
}

array LlamaVisionModel::forward(const array& tokens, int offset) {
    // 1. Process pending vision if set
    if (pending_pixel_values_.has_value()) {
        auto vision_out = vision_encoder_->forward(
            *pending_pixel_values_, pending_aspect_ratio_id_, pending_num_tiles_);
        // Project: 7680 -> hidden_size (4096)
        auto projected = linear_forward(vision_out, projector_weight_, projector_q_.get(), "multi_modal_projector");
        if (projector_bias_.has_value()) {
            projected = add(projected, *projector_bias_);
        }
        cross_attention_states_ = projected;
        pending_pixel_values_ = std::nullopt;
    }

    // 2. Embed tokens
    array input_tokens = tokens;
    if (input_tokens.ndim() == 1) {
        input_tokens = reshape(input_tokens, {1, static_cast<int>(input_tokens.shape(0))});
    }

    array flat_tokens = reshape(input_tokens, {-1});
    array h = take(*embed_tokens_, flat_tokens, 0);
    h = reshape(h, {input_tokens.shape(0), input_tokens.shape(1), config_.hidden_size});

    // 3. Run through decoder layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        if (cross_attn_layer_set_.count(i)) {
            // Cross-attention decoder layer: replaces self-attention entirely
            // Pre-norm -> cross-attention -> gated residual
            array normed = fast::rms_norm(h, *layers_[i].input_layernorm_weight, config_.rms_norm_eps);
            array cross_out = cross_attention(i, normed);
            h = add(h, multiply(tanh(*layers_[i].cross_attn_attn_gate), cross_out));

            // Post-norm -> MLP -> gated residual
            normed = fast::rms_norm(h, *layers_[i].post_attention_layernorm_weight, config_.rms_norm_eps);
            array mlp_out = mlp(i, normed);
            h = add(h, multiply(tanh(*layers_[i].cross_attn_mlp_gate), mlp_out));
        } else {
            // Standard self-attention decoder layer
            array normed = fast::rms_norm(h, *layers_[i].input_layernorm_weight, config_.rms_norm_eps);
            array attn_out = self_attention(i, normed, offset);
            h = add(h, attn_out);

            normed = fast::rms_norm(h, *layers_[i].post_attention_layernorm_weight, config_.rms_norm_eps);
            array mlp_out = mlp(i, normed);
            h = add(h, mlp_out);
        }
    }

    // Clear cross_attention_states_ after first pass so subsequent tokens use cache
    if (cross_attention_states_.has_value()) {
        cross_attention_states_ = std::nullopt;
    }

    // 4. Final norm + LM head
    h = fast::rms_norm(h, *final_norm_weight_, config_.rms_norm_eps);
    if (has_lm_head_) {
        h = linear_forward(h, lm_head_weight_, lm_head_q_.get(), "lm_head");
    } else {
        h = matmul(h, transpose(*embed_tokens_));
    }

    return h;
}

void LlamaVisionModel::load_weights(const std::string& path) {
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
        if (name.find("vision_tower.") == 0 || name.find("vision_model.") == 0) {
            vision_weights.insert_or_assign(name, arr);
        }
    }
    vision_encoder_->load_weights(vision_weights, group_size_, bits_);

    // --- Load multi-modal projector ---
    bool proj_quantized = false;
    load_linear("multi_modal_projector", all_weights,
                projector_weight_, projector_q_, proj_quantized);
    // Also check for linear bias (separate from quantization biases)
    auto proj_bias_it = all_weights.find("multi_modal_projector.bias");
    if (proj_bias_it != all_weights.end()) {
        projector_bias_ = proj_bias_it->second;
    }

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

        // MLP projections (present in both layer types)
        load_linear(prefix + ".mlp.gate_proj", all_weights,
                    layer.gate_proj_weight, layer.gate_proj_q, layer.is_quantized);
        load_linear(prefix + ".mlp.up_proj", all_weights,
                    layer.up_proj_weight, layer.up_proj_q, layer.is_quantized);
        load_linear(prefix + ".mlp.down_proj", all_weights,
                    layer.down_proj_weight, layer.down_proj_q, layer.is_quantized);

        if (layer.has_cross_attention) {
            // Cross-attention layers: use cross_attn.* (NO self_attn)
            bool cross_quantized = false;
            load_linear(prefix + ".cross_attn.q_proj", all_weights,
                        layer.cross_q_proj_weight, layer.cross_q_proj_q, cross_quantized);
            load_linear(prefix + ".cross_attn.k_proj", all_weights,
                        layer.cross_k_proj_weight, layer.cross_k_proj_q, cross_quantized);
            load_linear(prefix + ".cross_attn.v_proj", all_weights,
                        layer.cross_v_proj_weight, layer.cross_v_proj_q, cross_quantized);
            load_linear(prefix + ".cross_attn.o_proj", all_weights,
                        layer.cross_o_proj_weight, layer.cross_o_proj_q, cross_quantized);

            // Cross-attention norms
            auto qn_it = all_weights.find(prefix + ".cross_attn.q_norm.weight");
            if (qn_it != all_weights.end()) {
                layer.cross_q_norm_weight = qn_it->second;
            }
            auto kn_it = all_weights.find(prefix + ".cross_attn.k_norm.weight");
            if (kn_it != all_weights.end()) {
                layer.cross_k_norm_weight = kn_it->second;
            }

            // Cross-attention gates
            auto ag_it = all_weights.find(prefix + ".cross_attn_attn_gate");
            if (ag_it != all_weights.end()) {
                layer.cross_attn_attn_gate = ag_it->second;
            }
            auto mg_it = all_weights.find(prefix + ".cross_attn_mlp_gate");
            if (mg_it != all_weights.end()) {
                layer.cross_attn_mlp_gate = mg_it->second;
            }
        } else {
            // Self-attention layers: use self_attn.*
            load_linear(prefix + ".self_attn.q_proj", all_weights,
                        layer.q_proj_weight, layer.q_proj_q, layer.is_quantized);
            load_linear(prefix + ".self_attn.k_proj", all_weights,
                        layer.k_proj_weight, layer.k_proj_q, layer.is_quantized);
            load_linear(prefix + ".self_attn.v_proj", all_weights,
                        layer.v_proj_weight, layer.v_proj_q, layer.is_quantized);
            load_linear(prefix + ".self_attn.o_proj", all_weights,
                        layer.o_proj_weight, layer.o_proj_q, layer.is_quantized);
        }
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

void LlamaVisionModel::reset_cache() {
    kv_cache_->reset();
    cross_kv_cache_.clear();
    cross_attention_states_ = std::nullopt;
    pending_pixel_values_ = std::nullopt;
}

} // namespace gomlx
