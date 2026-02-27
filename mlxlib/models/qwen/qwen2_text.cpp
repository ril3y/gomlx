#include "models/qwen/qwen2_text.h"

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

Qwen2TextModel::Qwen2TextModel(const ModelConfig& config)
    : config_(config) {
    head_dim_ = config_.get_head_dim();
    num_kv_groups_ = config_.num_attention_heads / config_.num_key_value_heads;
    layers_.resize(config_.num_hidden_layers);
    kv_cache_ = std::make_unique<KVCache>(config_.num_hidden_layers);
}

void Qwen2TextModel::load_linear(
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

array Qwen2TextModel::linear_forward(
    const array& input,
    const std::optional<array>& weight,
    const QuantizedLinear* q_linear) const {
    if (q_linear) {
        return q_linear->forward(input);
    }
    if (!weight.has_value()) {
        throw std::runtime_error("linear_forward: no weight or quantized layer available");
    }
    return matmul(input, transpose(*weight));
}

// --- Attention with Q/K/V bias and standard RoPE ---

array Qwen2TextModel::attention(int layer_idx, const array& x, int offset) {
    auto& layer = layers_[layer_idx];

    int batch = x.shape(0);
    int seq_len = x.shape(1);

    // Q, K, V projections
    array q = linear_forward(x, layer.q_proj_weight, layer.q_proj_q.get());
    if (layer.q_proj_bias.has_value()) q = add(q, *layer.q_proj_bias);

    array k = linear_forward(x, layer.k_proj_weight, layer.k_proj_q.get());
    if (layer.k_proj_bias.has_value()) k = add(k, *layer.k_proj_bias);

    array v = linear_forward(x, layer.v_proj_weight, layer.v_proj_q.get());
    if (layer.v_proj_bias.has_value()) v = add(v, *layer.v_proj_bias);

    // Reshape: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    q = reshape(q, {batch, seq_len, config_.num_attention_heads, head_dim_});
    q = transpose(q, {0, 2, 1, 3});

    k = reshape(k, {batch, seq_len, config_.num_key_value_heads, head_dim_});
    k = transpose(k, {0, 2, 1, 3});

    v = reshape(v, {batch, seq_len, config_.num_key_value_heads, head_dim_});
    v = transpose(v, {0, 2, 1, 3});

    // Standard RoPE (same as Llama)
    int rope_offset = kv_cache_->sequence_length();
    q = fast::rope(q, head_dim_, false, config_.rope_theta, 1.0f, rope_offset);
    k = fast::rope(k, head_dim_, false, config_.rope_theta, 1.0f, rope_offset);

    // Update KV cache
    auto [cached_k, cached_v] = kv_cache_->update(layer_idx, k, v);

    // Scaled dot product attention with causal mask
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    array attn_out = fast::scaled_dot_product_attention(
        q, cached_k, cached_v, scale, "causal");

    // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch, seq_len, config_.num_attention_heads * head_dim_});

    // Output projection (NO bias for o_proj)
    return linear_forward(attn_out, layer.o_proj_weight, layer.o_proj_q.get());
}

// --- MLP (SwiGLU, same as Llama) ---

array Qwen2TextModel::mlp(int layer_idx, const array& x) {
    auto& layer = layers_[layer_idx];

    array gate = linear_forward(x, layer.gate_proj_weight, layer.gate_proj_q.get());
    array up = linear_forward(x, layer.up_proj_weight, layer.up_proj_q.get());
    array activated = multiply(silu(gate), up);

    return linear_forward(activated, layer.down_proj_weight, layer.down_proj_q.get());
}

// --- Forward pass ---

array Qwen2TextModel::forward(const array& tokens, int offset) {
    array input_tokens = tokens;
    if (input_tokens.ndim() == 1) {
        input_tokens = reshape(input_tokens, {1, static_cast<int>(input_tokens.shape(0))});
    }

    // Embedding lookup
    array flat_tokens = reshape(input_tokens, {-1});
    array h = take(*embed_tokens_, flat_tokens, 0);
    h = reshape(h, {input_tokens.shape(0), input_tokens.shape(1), config_.hidden_size});

    // Decoder layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        // Pre-attention norm
        array normed = fast::rms_norm(h, *layers_[i].input_layernorm_weight, config_.rms_norm_eps);
        array attn_out = attention(i, normed, offset);
        h = add(h, attn_out);

        // Post-attention norm + MLP
        normed = fast::rms_norm(h, *layers_[i].post_attention_layernorm_weight, config_.rms_norm_eps);
        array mlp_out = mlp(i, normed);
        h = add(h, mlp_out);
    }

    // Final norm
    h = fast::rms_norm(h, *final_norm_weight_, config_.rms_norm_eps);

    // LM head
    if (has_lm_head_) {
        h = linear_forward(h, lm_head_weight_, lm_head_q_.get());
    } else {
        h = matmul(h, transpose(*embed_tokens_));
    }

    return h;
}

// --- Weight loading ---

void Qwen2TextModel::load_weights(const std::string& path) {
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

    // Load all weights
    std::unordered_map<std::string, array> all_weights;
    for (const auto& file : safetensor_files) {
        auto [weights, metadata] = load_safetensors(file);
        for (auto& [name, arr] : weights) {
            all_weights.insert_or_assign(name, std::move(arr));
        }
    }

    // Detect quantization from first layer
    if (all_weights.count("model.layers.0.self_attn.q_proj.scales")) {
        quantized_ = true;
    }

    // Load embedding (dequantize if quantized for take() lookup)
    auto emb_it = all_weights.find("model.embed_tokens.weight");
    if (emb_it == all_weights.end()) {
        throw std::runtime_error("Missing model.embed_tokens.weight");
    }
    auto emb_scales_it = all_weights.find("model.embed_tokens.scales");
    if (emb_scales_it != all_weights.end()) {
        auto emb_biases_it = all_weights.find("model.embed_tokens.biases");
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

    // Load decoder layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string prefix = "model.layers." + std::to_string(i);
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

        // Attention projections
        load_linear(prefix + ".self_attn.q_proj", all_weights,
                    layer.q_proj_weight, layer.q_proj_q, layer.is_quantized);
        load_linear(prefix + ".self_attn.k_proj", all_weights,
                    layer.k_proj_weight, layer.k_proj_q, layer.is_quantized);
        load_linear(prefix + ".self_attn.v_proj", all_weights,
                    layer.v_proj_weight, layer.v_proj_q, layer.is_quantized);
        load_linear(prefix + ".self_attn.o_proj", all_weights,
                    layer.o_proj_weight, layer.o_proj_q, layer.is_quantized);

        // Q/K/V linear biases
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

        // MLP projections
        load_linear(prefix + ".mlp.gate_proj", all_weights,
                    layer.gate_proj_weight, layer.gate_proj_q, layer.is_quantized);
        load_linear(prefix + ".mlp.up_proj", all_weights,
                    layer.up_proj_weight, layer.up_proj_q, layer.is_quantized);
        load_linear(prefix + ".mlp.down_proj", all_weights,
                    layer.down_proj_weight, layer.down_proj_q, layer.is_quantized);
    }

    // Final norm
    auto fn_it = all_weights.find("model.norm.weight");
    if (fn_it == all_weights.end()) {
        throw std::runtime_error("Missing model.norm.weight");
    }
    final_norm_weight_ = fn_it->second;

    // LM head
    bool lm_head_quantized = false;
    load_linear("lm_head", all_weights,
                lm_head_weight_, lm_head_q_, lm_head_quantized);
    if (lm_head_quantized || all_weights.count("lm_head.weight")) {
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

void Qwen2TextModel::reset_cache() {
    kv_cache_->reset();
}

} // namespace gomlx
