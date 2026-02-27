#pragma once

#include "base_model.h"
#include "config.h"
#include "kv_cache.h"
#include "quantized_linear.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

namespace gomlx {

// A single Llama transformer layer.
struct LlamaLayer {
    // Attention norms
    std::optional<mlx::core::array> input_layernorm_weight;
    std::optional<mlx::core::array> post_attention_layernorm_weight;

    // Attention projections (may be quantized or dense)
    std::unique_ptr<QuantizedLinear> q_proj_q;
    std::unique_ptr<QuantizedLinear> k_proj_q;
    std::unique_ptr<QuantizedLinear> v_proj_q;
    std::unique_ptr<QuantizedLinear> o_proj_q;

    // Dense weights (used when not quantized)
    std::optional<mlx::core::array> q_proj_weight;
    std::optional<mlx::core::array> k_proj_weight;
    std::optional<mlx::core::array> v_proj_weight;
    std::optional<mlx::core::array> o_proj_weight;

    // MLP projections (may be quantized or dense)
    std::unique_ptr<QuantizedLinear> gate_proj_q;
    std::unique_ptr<QuantizedLinear> up_proj_q;
    std::unique_ptr<QuantizedLinear> down_proj_q;

    std::optional<mlx::core::array> gate_proj_weight;
    std::optional<mlx::core::array> up_proj_weight;
    std::optional<mlx::core::array> down_proj_weight;

    bool is_quantized = false;
};

class LlamaTextModel : public BaseModel {
public:
    explicit LlamaTextModel(const ModelConfig& config);

    mlx::core::array forward(const mlx::core::array& tokens, int offset = 0) override;
    void load_weights(const std::string& path) override;
    int vocab_size() const override { return config_.vocab_size; }
    void reset_cache() override;

    std::vector<int> stop_token_ids() const override { return {128001, 128009}; }

private:
    // Run a single transformer layer
    mlx::core::array run_layer(
        int layer_idx,
        const mlx::core::array& x,
        int offset);

    // Attention sub-layer
    mlx::core::array attention(
        int layer_idx,
        const mlx::core::array& x,
        int offset);

    // MLP sub-layer
    mlx::core::array mlp(int layer_idx, const mlx::core::array& x);

    // Linear projection (quantized or dense)
    mlx::core::array linear_forward(
        const mlx::core::array& input,
        const std::optional<mlx::core::array>& weight,
        const QuantizedLinear* q_linear) const;

    // Load a potentially quantized linear layer from weight map
    void load_linear(
        const std::string& prefix,
        const std::unordered_map<std::string, mlx::core::array>& weights,
        std::optional<mlx::core::array>& dense_weight,
        std::unique_ptr<QuantizedLinear>& q_linear,
        bool& is_quantized);

    ModelConfig config_;
    int head_dim_;
    int num_kv_groups_; // num_attention_heads / num_key_value_heads

    // Embedding (dequantized for take() lookup)
    std::optional<mlx::core::array> embed_tokens_;
    // Quantized embedding for tied lm_head (if embedding was quantized)
    std::unique_ptr<QuantizedLinear> embed_tokens_q_;

    // Transformer layers
    std::vector<LlamaLayer> layers_;

    // Final norm
    std::optional<mlx::core::array> final_norm_weight_;

    // LM head (may be tied to embeddings, or quantized)
    std::optional<mlx::core::array> lm_head_weight_;
    std::unique_ptr<QuantizedLinear> lm_head_q_;
    bool has_lm_head_ = false;

    // KV cache
    std::unique_ptr<KVCache> kv_cache_;

    // Track if model is quantized
    bool quantized_ = false;
    int group_size_ = 64;
    int bits_ = 4;
};

} // namespace gomlx
