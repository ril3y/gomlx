#pragma once

#include "base_model.h"
#include "config.h"
#include "kv_cache.h"
#include "quantized_linear.h"
#include "qwen_image_processor.h"
#include "models/qwen/qwen_vision_encoder.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

namespace gomlx {

struct Qwen2DecoderLayer {
    std::optional<mlx::core::array> input_layernorm_weight;
    std::optional<mlx::core::array> post_attention_layernorm_weight;

    // Q/K/V: quantized + linear bias; O: quantized, no linear bias
    std::optional<mlx::core::array> q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight;
    std::optional<mlx::core::array> q_proj_bias, k_proj_bias, v_proj_bias;
    std::unique_ptr<QuantizedLinear> q_proj_q, k_proj_q, v_proj_q, o_proj_q;

    // SwiGLU MLP (quantized, no linear bias)
    std::optional<mlx::core::array> gate_proj_weight, up_proj_weight, down_proj_weight;
    std::unique_ptr<QuantizedLinear> gate_proj_q, up_proj_q, down_proj_q;

    bool is_quantized = false;
};

class Qwen2_5VLModel : public BaseModel {
public:
    explicit Qwen2_5VLModel(const ModelConfig& config);

    mlx::core::array forward(const mlx::core::array& tokens, int offset = 0) override;
    void load_weights(const std::string& path) override;
    int vocab_size() const override { return config_.vocab_size; }
    bool supports_vision() const override { return true; }
    void reset_cache() override;

    // Vision API
    void set_image(const QwenImageProcessorResult& result);
    QwenImageProcessor& image_processor() { return *image_processor_; }
    int pending_vision_tokens() const;

    void set_image_from_file(const std::string& path) override {
        auto result = image_processor().process_from_file(path);
        set_image(result);
    }

    void set_image_from_bytes(const uint8_t* data, int len) override {
        auto result = image_processor().process_from_bytes(data, len);
        set_image(result);
    }

    int pending_vision_token_count() const override { return pending_vision_tokens(); }

    std::vector<int> stop_token_ids() const override { return {151643, 151645}; }

private:
    ModelConfig config_;
    int head_dim_;
    int num_kv_groups_;

    // Embedding
    std::optional<mlx::core::array> embed_tokens_;
    std::unique_ptr<QuantizedLinear> embed_tokens_q_;

    // 28 decoder layers
    std::vector<Qwen2DecoderLayer> layers_;

    // Final norm + LM head
    std::optional<mlx::core::array> final_norm_weight_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::unique_ptr<QuantizedLinear> lm_head_q_;
    bool has_lm_head_ = false;

    // KV cache
    std::unique_ptr<KVCache> kv_cache_;

    // Vision components
    std::unique_ptr<QwenVisionEncoder> vision_encoder_;
    std::unique_ptr<QwenImageProcessor> image_processor_;

    // Pending image data
    std::optional<QwenImageProcessorResult> pending_image_;

    // mRoPE tracking: after vision prefill, track the max position for autoregressive
    int mrope_next_pos_ = 0;
    bool has_done_vision_prefill_ = false;

    bool quantized_ = false;
    int group_size_ = 64;
    int bits_ = 4;

    // Helpers
    mlx::core::array self_attention(int layer_idx, const mlx::core::array& x,
                                     const mlx::core::array& position_ids);
    mlx::core::array mlp(int layer_idx, const mlx::core::array& x);
    mlx::core::array linear_forward(const mlx::core::array& input,
                                     const std::optional<mlx::core::array>& weight,
                                     const QuantizedLinear* q_linear,
                                     const std::string& debug_name = "") const;
    void load_linear(const std::string& prefix,
                     const std::unordered_map<std::string, mlx::core::array>& weights,
                     std::optional<mlx::core::array>& dense_weight,
                     std::unique_ptr<QuantizedLinear>& q_linear,
                     bool& is_quantized);

    // mRoPE helpers
    mlx::core::array compute_mrope_positions(int seq_len, int grid_h, int grid_w,
                                              const mlx::core::array& tokens);
    mlx::core::array apply_mrope(const mlx::core::array& x, const mlx::core::array& positions,
                                  const std::vector<int>& sections);
};

} // namespace gomlx
