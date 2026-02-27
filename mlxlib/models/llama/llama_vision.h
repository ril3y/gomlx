#pragma once

#include "base_model.h"
#include "config.h"
#include "kv_cache.h"
#include "quantized_linear.h"
#include "image_processor.h"
#include "models/llama/vision_encoder.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>

namespace gomlx {

struct MllamaDecoderLayer {
    // Self-attention (same as LlamaLayer)
    std::optional<mlx::core::array> input_layernorm_weight;
    std::optional<mlx::core::array> post_attention_layernorm_weight;

    // Self-attention projections (quantized or dense)
    std::optional<mlx::core::array> q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight;
    std::unique_ptr<QuantizedLinear> q_proj_q, k_proj_q, v_proj_q, o_proj_q;

    // MLP projections (quantized or dense)
    std::optional<mlx::core::array> gate_proj_weight, up_proj_weight, down_proj_weight;
    std::unique_ptr<QuantizedLinear> gate_proj_q, up_proj_q, down_proj_q;

    bool is_quantized = false;

    // Cross-attention (only for layers at positions {3,8,13,18,23,28,33,38})
    bool has_cross_attention = false;
    std::optional<mlx::core::array> cross_q_proj_weight, cross_k_proj_weight;
    std::optional<mlx::core::array> cross_v_proj_weight, cross_o_proj_weight;
    std::unique_ptr<QuantizedLinear> cross_q_proj_q, cross_k_proj_q;
    std::unique_ptr<QuantizedLinear> cross_v_proj_q, cross_o_proj_q;
    std::optional<mlx::core::array> cross_q_norm_weight;  // RMSNorm on head_dim=128
    std::optional<mlx::core::array> cross_k_norm_weight;  // RMSNorm on head_dim=128
    std::optional<mlx::core::array> cross_attn_attn_gate;  // scalar [1]
    std::optional<mlx::core::array> cross_attn_mlp_gate;   // scalar [1]
};

class LlamaVisionModel : public BaseModel {
public:
    explicit LlamaVisionModel(const ModelConfig& config);

    mlx::core::array forward(const mlx::core::array& tokens, int offset = 0) override;
    void load_weights(const std::string& path) override;
    int vocab_size() const override { return config_.vocab_size; }
    bool supports_vision() const override { return true; }
    void reset_cache() override;

    // Vision API
    void set_image(const ImageProcessorResult& result);
    ImageProcessor& image_processor() { return *image_processor_; }

    void set_image_from_file(const std::string& path) override {
        auto result = image_processor().process_from_file(path);
        set_image(result);
    }

    void set_image_from_bytes(const uint8_t* data, int len) override {
        auto result = image_processor().process_from_bytes(data, len);
        set_image(result);
    }

    std::vector<int> stop_token_ids() const override { return {128001, 128009}; }

private:
    ModelConfig config_;
    int head_dim_;
    int num_kv_groups_;
    std::set<int> cross_attn_layer_set_;

    // Embedding
    std::optional<mlx::core::array> embed_tokens_;
    std::unique_ptr<QuantizedLinear> embed_tokens_q_;

    // Decoder layers
    std::vector<MllamaDecoderLayer> layers_;

    // Final norm + LM head
    std::optional<mlx::core::array> final_norm_weight_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::unique_ptr<QuantizedLinear> lm_head_q_;
    bool has_lm_head_ = false;

    // Self-attention KV cache
    std::unique_ptr<KVCache> kv_cache_;

    // Cross-attention KV cache (computed once from vision, reused)
    struct CrossKV {
        std::optional<mlx::core::array> keys, values;
    };
    std::unordered_map<int, CrossKV> cross_kv_cache_;

    // Vision components
    std::unique_ptr<VisionEncoder> vision_encoder_;
    std::optional<mlx::core::array> projector_weight_, projector_bias_;
    std::unique_ptr<QuantizedLinear> projector_q_;
    std::unique_ptr<ImageProcessor> image_processor_;

    // Pending image data
    std::optional<mlx::core::array> pending_pixel_values_;
    int pending_aspect_ratio_id_ = 0;
    int pending_num_tiles_ = 0;
    std::optional<mlx::core::array> cross_attention_states_;

    bool quantized_ = false;
    int group_size_ = 64;
    int bits_ = 4;

    // Helpers
    mlx::core::array self_attention(int layer_idx, const mlx::core::array& x, int offset);
    mlx::core::array cross_attention(int layer_idx, const mlx::core::array& x);
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
};

} // namespace gomlx
