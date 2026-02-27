#pragma once

#include "base_model.h"
#include "config.h"
#include "kv_cache.h"
#include "models/moondream/moondream_image_processor.h"
#include "models/moondream/moondream_vision_encoder.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace gomlx {

struct MoondreamDecoderLayer {
    // Single LayerNorm per block (Phi-style, weight + bias)
    std::optional<mlx::core::array> ln_weight, ln_bias;

    // Attention: Q/K/V split from fused Wqkv at load time + biases
    std::optional<mlx::core::array> q_proj_weight, k_proj_weight, v_proj_weight;
    std::optional<mlx::core::array> q_proj_bias, k_proj_bias, v_proj_bias;
    std::optional<mlx::core::array> out_proj_weight, out_proj_bias;

    // MLP: fc1/fc2 with bias (simple 2-layer MLP with GELU, NOT SwiGLU)
    std::optional<mlx::core::array> fc1_weight, fc1_bias;
    std::optional<mlx::core::array> fc2_weight, fc2_bias;
};

class MoondreamModel : public BaseModel {
public:
    explicit MoondreamModel(const ModelConfig& config);

    mlx::core::array forward(const mlx::core::array& tokens, int offset = 0) override;
    void load_weights(const std::string& path) override;
    int vocab_size() const override { return config_.moondream_config.text_vocab_size; }
    bool supports_vision() const override { return true; }
    void reset_cache() override;

    // Vision API
    void set_image_from_file(const std::string& path) override {
        pending_image_ = image_processor_->process_from_file(path);
    }

    void set_image_from_bytes(const uint8_t* data, int len) override {
        pending_image_ = image_processor_->process_from_bytes(data, len);
    }

    int pending_vision_token_count() const override { return 0; }

    std::vector<int> stop_token_ids() const override { return {0}; }

    // Returns {logits, hidden_state} — hidden state before lm_head
    std::pair<mlx::core::array, mlx::core::array> forward_with_hidden(const mlx::core::array& tokens, int offset = 0);

    // Point detection: returns vector of {x, y} normalized coordinates
    std::vector<std::pair<float, float>> detect_points(const std::vector<int>& prompt_tokens, int max_objects = 50);

    // Object detection: returns vector of {x_min, y_min, x_max, y_max} bounding boxes
    std::vector<std::array<float, 4>> detect_objects(const std::vector<int>& prompt_tokens, int max_objects = 50);

private:
    ModelConfig config_;
    MoondreamConfig md_config_;

    // Token embedding
    std::optional<mlx::core::array> wte_;

    // 24 decoder layers
    std::vector<MoondreamDecoderLayer> layers_;

    // LM head with its own LayerNorm
    std::optional<mlx::core::array> lm_head_ln_weight_, lm_head_ln_bias_;
    std::optional<mlx::core::array> lm_head_weight_, lm_head_bias_;

    // KV cache
    std::unique_ptr<KVCache> kv_cache_;

    // Vision components
    std::unique_ptr<MoondreamVisionEncoder> vision_encoder_;
    std::unique_ptr<MoondreamImageProcessor> image_processor_;

    // Vision projection MLP
    std::optional<mlx::core::array> proj_fc1_weight_, proj_fc1_bias_;
    std::optional<mlx::core::array> proj_fc2_weight_, proj_fc2_bias_;

    // Pending image data
    std::optional<MoondreamImageProcessorResult> pending_image_;

    // Track whether vision prefill has been done
    bool has_done_vision_prefill_ = false;

    // Region model weights (coordinate encoder/decoder)
    std::optional<mlx::core::array> coord_features_;       // (1, 128) - Fourier basis
    std::optional<mlx::core::array> coord_encoder_weight_; // (2048, 256)
    std::optional<mlx::core::array> coord_encoder_bias_;   // (2048)
    std::optional<mlx::core::array> coord_dec_fc1_weight_; // (8192, 2048)
    std::optional<mlx::core::array> coord_dec_fc1_bias_;   // (8192)
    std::optional<mlx::core::array> coord_dec_fc2_weight_; // (1024, 8192)
    std::optional<mlx::core::array> coord_dec_fc2_bias_;   // (1024)

    // Region model weights (size encoder/decoder)
    std::optional<mlx::core::array> size_features_;        // (2, 256) - Fourier basis
    std::optional<mlx::core::array> size_encoder_weight_;  // (2048, 512)
    std::optional<mlx::core::array> size_encoder_bias_;    // (2048)
    std::optional<mlx::core::array> size_dec_fc1_weight_;  // (8192, 2048)
    std::optional<mlx::core::array> size_dec_fc1_bias_;    // (8192)
    std::optional<mlx::core::array> size_dec_fc2_weight_;  // (2048, 8192)
    std::optional<mlx::core::array> size_dec_fc2_bias_;    // (2048)

    // Helpers
    mlx::core::array self_attention(int layer_idx, const mlx::core::array& x,
                                     const std::optional<mlx::core::array>& mask_arr, int offset);
    mlx::core::array mlp(int layer_idx, const mlx::core::array& x);
    mlx::core::array project_vision(const mlx::core::array& vision_out);
    mlx::core::array build_prefix_mask(int prefix_len, int total_seq);

    // Region model helpers
    mlx::core::array fourier_features(const mlx::core::array& x, const mlx::core::array& w);
    mlx::core::array encode_coordinate(float coord);
    mlx::core::array decode_coordinate(const mlx::core::array& hidden);
    mlx::core::array encode_size(float w, float h);
    mlx::core::array decode_size(const mlx::core::array& hidden);
    std::vector<std::pair<float,float>> generate_points(mlx::core::array hidden, int next_token, int pos, bool include_size, int max_objects);
};

} // namespace gomlx
