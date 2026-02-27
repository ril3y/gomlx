#pragma once

#include <string>
#include <vector>
#include <utility>

namespace gomlx {

struct VisionModelConfig {
    int image_size = 560;
    int patch_size = 14;
    int num_channels = 3;
    int hidden_size = 1280;
    int intermediate_size = 5120;
    int num_hidden_layers = 32;
    int num_attention_heads = 16;
    int num_global_layers = 8;
    int max_num_tiles = 4;
    int max_aspect_ratio_id = 8;
    float norm_eps = 1e-5f;
    int vision_output_dim = 7680;
    std::vector<int> intermediate_layers_indices;
    std::vector<std::pair<int,int>> supported_aspect_ratios;

    int num_patches_per_tile() const {
        return (image_size / patch_size) * (image_size / patch_size);
    }
};

struct TextModelConfig {
    std::vector<int> cross_attention_layers;
};

struct Qwen2_5VisionConfig {
    int depth = 32;
    int hidden_size = 1280;
    int intermediate_size = 3420;
    int out_hidden_size = 3584;
    int num_heads = 16;
    int patch_size = 14;
    int temporal_patch_size = 2;
    int spatial_merge_size = 2;
    int window_size = 112;
    int in_channels = 3;
    float layer_norm_eps = 1e-6f;
    std::vector<int> fullatt_block_indexes;  // {7,15,23,31}
};

struct ModelConfig {
    std::string model_type;
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int intermediate_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int vocab_size = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int max_position_embeddings = 2048;
    int head_dim = 0; // if 0, computed as hidden_size / num_attention_heads
    bool tie_word_embeddings = true;

    bool is_vision_model = false;
    int image_token_index = 128256;
    VisionModelConfig vision_config;
    TextModelConfig text_config;
    Qwen2_5VisionConfig qwen_vision_config;
    std::vector<int> mrope_sections;

    // Compute head_dim if not explicitly set
    int get_head_dim() const {
        if (head_dim > 0) return head_dim;
        if (num_attention_heads > 0) return hidden_size / num_attention_heads;
        return 0;
    }
};

// Parse config.json from a model directory.
// model_path should be the directory containing config.json.
ModelConfig parse_config(const std::string& model_path);

} // namespace gomlx
