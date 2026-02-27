#include "config.h"
#include "third_party/json.hpp"

#include <fstream>
#include <stdexcept>

namespace gomlx {

ModelConfig parse_config(const std::string& model_path) {
    std::string config_file = model_path + "/config.json";
    std::ifstream ifs(config_file);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_file);
    }

    nlohmann::json j;
    ifs >> j;

    ModelConfig config;

    if (j.contains("model_type")) {
        config.model_type = j["model_type"].get<std::string>();
    }
    if (j.contains("hidden_size")) {
        config.hidden_size = j["hidden_size"].get<int>();
    }
    if (j.contains("num_hidden_layers")) {
        config.num_hidden_layers = j["num_hidden_layers"].get<int>();
    }
    if (j.contains("intermediate_size")) {
        config.intermediate_size = j["intermediate_size"].get<int>();
    }
    if (j.contains("num_attention_heads")) {
        config.num_attention_heads = j["num_attention_heads"].get<int>();
    }
    if (j.contains("num_key_value_heads")) {
        config.num_key_value_heads = j["num_key_value_heads"].get<int>();
    }
    if (j.contains("vocab_size")) {
        config.vocab_size = j["vocab_size"].get<int>();
    }
    if (j.contains("rms_norm_eps")) {
        config.rms_norm_eps = j["rms_norm_eps"].get<float>();
    }
    if (j.contains("rope_theta")) {
        config.rope_theta = j["rope_theta"].get<float>();
    }
    if (j.contains("max_position_embeddings")) {
        config.max_position_embeddings = j["max_position_embeddings"].get<int>();
    }
    if (j.contains("head_dim")) {
        config.head_dim = j["head_dim"].get<int>();
    }
    if (j.contains("tie_word_embeddings")) {
        config.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    }

    // Handle mllama (Llama 3.2 Vision) model type
    if (config.model_type == "mllama") {
        config.is_vision_model = true;

        if (j.contains("image_token_index")) {
            config.image_token_index = j["image_token_index"].get<int>();
        }

        // Parse nested text_config
        if (j.contains("text_config")) {
            auto& tc = j["text_config"];
            if (tc.contains("hidden_size")) config.hidden_size = tc["hidden_size"].get<int>();
            if (tc.contains("num_hidden_layers")) config.num_hidden_layers = tc["num_hidden_layers"].get<int>();
            if (tc.contains("intermediate_size")) config.intermediate_size = tc["intermediate_size"].get<int>();
            if (tc.contains("num_attention_heads")) config.num_attention_heads = tc["num_attention_heads"].get<int>();
            if (tc.contains("num_key_value_heads")) config.num_key_value_heads = tc["num_key_value_heads"].get<int>();
            if (tc.contains("vocab_size")) config.vocab_size = tc["vocab_size"].get<int>();
            if (tc.contains("rms_norm_eps")) config.rms_norm_eps = tc["rms_norm_eps"].get<float>();
            if (tc.contains("rope_theta")) config.rope_theta = tc["rope_theta"].get<float>();
            if (tc.contains("max_position_embeddings")) config.max_position_embeddings = tc["max_position_embeddings"].get<int>();
            if (tc.contains("tie_word_embeddings")) config.tie_word_embeddings = tc["tie_word_embeddings"].get<bool>();
            if (tc.contains("head_dim")) config.head_dim = tc["head_dim"].get<int>();
            if (tc.contains("cross_attention_layers")) {
                for (auto& el : tc["cross_attention_layers"]) {
                    config.text_config.cross_attention_layers.push_back(el.get<int>());
                }
            }
        }

        // Parse nested vision_config
        if (j.contains("vision_config")) {
            auto& vc = j["vision_config"];
            auto& vcfg = config.vision_config;
            if (vc.contains("image_size")) vcfg.image_size = vc["image_size"].get<int>();
            if (vc.contains("patch_size")) vcfg.patch_size = vc["patch_size"].get<int>();
            if (vc.contains("num_channels")) vcfg.num_channels = vc["num_channels"].get<int>();
            if (vc.contains("hidden_size")) vcfg.hidden_size = vc["hidden_size"].get<int>();
            if (vc.contains("intermediate_size")) vcfg.intermediate_size = vc["intermediate_size"].get<int>();
            if (vc.contains("num_hidden_layers")) vcfg.num_hidden_layers = vc["num_hidden_layers"].get<int>();
            if (vc.contains("num_attention_heads")) vcfg.num_attention_heads = vc["num_attention_heads"].get<int>();
            else if (vc.contains("attention_heads")) vcfg.num_attention_heads = vc["attention_heads"].get<int>();
            if (vc.contains("num_global_layers")) vcfg.num_global_layers = vc["num_global_layers"].get<int>();
            if (vc.contains("max_num_tiles")) vcfg.max_num_tiles = vc["max_num_tiles"].get<int>();
            if (vc.contains("max_aspect_ratio_id")) vcfg.max_aspect_ratio_id = vc["max_aspect_ratio_id"].get<int>();
            if (vc.contains("norm_eps")) vcfg.norm_eps = vc["norm_eps"].get<float>();
            if (vc.contains("vision_output_dim")) vcfg.vision_output_dim = vc["vision_output_dim"].get<int>();
            if (vc.contains("intermediate_layers_indices")) {
                for (auto& el : vc["intermediate_layers_indices"]) {
                    vcfg.intermediate_layers_indices.push_back(el.get<int>());
                }
            }
            if (vc.contains("supported_aspect_ratios")) {
                for (auto& pair : vc["supported_aspect_ratios"]) {
                    vcfg.supported_aspect_ratios.push_back({pair[0].get<int>(), pair[1].get<int>()});
                }
            }
        }

        // Set defaults if not provided
        if (config.vision_config.intermediate_layers_indices.empty()) {
            config.vision_config.intermediate_layers_indices = {3, 7, 15, 23, 30};
        }
        if (config.vision_config.supported_aspect_ratios.empty()) {
            config.vision_config.supported_aspect_ratios = {
                {1,1}, {1,2}, {1,3}, {1,4}, {2,1}, {2,2}, {3,1}, {4,1}
            };
        }
    }

    // Handle Qwen2.5-VL model type
    if (config.model_type == "qwen2_5_vl") {
        config.is_vision_model = true;

        if (j.contains("image_token_id")) {
            config.image_token_index = j["image_token_id"].get<int>();
        }

        // Parse rope_scaling.mrope_section
        if (j.contains("rope_scaling")) {
            auto& rs = j["rope_scaling"];
            if (rs.contains("mrope_section")) {
                for (auto& el : rs["mrope_section"]) {
                    config.mrope_sections.push_back(el.get<int>());
                }
            }
        }

        // Parse nested vision_config
        if (j.contains("vision_config")) {
            auto& vc = j["vision_config"];
            auto& qvc = config.qwen_vision_config;
            if (vc.contains("depth")) qvc.depth = vc["depth"].get<int>();
            if (vc.contains("hidden_size")) qvc.hidden_size = vc["hidden_size"].get<int>();
            if (vc.contains("intermediate_size")) qvc.intermediate_size = vc["intermediate_size"].get<int>();
            if (vc.contains("out_hidden_size")) qvc.out_hidden_size = vc["out_hidden_size"].get<int>();
            if (vc.contains("num_heads")) qvc.num_heads = vc["num_heads"].get<int>();
            if (vc.contains("patch_size")) qvc.patch_size = vc["patch_size"].get<int>();
            if (vc.contains("temporal_patch_size")) qvc.temporal_patch_size = vc["temporal_patch_size"].get<int>();
            if (vc.contains("spatial_merge_size")) qvc.spatial_merge_size = vc["spatial_merge_size"].get<int>();
            if (vc.contains("window_size")) qvc.window_size = vc["window_size"].get<int>();
            if (vc.contains("in_chans")) qvc.in_channels = vc["in_chans"].get<int>();
            if (vc.contains("fullatt_block_indexes")) {
                for (auto& el : vc["fullatt_block_indexes"]) {
                    qvc.fullatt_block_indexes.push_back(el.get<int>());
                }
            }
        }
    }

    // Default num_key_value_heads to num_attention_heads if not specified (MHA)
    if (config.num_key_value_heads == 0 && config.num_attention_heads > 0) {
        config.num_key_value_heads = config.num_attention_heads;
    }

    return config;
}

} // namespace gomlx
