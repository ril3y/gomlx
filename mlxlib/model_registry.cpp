#include "model_registry.h"
#include "models/llama/llama_text.h"
#include "models/llama/llama_vision.h"
#include "models/mistral/mistral_text.h"
#include "models/qwen/qwen2_text.h"
#include "models/qwen/qwen2_5_vl.h"
#include "models/gemma/gemma2_text.h"

#include <stdexcept>

namespace gomlx {

std::unique_ptr<BaseModel> create_model(const ModelConfig& config) {
    if (config.model_type == "llama") {
        return std::make_unique<LlamaTextModel>(config);
    }
    if (config.model_type == "mllama") {
        return std::make_unique<LlamaVisionModel>(config);
    }
    if (config.model_type == "mistral") {
        return std::make_unique<MistralTextModel>(config);
    }
    if (config.model_type == "qwen2") {
        return std::make_unique<Qwen2TextModel>(config);
    }
    if (config.model_type == "qwen2_5_vl") {
        return std::make_unique<Qwen2_5VLModel>(config);
    }
    if (config.model_type == "gemma2") {
        return std::make_unique<Gemma2TextModel>(config);
    }

    throw std::runtime_error(
        "Unsupported model type: '" + config.model_type +
        "'. Supported types: llama, mllama, mistral, qwen2, qwen2_5_vl, gemma2");
}

std::unique_ptr<BaseModel> create_model_from_path(const std::string& model_path) {
    ModelConfig config = parse_config(model_path);
    return create_model(config);
}

} // namespace gomlx
