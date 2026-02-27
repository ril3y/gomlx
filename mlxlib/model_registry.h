#pragma once

#include "base_model.h"
#include "config.h"

#include <memory>
#include <string>

namespace gomlx {

// Create a model based on the config.json in the model directory.
// Reads config.json, matches "model_type" to a known architecture,
// and returns the appropriate model instance.
// Currently supported: "llama"
std::unique_ptr<BaseModel> create_model(const ModelConfig& config);

// Convenience: parse config and create model in one step.
std::unique_ptr<BaseModel> create_model_from_path(const std::string& model_path);

} // namespace gomlx
