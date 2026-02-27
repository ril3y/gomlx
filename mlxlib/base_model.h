#pragma once

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace gomlx {

class BaseModel {
public:
    virtual ~BaseModel() = default;

    // Run forward pass: tokens -> logits
    // offset is the position offset for KV cache (number of previously cached tokens)
    virtual mlx::core::array forward(const mlx::core::array& tokens, int offset = 0) = 0;

    // Load model weights from a directory containing safetensors files
    virtual void load_weights(const std::string& path) = 0;

    // Return the vocabulary size
    virtual int vocab_size() const = 0;

    // Whether this model supports vision/image inputs
    virtual bool supports_vision() const { return false; }

    // Reset the KV cache for a new generation
    virtual void reset_cache() = 0;

    // Vision: load image from file. Default throws "not supported".
    virtual void set_image_from_file(const std::string& path) {
        throw std::runtime_error("Model does not support vision");
    }

    // Vision: load image from bytes. Default throws "not supported".
    virtual void set_image_from_bytes(const uint8_t* data, int len) {
        throw std::runtime_error("Model does not support vision");
    }

    // Vision: how many placeholder tokens does the chat template need?
    // Returns 0 for models that don't use placeholder tokens (default).
    virtual int pending_vision_token_count() const { return 0; }

    // Stop tokens: each model declares its own. Pure virtual — compile error if forgotten.
    virtual std::vector<int> stop_token_ids() const = 0;
};

} // namespace gomlx
