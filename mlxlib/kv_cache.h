#pragma once

#include <optional>
#include <utility>
#include <vector>
#include "mlx/mlx.h"

namespace gomlx {

// KV cache for transformer autoregressive generation.
// Stores key and value tensors per layer, concatenating along the
// sequence dimension (dim 2) as new tokens are generated.
// Expected tensor shape: [batch, num_heads, seq_len, head_dim]
class KVCache {
public:
    explicit KVCache(int num_layers);

    // Update cache for a given layer by concatenating new keys/values
    // along the sequence dimension (axis 2).
    // Returns the full (concatenated) keys and values for this layer.
    std::pair<mlx::core::array, mlx::core::array> update(
        int layer,
        const mlx::core::array& keys,
        const mlx::core::array& values);

    // Current sequence length in the cache (from layer 0).
    int sequence_length() const;

    // Clear all cached data.
    void reset();

    int num_layers() const { return num_layers_; }

private:
    int num_layers_;
    // Per-layer storage: keys and values (optional since array has no default ctor)
    std::vector<std::optional<mlx::core::array>> keys_;
    std::vector<std::optional<mlx::core::array>> values_;
};

} // namespace gomlx
