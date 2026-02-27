#include "kv_cache.h"

#include <stdexcept>

namespace gomlx {

using namespace mlx::core;

KVCache::KVCache(int num_layers)
    : num_layers_(num_layers),
      keys_(num_layers, std::nullopt),
      values_(num_layers, std::nullopt) {}

std::pair<array, array> KVCache::update(
    int layer,
    const array& new_keys,
    const array& new_values) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("KVCache: layer index out of range");
    }

    if (!keys_[layer].has_value()) {
        keys_[layer] = new_keys;
        values_[layer] = new_values;
    } else {
        // Concatenate along sequence dimension (axis 2)
        // Shape: [batch, num_heads, seq_len, head_dim]
        keys_[layer] = concatenate({*keys_[layer], new_keys}, 2);
        values_[layer] = concatenate({*values_[layer], new_values}, 2);
    }

    return {*keys_[layer], *values_[layer]};
}

int KVCache::sequence_length() const {
    if (num_layers_ > 0 && keys_[0].has_value()) {
        return keys_[0]->shape(2);
    }
    return 0;
}

void KVCache::reset() {
    for (int i = 0; i < num_layers_; i++) {
        keys_[i] = std::nullopt;
        values_[i] = std::nullopt;
    }
}

} // namespace gomlx
