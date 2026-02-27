#pragma once

#include "mlx/mlx.h"

namespace gomlx {

// Sample a token from logits.
// - If temperature <= 0: greedy argmax
// - If temperature > 0: scale logits by temperature, apply softmax,
//   then nucleus (top_p) sampling
int sample_token(const mlx::core::array& logits, float temperature = 0.0f, float top_p = 1.0f);

} // namespace gomlx
