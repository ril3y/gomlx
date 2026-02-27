#include "sampling.h"

namespace gomlx {

using namespace mlx::core;

int sample_token(const array& logits, float temperature, float top_p) {
    // logits shape: [batch, seq_len, vocab_size] or [seq_len, vocab_size] or [vocab_size]
    // We want the last token's logits as a 1D array of [vocab_size].
    array token_logits = logits;

    // Reduce to 1D [vocab_size] by taking the last position along all leading dims
    while (token_logits.ndim() > 1) {
        int last_pos = token_logits.shape(0) - 1;
        token_logits = take(token_logits, last_pos, 0);
    }

    // Greedy: argmax
    if (temperature <= 0.0f) {
        array token = argmax(token_logits);
        eval(token);
        return token.item<int>();
    }

    // Temperature scaling
    token_logits = divide(token_logits, array(temperature));

    // If top_p >= 1.0, just use categorical sampling directly (no nucleus filtering)
    if (top_p >= 1.0f) {
        // categorical expects logits (unnormalized log-probs)
        array token = random::categorical(token_logits);
        eval(token);
        return token.item<int>();
    }

    // Nucleus (top_p) sampling
    // 1. Compute softmax probabilities
    array probs = softmax(token_logits);

    // 2. Sort probabilities in descending order
    array sorted_indices = argsort(probs);
    // Reverse to get descending order
    array rev_idx = arange(static_cast<int>(probs.shape(0)) - 1, -1, -1);
    sorted_indices = take(sorted_indices, rev_idx);
    array sorted_probs = take(probs, sorted_indices);

    // 3. Compute cumulative sum
    array cumsum_probs = cumsum(sorted_probs);

    // 4. Create mask: keep tokens where cumulative prob hasn't exceeded top_p
    // Shift cumsum right by 1 so we include the token that crosses the threshold
    array shifted_cumsum = subtract(cumsum_probs, sorted_probs);
    array mask = less(shifted_cumsum, array(top_p));

    // 5. Zero out probabilities below the threshold
    sorted_probs = multiply(sorted_probs, astype(mask, sorted_probs.dtype()));

    // 6. Renormalize
    array sum_probs = sum(sorted_probs);
    sorted_probs = divide(sorted_probs, sum_probs);

    // 7. Sample from filtered distribution using categorical (takes logits)
    array filtered_logits = log(maximum(sorted_probs, array(1e-12f)));
    array sampled_idx = random::categorical(filtered_logits);
    eval(sampled_idx);

    // 8. Map back to original vocab index
    int idx_in_sorted = sampled_idx.item<int>();
    eval(sorted_indices);
    array original_idx = take(sorted_indices, idx_in_sorted);
    eval(original_idx);
    return original_idx.item<int>();
}

} // namespace gomlx
