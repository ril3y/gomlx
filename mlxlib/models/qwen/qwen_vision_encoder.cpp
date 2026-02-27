#include "models/qwen/qwen_vision_encoder.h"

#include <cmath>
#include <stdexcept>

namespace gomlx {

using namespace mlx::core;

// GELU activation (exact form)
static array gelu(const array& x) {
    auto cdf = multiply(
        array(0.5f, float32),
        add(array(1.0f, float32),
            erf(divide(x, array(std::sqrt(2.0f), float32)))));
    return multiply(x, cdf);
}

// SiLU activation: x * sigmoid(x)
static array silu(const array& x) {
    return multiply(x, sigmoid(x));
}

// rotate_half for 2D RoPE: applied independently to two sub-blocks of head_dim
// x: [..., head_dim] where head_dim is split into two halves (row and col)
// Each half uses the standard rotate_half (split at midpoint within the half)
static array rotate_half_2d(const array& x, int head_dim) {
    int half = head_dim / 2;      // 40: size of each spatial sub-block
    int quarter = half / 2;        // 20: midpoint within each sub-block

    // Row sub-block: dims [0..quarter) and [quarter..half)
    auto x_r1 = slice(x, {0, 0, 0, 0},
                       {x.shape(0), x.shape(1), x.shape(2), quarter});
    auto x_r2 = slice(x, {0, 0, 0, quarter},
                       {x.shape(0), x.shape(1), x.shape(2), half});

    // Col sub-block: dims [half..half+quarter) and [half+quarter..head_dim)
    auto x_c1 = slice(x, {0, 0, 0, half},
                       {x.shape(0), x.shape(1), x.shape(2), half + quarter});
    auto x_c2 = slice(x, {0, 0, 0, half + quarter},
                       {x.shape(0), x.shape(1), x.shape(2), head_dim});

    // rotate_half within each sub-block: [-second, first]
    return concatenate({negative(x_r2), x_r1, negative(x_c2), x_c1}, -1);
}

// Apply 2D RoPE: x * cos + rotate_half(x) * sin
static array apply_rope_2d(const array& x, const array& cos_t,
                            const array& sin_t, int head_dim) {
    // x: [batch, num_heads, seq, head_dim]
    // cos_t, sin_t: [seq, head_dim]
    auto c = reshape(cos_t, {1, 1, cos_t.shape(0), cos_t.shape(1)});
    auto s = reshape(sin_t, {1, 1, sin_t.shape(0), sin_t.shape(1)});
    auto rotated = rotate_half_2d(x, head_dim);
    return add(multiply(x, c), multiply(rotated, s));
}

QwenVisionEncoder::QwenVisionEncoder(const Qwen2_5VisionConfig& config)
    : config_(config) {
    head_dim_ = config_.hidden_size / config_.num_heads;  // 80
    blocks_.resize(config_.depth);
    for (int idx : config_.fullatt_block_indexes) {
        fullatt_set_.insert(idx);
    }
}

std::pair<array, array> QwenVisionEncoder::compute_2d_rope(int grid_h, int grid_w) {
    // head_dim = 80, split into two 40-dim sub-blocks (row, col)
    // Each sub-block uses 20 frequency pairs with the half-split rotate convention
    int rope_half = head_dim_ / 2;  // 40 dims per spatial axis
    int num_freq = rope_half / 2;   // 20 frequency pairs
    float theta = 10000.0f;
    int seq_len = grid_h * grid_w;

    // Compute base frequencies: 1 / (theta ^ (2i / rope_half))
    std::vector<float> base_freq(num_freq);
    for (int i = 0; i < num_freq; i++) {
        base_freq[i] = 1.0f / std::pow(theta, (2.0f * i) / static_cast<float>(rope_half));
    }

    // Build cos/sin tables [seq_len, head_dim] using half-split convention
    std::vector<float> cos_data(seq_len * head_dim_, 1.0f);
    std::vector<float> sin_data(seq_len * head_dim_, 0.0f);

    for (int r = 0; r < grid_h; r++) {
        for (int c = 0; c < grid_w; c++) {
            int pos = r * grid_w + c;

            // Row sub-block: dims [0..39], half-split at dim 20
            for (int i = 0; i < num_freq; i++) {
                float angle = static_cast<float>(r) * base_freq[i];
                float cv = std::cos(angle);
                float sv = std::sin(angle);
                // First half of row sub-block (dims 0..19)
                cos_data[pos * head_dim_ + i] = cv;
                sin_data[pos * head_dim_ + i] = sv;
                // Second half of row sub-block (dims 20..39)
                cos_data[pos * head_dim_ + num_freq + i] = cv;
                sin_data[pos * head_dim_ + num_freq + i] = sv;
            }

            // Col sub-block: dims [40..79], half-split at dim 60
            int col_offset = rope_half;
            for (int i = 0; i < num_freq; i++) {
                float angle = static_cast<float>(c) * base_freq[i];
                float cv = std::cos(angle);
                float sv = std::sin(angle);
                // First half of col sub-block (dims 40..59)
                cos_data[pos * head_dim_ + col_offset + i] = cv;
                sin_data[pos * head_dim_ + col_offset + i] = sv;
                // Second half of col sub-block (dims 60..79)
                cos_data[pos * head_dim_ + col_offset + num_freq + i] = cv;
                sin_data[pos * head_dim_ + col_offset + num_freq + i] = sv;
            }
        }
    }

    return {
        array(cos_data.data(), {seq_len, head_dim_}, float32),
        array(sin_data.data(), {seq_len, head_dim_}, float32)
    };
}

array QwenVisionEncoder::run_block(QwenVisionBlock& block, const array& x,
                                    const array& rope_cos, const array& rope_sin,
                                    bool full_attention, int grid_h, int grid_w) {
    int batch = x.shape(0);
    int seq_len = x.shape(1);
    int hidden = config_.hidden_size;
    int num_heads = config_.num_heads;

    // 1. Pre-attention RMSNorm
    auto normed = fast::rms_norm(x, *block.norm1_weight, config_.layer_norm_eps);

    // 2. QKV projection: [batch, seq, hidden] -> [batch, seq, 3*hidden]
    auto qkv = add(matmul(normed, transpose(*block.qkv_weight)), *block.qkv_bias);

    // 3. Split into Q, K, V: each [batch, seq, hidden]
    auto q = slice(qkv, {0, 0, 0}, {batch, seq_len, hidden});
    auto k = slice(qkv, {0, 0, hidden}, {batch, seq_len, 2 * hidden});
    auto v = slice(qkv, {0, 0, 2 * hidden}, {batch, seq_len, 3 * hidden});

    // 4. Reshape to [batch, num_heads, seq, head_dim]
    q = transpose(reshape(q, {batch, seq_len, num_heads, head_dim_}), {0, 2, 1, 3});
    k = transpose(reshape(k, {batch, seq_len, num_heads, head_dim_}), {0, 2, 1, 3});
    v = transpose(reshape(v, {batch, seq_len, num_heads, head_dim_}), {0, 2, 1, 3});

    // 5. Apply 2D RoPE to Q and K (global positions)
    q = apply_rope_2d(q, rope_cos, rope_sin, head_dim_);
    k = apply_rope_2d(k, rope_cos, rope_sin, head_dim_);

    // 6. Attention (full or windowed)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    auto run_attention = [&]() -> array {
        if (full_attention) {
            return fast::scaled_dot_product_attention(q, k, v, scale, "");
        }

        // Windowed attention: partition into windows of win_patches x win_patches
        int win = config_.window_size / config_.patch_size;  // 8

        // Pad grid to multiples of window size
        int padded_h = ((grid_h + win - 1) / win) * win;
        int padded_w = ((grid_w + win - 1) / win) * win;

        // Rearrange Q, K, V from [batch, heads, seq, dim] to windowed form
        auto reshape_to_windows = [&](const array& tensor) -> array {
            auto t = reshape(tensor, {batch, num_heads, grid_h, grid_w, head_dim_});

            if (padded_h != grid_h || padded_w != grid_w) {
                int pad_h = padded_h - grid_h;
                int pad_w = padded_w - grid_w;
                t = pad(t, {{0, 0}, {0, 0}, {0, pad_h}, {0, pad_w}, {0, 0}});
            }

            int nwh = padded_h / win;
            int nww = padded_w / win;
            t = reshape(t, {batch, num_heads, nwh, win, nww, win, head_dim_});
            t = transpose(t, {0, 1, 2, 4, 3, 5, 6});
            t = reshape(t, {batch * nwh * nww, num_heads, win * win, head_dim_});
            return t;
        };

        auto q_win = reshape_to_windows(q);
        auto k_win = reshape_to_windows(k);
        auto v_win = reshape_to_windows(v);

        auto win_out = fast::scaled_dot_product_attention(q_win, k_win, v_win, scale, "");

        int nwh = padded_h / win;
        int nww = padded_w / win;
        win_out = reshape(win_out, {batch, nwh, nww, num_heads, win, win, head_dim_});
        win_out = transpose(win_out, {0, 3, 1, 4, 2, 5, 6});
        win_out = reshape(win_out, {batch, num_heads, padded_h, padded_w, head_dim_});

        if (padded_h != grid_h || padded_w != grid_w) {
            win_out = slice(win_out, {0, 0, 0, 0, 0},
                            {batch, num_heads, grid_h, grid_w, head_dim_});
        }

        return reshape(win_out, {batch, num_heads, seq_len, head_dim_});
    };

    auto attn_out = run_attention();

    // 7. Reshape attention output: [batch, heads, seq, dim] -> [batch, seq, hidden]
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch, seq_len, hidden});

    // 8. Output projection + residual
    attn_out = add(matmul(attn_out, transpose(*block.proj_weight)), *block.proj_bias);
    auto h = add(x, attn_out);

    // 9. Post-attention RMSNorm
    normed = fast::rms_norm(h, *block.norm2_weight, config_.layer_norm_eps);

    // 10. SwiGLU MLP: silu(gate_proj(x)) * up_proj(x) -> down_proj
    auto gate = add(matmul(normed, transpose(*block.gate_proj_weight)), *block.gate_proj_bias);
    auto up = add(matmul(normed, transpose(*block.up_proj_weight)), *block.up_proj_bias);
    auto mlp_out = multiply(silu(gate), up);
    mlp_out = add(matmul(mlp_out, transpose(*block.down_proj_weight)), *block.down_proj_bias);

    // 11. MLP residual
    return add(h, mlp_out);
}

array QwenVisionEncoder::forward(const array& pixel_values, int grid_h, int grid_w) {
    int hidden = config_.hidden_size;
    int merge = config_.spatial_merge_size;

    // --- 1. Patch embedding ---
    // Input: [1, 3, 2, H, W] (batch, channels, temporal, height, width)
    // Take first temporal frame: [1, 3, H, W]
    auto frame = slice(pixel_values, {0, 0, 0, 0, 0},
                       {1, pixel_values.shape(1), 1, pixel_values.shape(3), pixel_values.shape(4)});
    // Squeeze temporal dim: [1, 3, H, W]
    frame = reshape(frame, {1, pixel_values.shape(1), pixel_values.shape(3), pixel_values.shape(4)});
    // Transpose NCHW -> NHWC: [1, H, W, 3]
    frame = transpose(frame, {0, 2, 3, 1});

    // Conv2D with stride=patch_size, using temporally-summed kernel
    // patch_embed_weight_: [1280, 14, 14, 3] (OHWI format for MLX)
    auto x = conv2d(frame, *patch_embed_weight_,
                    /*stride=*/{config_.patch_size, config_.patch_size});
    // Result: [1, grid_h, grid_w, 1280]

    // Flatten spatial dims: [1, grid_h*grid_w, 1280]
    int seq_len = grid_h * grid_w;
    x = reshape(x, {1, seq_len, hidden});

    // --- 2. Compute 2D RoPE embeddings ---
    auto [rope_cos, rope_sin] = compute_2d_rope(grid_h, grid_w);
    // --- 3. Run 32 ViT blocks ---
    for (int i = 0; i < config_.depth; i++) {
        bool full_att = fullatt_set_.count(i) > 0;
        x = run_block(blocks_[i], x, rope_cos, rope_sin, full_att, grid_h, grid_w);
    }

    // --- 4. PatchMerger ---
    // RMSNorm
    x = fast::rms_norm(x, *merger_ln_weight_, config_.layer_norm_eps);
    // Reshape for 2x2 spatial merge
    x = reshape(x, {1, grid_h, grid_w, hidden});

    int merged_h = grid_h / merge;
    int merged_w = grid_w / merge;

    // Rearrange into 2x2 blocks: [1, merged_h, merge, merged_w, merge, hidden]
    x = reshape(x, {1, merged_h, merge, merged_w, merge, hidden});
    // Transpose to group merge patches together: [1, merged_h, merged_w, merge, merge, hidden]
    x = transpose(x, {0, 1, 3, 2, 4, 5});
    // Flatten merge patches: [1, merged_h*merged_w, merge*merge*hidden]
    int merged_tokens = merged_h * merged_w;
    int merged_dim = merge * merge * hidden;  // 4 * 1280 = 5120
    x = reshape(x, {1, merged_tokens, merged_dim});

    // MLP: Linear(5120, 5120) -> GELU -> Linear(5120, out_hidden_size)
    x = add(matmul(x, transpose(*merger_mlp0_weight_)), *merger_mlp0_bias_);
    x = gelu(x);
    x = add(matmul(x, transpose(*merger_mlp2_weight_)), *merger_mlp2_bias_);

    // Output: [1, num_vision_tokens, out_hidden_size (3584)]
    return x;
}

void QwenVisionEncoder::load_weights(
    const std::unordered_map<std::string, array>& weights) {

    auto find_weight = [&](const std::string& name) -> const array& {
        auto it = weights.find(name);
        if (it == weights.end()) {
            throw std::runtime_error("Missing Qwen vision weight: " + name);
        }
        return it->second;
    };

    // --- Patch embedding: Conv3D -> Conv2D ---
    {
        auto patch_w = find_weight("vision_tower.patch_embed.proj.weight");
        // Shape: [1280, 2, 14, 14, 3] - sum over temporal dim (axis 1)
        // Result: [1280, 14, 14, 3] (OHWI format, what MLX conv2d expects)
        patch_embed_weight_ = sum(patch_w, 1);
    }

    // --- Transformer blocks ---
    for (int i = 0; i < config_.depth; i++) {
        std::string prefix = "vision_tower.blocks." + std::to_string(i);
        auto& block = blocks_[i];

        // RMSNorm weights (no bias)
        block.norm1_weight = find_weight(prefix + ".norm1.weight");
        block.norm2_weight = find_weight(prefix + ".norm2.weight");

        // Combined QKV
        block.qkv_weight = find_weight(prefix + ".attn.qkv.weight");
        block.qkv_bias = find_weight(prefix + ".attn.qkv.bias");

        // Output projection
        block.proj_weight = find_weight(prefix + ".attn.proj.weight");
        block.proj_bias = find_weight(prefix + ".attn.proj.bias");

        // SwiGLU MLP
        block.gate_proj_weight = find_weight(prefix + ".mlp.gate_proj.weight");
        block.gate_proj_bias = find_weight(prefix + ".mlp.gate_proj.bias");
        block.up_proj_weight = find_weight(prefix + ".mlp.up_proj.weight");
        block.up_proj_bias = find_weight(prefix + ".mlp.up_proj.bias");
        block.down_proj_weight = find_weight(prefix + ".mlp.down_proj.weight");
        block.down_proj_bias = find_weight(prefix + ".mlp.down_proj.bias");
    }

    // --- PatchMerger ---
    merger_ln_weight_ = find_weight("vision_tower.merger.ln_q.weight");
    merger_mlp0_weight_ = find_weight("vision_tower.merger.mlp.0.weight");
    merger_mlp0_bias_ = find_weight("vision_tower.merger.mlp.0.bias");
    merger_mlp2_weight_ = find_weight("vision_tower.merger.mlp.2.weight");
    merger_mlp2_bias_ = find_weight("vision_tower.merger.mlp.2.bias");
}

} // namespace gomlx
