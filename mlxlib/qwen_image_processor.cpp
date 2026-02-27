#include "third_party/stb_image.h"

#include "qwen_image_processor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gomlx {

// CLIP normalization constants
static const float CLIP_MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
static const float CLIP_STD[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

// Simple bilinear resize on float RGB data [h, w, 3]
static std::vector<float> bilinear_resize(const float* src, int src_w, int src_h,
                                           int dst_w, int dst_h) {
    std::vector<float> dst(dst_w * dst_h * 3);

    for (int y = 0; y < dst_h; y++) {
        float src_y = (y + 0.5f) * src_h / dst_h - 0.5f;
        src_y = std::max(0.0f, std::min(src_y, static_cast<float>(src_h - 1)));
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, src_h - 1);
        float fy = src_y - y0;

        for (int x = 0; x < dst_w; x++) {
            float src_x = (x + 0.5f) * src_w / dst_w - 0.5f;
            src_x = std::max(0.0f, std::min(src_x, static_cast<float>(src_w - 1)));
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, src_w - 1);
            float fx = src_x - x0;

            for (int c = 0; c < 3; c++) {
                float v00 = src[(y0 * src_w + x0) * 3 + c];
                float v01 = src[(y0 * src_w + x1) * 3 + c];
                float v10 = src[(y1 * src_w + x0) * 3 + c];
                float v11 = src[(y1 * src_w + x1) * 3 + c];

                float v = (1 - fy) * ((1 - fx) * v00 + fx * v01) +
                          fy * ((1 - fx) * v10 + fx * v11);
                dst[(y * dst_w + x) * 3 + c] = v;
            }
        }
    }

    return dst;
}

QwenImageProcessor::QwenImageProcessor(const Qwen2_5VisionConfig& config)
    : config_(config) {}

QwenImageProcessorResult QwenImageProcessor::process_from_file(const std::string& path) {
    int w, h, channels;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    auto result = process_pixels(data, w, h);
    stbi_image_free(data);
    return result;
}

QwenImageProcessorResult QwenImageProcessor::process_from_bytes(const uint8_t* data, int len) {
    int w, h, channels;
    uint8_t* pixels = stbi_load_from_memory(data, len, &w, &h, &channels, 3);
    if (!pixels) {
        throw std::runtime_error("Failed to decode image from memory");
    }
    auto result = process_pixels(pixels, w, h);
    stbi_image_free(pixels);
    return result;
}

std::pair<int,int> QwenImageProcessor::smart_resize(int w, int h) {
    int factor = config_.patch_size * config_.spatial_merge_size; // 14 * 2 = 28

    int target_w = w;
    int target_h = h;

    // Scale down to fit within MAX_PIXELS
    if (static_cast<long long>(target_w) * target_h > MAX_PIXELS) {
        double scale = std::sqrt(static_cast<double>(MAX_PIXELS) / (static_cast<double>(w) * h));
        target_w = static_cast<int>(std::round(w * scale));
        target_h = static_cast<int>(std::round(h * scale));
    }

    // Scale up if below MIN_PIXELS
    if (static_cast<long long>(target_w) * target_h < MIN_PIXELS) {
        double scale = std::sqrt(static_cast<double>(MIN_PIXELS) / (static_cast<double>(target_w) * target_h));
        target_w = static_cast<int>(std::round(target_w * scale));
        target_h = static_cast<int>(std::round(target_h * scale));
    }

    // Round to nearest multiple of factor
    target_w = std::max(factor, static_cast<int>(std::round(static_cast<double>(target_w) / factor)) * factor);
    target_h = std::max(factor, static_cast<int>(std::round(static_cast<double>(target_h) / factor)) * factor);

    return {target_w, target_h};
}

QwenImageProcessorResult QwenImageProcessor::process_pixels(const uint8_t* rgb, int w, int h) {
    // Step 1: Compute target dimensions
    auto [target_w, target_h] = smart_resize(w, h);

    // Step 2: Convert uint8 to float [0, 1]
    std::vector<float> src_float(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) {
        src_float[i] = rgb[i] / 255.0f;
    }

    // Step 3: Bilinear resize to target dimensions
    std::vector<float> resized = bilinear_resize(src_float.data(), w, h, target_w, target_h);

    // Step 4: Normalize with CLIP constants
    for (int i = 0; i < target_h * target_w; i++) {
        for (int c = 0; c < 3; c++) {
            resized[i * 3 + c] = (resized[i * 3 + c] - CLIP_MEAN[c]) / CLIP_STD[c];
        }
    }

    // Step 5: Arrange into [1, 3, 2, target_h, target_w]
    // For each channel c, for temporal frames t=0 and t=1 (both identical): H*W values
    std::vector<float> output(1 * 3 * 2 * target_h * target_w);
    for (int c = 0; c < 3; c++) {
        for (int t = 0; t < 2; t++) {
            for (int y = 0; y < target_h; y++) {
                for (int x = 0; x < target_w; x++) {
                    float val = resized[(y * target_w + x) * 3 + c];
                    // Index: [0, c, t, y, x] in shape [1, 3, 2, H, W]
                    int idx = c * (2 * target_h * target_w) +
                              t * (target_h * target_w) +
                              y * target_w + x;
                    output[idx] = val;
                }
            }
        }
    }

    // Step 6: Create MLX array
    auto pixel_values = mlx::core::array(
        output.data(),
        {1, 3, 2, target_h, target_w},
        mlx::core::float32
    );

    // Step 7: Compute grid dimensions
    int grid_h = target_h / config_.patch_size;
    int grid_w = target_w / config_.patch_size;
    int num_vision_tokens = (grid_h / config_.spatial_merge_size) *
                            (grid_w / config_.spatial_merge_size);

    return QwenImageProcessorResult{pixel_values, grid_h, grid_w, num_vision_tokens};
}

} // namespace gomlx
