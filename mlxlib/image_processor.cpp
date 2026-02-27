#include "third_party/stb_image.h"

#include "image_processor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gomlx {

// ImageNet normalization constants
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

ImageProcessor::ImageProcessor(const VisionModelConfig& config)
    : config_(config) {}

ImageProcessorResult ImageProcessor::process_from_file(const std::string& path) {
    int w, h, channels;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    auto result = process_pixels(data, w, h);
    stbi_image_free(data);
    return result;
}

ImageProcessorResult ImageProcessor::process_from_bytes(const uint8_t* data, int len) {
    int w, h, channels;
    uint8_t* pixels = stbi_load_from_memory(data, len, &w, &h, &channels, 3);
    if (!pixels) {
        throw std::runtime_error("Failed to decode image from memory");
    }
    auto result = process_pixels(pixels, w, h);
    stbi_image_free(pixels);
    return result;
}

std::pair<int,int> ImageProcessor::get_optimal_tiled_canvas(int w, int h) {
    const auto& ratios = config_.supported_aspect_ratios;
    int tile_size = config_.image_size;

    std::pair<int,int> best = ratios[0];
    int best_area = best.first * best.second * tile_size * tile_size;
    bool found = false;

    for (auto& [rh, rw] : ratios) {
        int canvas_h = rh * tile_size;
        int canvas_w = rw * tile_size;

        // Check if the image fits in this canvas when scaled to fill one dimension
        // Scale so the image fits within canvas while maintaining aspect ratio
        float scale = std::min(static_cast<float>(canvas_w) / w,
                               static_cast<float>(canvas_h) / h);
        int scaled_w = static_cast<int>(w * scale);
        int scaled_h = static_cast<int>(h * scale);

        if (scaled_w <= canvas_w && scaled_h <= canvas_h) {
            int area = canvas_h * canvas_w;
            if (!found || area < best_area) {
                best = {rh, rw};
                best_area = area;
                found = true;
            }
        }
    }

    return best;
}

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

ImageProcessorResult ImageProcessor::process_pixels(const uint8_t* rgb, int w, int h) {
    int tile_size = config_.image_size;

    // Find optimal tiled canvas
    auto [rh, rw] = get_optimal_tiled_canvas(w, h);
    int canvas_h = rh * tile_size;
    int canvas_w = rw * tile_size;
    int num_tiles = rh * rw;

    // Find aspect_ratio_id (1-indexed)
    int aspect_ratio_id = 1;
    for (int i = 0; i < static_cast<int>(config_.supported_aspect_ratios.size()); i++) {
        if (config_.supported_aspect_ratios[i].first == rh &&
            config_.supported_aspect_ratios[i].second == rw) {
            aspect_ratio_id = i + 1;
            break;
        }
    }

    // Convert uint8 to float [0, 1]
    std::vector<float> src_float(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) {
        src_float[i] = rgb[i] / 255.0f;
    }

    // Compute scaled dimensions maintaining aspect ratio
    float scale = std::min(static_cast<float>(canvas_w) / w,
                           static_cast<float>(canvas_h) / h);
    int scaled_w = static_cast<int>(std::round(w * scale));
    int scaled_h = static_cast<int>(std::round(h * scale));
    scaled_w = std::max(1, std::min(scaled_w, canvas_w));
    scaled_h = std::max(1, std::min(scaled_h, canvas_h));

    // Bilinear resize
    std::vector<float> resized = bilinear_resize(src_float.data(), w, h, scaled_w, scaled_h);

    // Create padded canvas (zero-padded), image centered
    std::vector<float> canvas(canvas_h * canvas_w * 3, 0.0f);
    int offset_y = (canvas_h - scaled_h) / 2;
    int offset_x = (canvas_w - scaled_w) / 2;

    for (int y = 0; y < scaled_h; y++) {
        for (int x = 0; x < scaled_w; x++) {
            int src_idx = (y * scaled_w + x) * 3;
            int dst_idx = ((offset_y + y) * canvas_w + (offset_x + x)) * 3;
            canvas[dst_idx + 0] = resized[src_idx + 0];
            canvas[dst_idx + 1] = resized[src_idx + 1];
            canvas[dst_idx + 2] = resized[src_idx + 2];
        }
    }

    // Normalize with ImageNet mean/std
    for (int i = 0; i < canvas_h * canvas_w; i++) {
        for (int c = 0; c < 3; c++) {
            canvas[i * 3 + c] = (canvas[i * 3 + c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }

    // Split into tiles and rearrange to [num_tiles, 3, tile_h, tile_w] (CHW format)
    // Canvas layout: rh rows x rw cols of tiles, each tile_size x tile_size
    std::vector<float> tiles(num_tiles * 3 * tile_size * tile_size);

    for (int tr = 0; tr < rh; tr++) {
        for (int tc = 0; tc < rw; tc++) {
            int tile_idx = tr * rw + tc;
            for (int c = 0; c < 3; c++) {
                for (int ty = 0; ty < tile_size; ty++) {
                    for (int tx = 0; tx < tile_size; tx++) {
                        int canvas_y = tr * tile_size + ty;
                        int canvas_x = tc * tile_size + tx;
                        float val = canvas[(canvas_y * canvas_w + canvas_x) * 3 + c];
                        int out_idx = tile_idx * (3 * tile_size * tile_size) +
                                      c * (tile_size * tile_size) +
                                      ty * tile_size + tx;
                        tiles[out_idx] = val;
                    }
                }
            }
        }
    }

    // Create MLX array [num_tiles, 3, tile_size, tile_size]
    auto pixel_values = mlx::core::array(
        tiles.data(),
        {num_tiles, 3, tile_size, tile_size},
        mlx::core::float32
    );

    return ImageProcessorResult{pixel_values, aspect_ratio_id, num_tiles};
}

} // namespace gomlx
