#include "third_party/stb_image.h"

#include "models/moondream/moondream_image_processor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gomlx {

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

MoondreamImageProcessorResult MoondreamImageProcessor::process_from_file(const std::string& path) {
    int w, h, channels;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    auto result = process_pixels(data, w, h);
    stbi_image_free(data);
    return result;
}

MoondreamImageProcessorResult MoondreamImageProcessor::process_from_bytes(const uint8_t* data, int len) {
    int w, h, channels;
    uint8_t* pixels = stbi_load_from_memory(data, len, &w, &h, &channels, 3);
    if (!pixels) {
        throw std::runtime_error("Failed to decode image from memory");
    }
    auto result = process_pixels(pixels, w, h);
    stbi_image_free(pixels);
    return result;
}

MoondreamImageProcessorResult MoondreamImageProcessor::process_pixels(const uint8_t* rgb, int w, int h) {
    // Step 1: Convert uint8 to float [0, 1]
    std::vector<float> src_float(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) {
        src_float[i] = rgb[i] / 255.0f;
    }

    // Step 2: Bilinear resize to 378x378
    std::vector<float> resized = bilinear_resize(src_float.data(), w, h, IMAGE_SIZE, IMAGE_SIZE);

    // Step 3: Normalize: (pixel - 0.5) / 0.5
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE * 3; i++) {
        resized[i] = (resized[i] - IMAGE_MEAN) / IMAGE_STD;
    }

    // Step 4: Arrange into [1, 3, 378, 378] (NCHW)
    std::vector<float> output(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                float val = resized[(y * IMAGE_SIZE + x) * 3 + c];
                output[c * (IMAGE_SIZE * IMAGE_SIZE) + y * IMAGE_SIZE + x] = val;
            }
        }
    }

    // Step 5: Create MLX array
    auto pixel_values = mlx::core::array(
        output.data(),
        {1, 3, IMAGE_SIZE, IMAGE_SIZE},
        mlx::core::float32
    );

    return MoondreamImageProcessorResult{pixel_values};
}

} // namespace gomlx
