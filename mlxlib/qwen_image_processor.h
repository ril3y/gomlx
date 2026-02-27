#pragma once

#include <string>
#include <vector>
#include "mlx/mlx.h"
#include "config.h"

namespace gomlx {

struct QwenImageProcessorResult {
    mlx::core::array pixel_values;  // [1, 3, 2, H, W]
    int grid_h, grid_w;             // patch grid dims (H/14, W/14)
    int num_vision_tokens;          // after 2x2 merge: (grid_h/2)*(grid_w/2)
};

class QwenImageProcessor {
public:
    explicit QwenImageProcessor(const Qwen2_5VisionConfig& config);

    QwenImageProcessorResult process_from_file(const std::string& path);
    QwenImageProcessorResult process_from_bytes(const uint8_t* data, int len);

private:
    QwenImageProcessorResult process_pixels(const uint8_t* rgb, int w, int h);
    std::pair<int,int> smart_resize(int w, int h);

    Qwen2_5VisionConfig config_;

    static constexpr int MIN_PIXELS = 3136;     // 56*56
    static constexpr int MAX_PIXELS = 12845056; // ~3584*3584
};

} // namespace gomlx
