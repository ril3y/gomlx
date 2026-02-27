#pragma once

#include <string>
#include <vector>
#include <utility>

#include "mlx/mlx.h"
#include "config.h"

namespace gomlx {

struct ImageProcessorResult {
    mlx::core::array pixel_values;  // [num_tiles, 3, tile_h, tile_w]
    int aspect_ratio_id;            // 1-indexed into supported ratios
    int num_tiles;
};

class ImageProcessor {
public:
    explicit ImageProcessor(const VisionModelConfig& config);

    ImageProcessorResult process_from_file(const std::string& path);
    ImageProcessorResult process_from_bytes(const uint8_t* data, int len);

private:
    ImageProcessorResult process_pixels(const uint8_t* rgb, int w, int h);
    std::pair<int,int> get_optimal_tiled_canvas(int w, int h);

    VisionModelConfig config_;
};

} // namespace gomlx
