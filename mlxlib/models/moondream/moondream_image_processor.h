#pragma once

#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace gomlx {

struct MoondreamImageProcessorResult {
    mlx::core::array pixel_values;  // [1, 3, 378, 378]
};

class MoondreamImageProcessor {
public:
    MoondreamImageProcessor() = default;

    MoondreamImageProcessorResult process_from_file(const std::string& path);
    MoondreamImageProcessorResult process_from_bytes(const uint8_t* data, int len);

private:
    MoondreamImageProcessorResult process_pixels(const uint8_t* rgb, int w, int h);

    static constexpr int IMAGE_SIZE = 378;
    static constexpr float IMAGE_MEAN = 0.5f;
    static constexpr float IMAGE_STD = 0.5f;
};

} // namespace gomlx
