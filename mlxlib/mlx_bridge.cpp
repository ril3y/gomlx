#include "mlx_bridge.h"
#include "base_model.h"
#include "config.h"
#include "model_registry.h"
#include "sampling.h"
#include "models/moondream/moondream.h"

#include "mlx/mlx.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace mlx::core;

// Thread-local error string
static thread_local std::string g_last_error;

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

static void clear_error() {
    g_last_error.clear();
}

struct MLXModel {
    std::unique_ptr<gomlx::BaseModel> model;
    gomlx::ModelConfig config;
    std::string architecture;
    std::optional<array> last_logits;
    int last_token = -1; // last sampled token, used for autoregressive decode
};

extern "C" {

MLXModel* mlx_model_create(const char* model_path, mlx_progress_fn fn, void* user_data) {
    clear_error();
    try {
        auto m = new MLXModel();

        if (fn) fn(0.0f, "Parsing config...", user_data);
        m->config = gomlx::parse_config(model_path);
        m->architecture = m->config.model_type;

        if (fn) fn(0.1f, "Creating model...", user_data);
        m->model = gomlx::create_model(m->config);

        if (fn) fn(0.2f, "Loading weights...", user_data);
        m->model->load_weights(model_path);

        if (fn) fn(1.0f, "Model loaded.", user_data);
        return m;
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

void mlx_model_free(MLXModel* m) {
    delete m;
}

int mlx_model_vocab_size(MLXModel* m) {
    if (!m || !m->model) return 0;
    return m->model->vocab_size();
}

const char* mlx_model_architecture(MLXModel* m) {
    if (!m) return "";
    return m->architecture.c_str();
}

int mlx_model_supports_vision(MLXModel* m) {
    if (!m || !m->model) return 0;
    return m->model->supports_vision() ? 1 : 0;
}

int mlx_model_prefill(MLXModel* m, const int* tokens, int n_tokens) {
    clear_error();
    if (!m || !m->model || !tokens || n_tokens <= 0) {
        set_error("Invalid arguments to mlx_model_prefill");
        return -1;
    }

    try {
        // Create token array: [1, n_tokens]
        std::vector<int> token_vec(tokens, tokens + n_tokens);
        array token_arr(token_vec.data(), {1, n_tokens}, int32);

        // Forward pass (offset=0 since cache is empty at prefill time)
        auto logits = m->model->forward(token_arr, 0);
        eval(logits);
        m->last_logits = logits;

        // Sample first generated token (greedy for prefill)
        int token = gomlx::sample_token(logits, 0.0f, 1.0f);
        m->last_token = token;
        return token;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    }
}

int mlx_model_next_token(MLXModel* m, float temperature, float top_p) {
    clear_error();
    if (!m || !m->model) {
        set_error("Invalid model in mlx_model_next_token");
        return -1;
    }

    try {
        if (m->last_token < 0) {
            set_error("Must call mlx_model_prefill before mlx_model_next_token");
            return -1;
        }

        // Run forward with the previously generated token
        // The KV cache tracks sequence position internally via rope_offset
        std::vector<int> tv = {m->last_token};
        array token_arr(tv.data(), {1, 1}, int32);
        auto logits = m->model->forward(token_arr, 0);
        eval(logits);
        m->last_logits = logits;

        // Sample next token from the new logits
        int token = gomlx::sample_token(logits, temperature, top_p);
        m->last_token = token;
        return token;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    }
}

void mlx_model_reset(MLXModel* m) {
    if (m && m->model) {
        m->model->reset_cache();
        m->last_logits = std::nullopt;
        m->last_token = -1;
    }
}

int mlx_model_set_image_from_file(MLXModel* m, const char* image_path) {
    clear_error();
    if (!m || !m->model || !image_path) {
        set_error("Invalid arguments to mlx_model_set_image_from_file");
        return -1;
    }
    try {
        m->model->set_image_from_file(image_path);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    }
}

int mlx_model_set_image_from_bytes(MLXModel* m, const uint8_t* data, int len) {
    clear_error();
    if (!m || !m->model || !data || len <= 0) {
        set_error("Invalid arguments to mlx_model_set_image_from_bytes");
        return -1;
    }
    try {
        m->model->set_image_from_bytes(data, len);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    }
}

int mlx_model_prefill_vision(MLXModel* m, const int* tokens, int n_tokens) {
    return mlx_model_prefill(m, tokens, n_tokens);
}

int mlx_model_get_vision_token_count(MLXModel* m) {
    if (!m || !m->model) return 0;
    return m->model->pending_vision_token_count();
}

size_t mlx_get_active_memory(void) {
    return mlx::core::get_active_memory();
}

size_t mlx_get_peak_memory(void) {
    return mlx::core::get_peak_memory();
}

const char* mlx_last_error(void) {
    if (g_last_error.empty()) return nullptr;
    return g_last_error.c_str();
}

int mlx_model_get_stop_tokens(MLXModel* m, int* out_tokens, int max_tokens) {
    if (!m || !m->model || !out_tokens || max_tokens <= 0) return 0;
    auto ids = m->model->stop_token_ids();
    int count = std::min((int)ids.size(), max_tokens);
    for (int i = 0; i < count; i++) {
        out_tokens[i] = ids[i];
    }
    return count;
}

int mlx_model_detect_points(MLXModel* m, const int* tokens, int n_tokens, float* out_coords, int max_objects) {
    clear_error();
    if (!m || !m->model || !tokens || n_tokens <= 0 || !out_coords || max_objects <= 0) {
        set_error("Invalid arguments to mlx_model_detect_points");
        return -1;
    }
    try {
        auto* moondream = dynamic_cast<gomlx::MoondreamModel*>(m->model.get());
        if (!moondream) {
            set_error("detect_points is only supported for Moondream models");
            return -1;
        }
        // Reset state for fresh detection
        m->model->reset_cache();
        m->last_logits = std::nullopt;
        m->last_token = -1;

        std::vector<int> token_vec(tokens, tokens + n_tokens);
        auto points = moondream->detect_points(token_vec, max_objects);

        int count = static_cast<int>(points.size());
        for (int i = 0; i < count; i++) {
            out_coords[i * 2] = points[i].first;
            out_coords[i * 2 + 1] = points[i].second;
        }
        return count;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    }
}

int mlx_model_detect_objects(MLXModel* m, const int* tokens, int n_tokens, float* out_boxes, int max_objects) {
    clear_error();
    if (!m || !m->model || !tokens || n_tokens <= 0 || !out_boxes || max_objects <= 0) {
        set_error("Invalid arguments to mlx_model_detect_objects");
        return -1;
    }
    try {
        auto* moondream = dynamic_cast<gomlx::MoondreamModel*>(m->model.get());
        if (!moondream) {
            set_error("detect_objects is only supported for Moondream models");
            return -1;
        }
        // Reset state for fresh detection
        m->model->reset_cache();
        m->last_logits = std::nullopt;
        m->last_token = -1;

        std::vector<int> token_vec(tokens, tokens + n_tokens);
        auto boxes = moondream->detect_objects(token_vec, max_objects);

        int count = static_cast<int>(boxes.size());
        for (int i = 0; i < count; i++) {
            out_boxes[i * 4] = boxes[i][0];
            out_boxes[i * 4 + 1] = boxes[i][1];
            out_boxes[i * 4 + 2] = boxes[i][2];
            out_boxes[i * 4 + 3] = boxes[i][3];
        }
        return count;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    }
}

} // extern "C"
