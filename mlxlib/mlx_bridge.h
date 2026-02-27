#ifndef MLX_BRIDGE_H
#define MLX_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLXModel MLXModel;
typedef void (*mlx_progress_fn)(float progress, const char* message, void* user_data);

// --- Lifecycle ---

// Create a model from a directory containing config.json and safetensors weights.
// The progress callback is called during loading (can be NULL).
// Returns NULL on error (check mlx_last_error()).
MLXModel* mlx_model_create(const char* model_path, mlx_progress_fn fn, void* user_data);

// Free a model and all associated resources.
void mlx_model_free(MLXModel* m);

// --- Model Info ---

// Return the vocabulary size.
int mlx_model_vocab_size(MLXModel* m);

// Return the model architecture string (e.g. "llama"). Valid until model is freed.
const char* mlx_model_architecture(MLXModel* m);

// Return 1 if the model supports vision inputs, 0 otherwise.
int mlx_model_supports_vision(MLXModel* m);

// --- Text Generation ---

// Process a prompt: run forward pass on the given tokens.
// Returns the first predicted token id, or -1 on error.
int mlx_model_prefill(MLXModel* m, const int* tokens, int n_tokens);

// Generate the next token given previous state.
// Uses temperature and top_p for sampling.
// Returns the predicted token id, or -1 on error.
int mlx_model_next_token(MLXModel* m, float temperature, float top_p);

// Reset generation state (clears KV cache).
void mlx_model_reset(MLXModel* m);

// --- Vision (stubs for Phase 1) ---

// Set an image from file for vision models. Returns 0 on success, -1 on error.
int mlx_model_set_image_from_file(MLXModel* m, const char* image_path);

// Set an image from raw bytes for vision models. Returns 0 on success, -1 on error.
int mlx_model_set_image_from_bytes(MLXModel* m, const uint8_t* data, int len);

// Process a multimodal prompt with vision. Returns first token or -1 on error.
int mlx_model_prefill_vision(MLXModel* m, const int* tokens, int n_tokens);

// Get the number of vision tokens for the current pending image.
// Returns 0 if no image is pending or not a vision model.
int mlx_model_get_vision_token_count(MLXModel* m);

// Get the model's stop token IDs. Writes up to max_tokens IDs into out_tokens.
// Returns the number of stop tokens written.
int mlx_model_get_stop_tokens(MLXModel* m, int* out_tokens, int max_tokens);

// --- Memory & Error ---

// Get current active memory usage in bytes.
size_t mlx_get_active_memory(void);

// Get peak memory usage in bytes.
size_t mlx_get_peak_memory(void);

// Get the last error message, or NULL if no error.
// The returned string is valid until the next call from the same thread.
const char* mlx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // MLX_BRIDGE_H
