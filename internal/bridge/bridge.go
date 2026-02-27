package bridge

/*
#cgo CFLAGS: -I${SRCDIR}/../../mlxlib
#cgo LDFLAGS: -L${SRCDIR}/../../build -lmlxbridge -L${SRCDIR}/../../build/mlx_build -lmlx -L${SRCDIR}/../../lib -ltokenizers -framework Metal -framework Foundation -framework Accelerate -framework MetalPerformanceShaders -lc++

#include "mlx_bridge.h"
#include <stdlib.h>

// Forward declaration of the Go-exported callback.
extern void goProgressCallback(float progress, const char* message, void* user_data);

// goProgressTrampoline wraps the Go callback as a C function pointer.
void goProgressTrampoline(float progress, const char* message, void* user_data) {
    goProgressCallback(progress, message, user_data);
}
*/
import "C"
import (
	"unsafe"
)

// BridgeModel wraps a pointer to the C MLXModel struct.
type BridgeModel struct {
	ptr *C.MLXModel
}

// CreateModel loads an MLX model from the given path.
// The callbackHandle is used to route C progress callbacks to Go functions.
// Returns nil if the C layer returns NULL (check LastError for details).
func CreateModel(modelPath string, callbackHandle uintptr) *BridgeModel {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	var fn C.mlx_progress_fn
	var userData unsafe.Pointer
	if callbackHandle != 0 {
		fn = C.mlx_progress_fn(C.goProgressTrampoline)
		userData = unsafe.Pointer(callbackHandle)
	}

	ptr := C.mlx_model_create(cPath, fn, userData)
	if ptr == nil {
		return nil
	}
	return &BridgeModel{ptr: ptr}
}

// Free releases the C model resources.
func (m *BridgeModel) Free() {
	if m.ptr != nil {
		C.mlx_model_free(m.ptr)
		m.ptr = nil
	}
}

// VocabSize returns the vocabulary size of the loaded model.
func (m *BridgeModel) VocabSize() int {
	return int(C.mlx_model_vocab_size(m.ptr))
}

// Architecture returns the model architecture string (e.g., "llama").
func (m *BridgeModel) Architecture() string {
	cStr := C.mlx_model_architecture(m.ptr)
	return C.GoString(cStr)
}

// SupportsVision returns true if the model supports vision inputs.
func (m *BridgeModel) SupportsVision() bool {
	return C.mlx_model_supports_vision(m.ptr) != 0
}

// Prefill runs the prefill phase with the given token IDs.
// Returns the first predicted token ID, or -1 on error.
func (m *BridgeModel) Prefill(tokens []int32) int {
	if len(tokens) == 0 {
		return -1
	}
	cTokens := (*C.int)(unsafe.Pointer(&tokens[0]))
	return int(C.mlx_model_prefill(m.ptr, cTokens, C.int(len(tokens))))
}

// NextToken generates the next token given the current KV cache state.
// Returns the token ID, or -1 on error.
func (m *BridgeModel) NextToken(temperature, topP float32) int {
	return int(C.mlx_model_next_token(m.ptr, C.float(temperature), C.float(topP)))
}

// Reset clears the model's KV cache state for a new conversation.
func (m *BridgeModel) Reset() {
	C.mlx_model_reset(m.ptr)
}

// GetActiveMemory returns the current active MLX memory usage in bytes.
func GetActiveMemory() uint64 {
	return uint64(C.mlx_get_active_memory())
}

// GetPeakMemory returns the peak MLX memory usage in bytes.
func GetPeakMemory() uint64 {
	return uint64(C.mlx_get_peak_memory())
}

// GetStopTokens returns the model's stop token IDs.
func (m *BridgeModel) GetStopTokens() []int32 {
	var buf [16]C.int
	count := int(C.mlx_model_get_stop_tokens(m.ptr, &buf[0], 16))
	result := make([]int32, count)
	for i := 0; i < count; i++ {
		result[i] = int32(buf[i])
	}
	return result
}

// LastError returns the last error message from the C bridge, or empty string if none.
func LastError() string {
	cStr := C.mlx_last_error()
	if cStr == nil {
		return ""
	}
	return C.GoString(cStr)
}
