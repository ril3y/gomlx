package bridge

/*
#include "mlx_bridge.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// SetImageFromFile loads an image from a file path for vision processing.
// Returns 0 on success, -1 on error.
func (m *BridgeModel) SetImageFromFile(imagePath string) int {
	cPath := C.CString(imagePath)
	defer C.free(unsafe.Pointer(cPath))
	return int(C.mlx_model_set_image_from_file(m.ptr, cPath))
}

// SetImageFromBytes loads an image from raw bytes for vision processing.
// Returns 0 on success, -1 on error.
func (m *BridgeModel) SetImageFromBytes(data []byte) int {
	if len(data) == 0 {
		return -1
	}
	cData := (*C.uint8_t)(unsafe.Pointer(&data[0]))
	return int(C.mlx_model_set_image_from_bytes(m.ptr, cData, C.int(len(data))))
}

// GetVisionTokenCount returns the number of vision tokens for the current pending image.
// Returns 0 if no image is pending or model doesn't support it.
func (m *BridgeModel) GetVisionTokenCount() int {
	return int(C.mlx_model_get_vision_token_count(m.ptr))
}
