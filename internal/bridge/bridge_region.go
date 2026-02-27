package bridge

/*
#include "mlx_bridge.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// DetectPoints runs point detection, returning flat [x0,y0, x1,y1, ...] coords and count.
// Returns count=-1 on error.
func (m *BridgeModel) DetectPoints(tokens []int32, maxObjects int) ([]float32, int) {
	if len(tokens) == 0 || maxObjects <= 0 {
		return nil, -1
	}
	cTokens := (*C.int)(unsafe.Pointer(&tokens[0]))
	outCoords := make([]float32, maxObjects*2)
	cCoords := (*C.float)(unsafe.Pointer(&outCoords[0]))

	count := int(C.mlx_model_detect_points(m.ptr, cTokens, C.int(len(tokens)), cCoords, C.int(maxObjects)))
	if count < 0 {
		return nil, count
	}
	return outCoords[:count*2], count
}

// DetectObjects runs object detection, returning flat [x_min,y_min,x_max,y_max, ...] boxes and count.
// Returns count=-1 on error.
func (m *BridgeModel) DetectObjects(tokens []int32, maxObjects int) ([]float32, int) {
	if len(tokens) == 0 || maxObjects <= 0 {
		return nil, -1
	}
	cTokens := (*C.int)(unsafe.Pointer(&tokens[0]))
	outBoxes := make([]float32, maxObjects*4)
	cBoxes := (*C.float)(unsafe.Pointer(&outBoxes[0]))

	count := int(C.mlx_model_detect_objects(m.ptr, cTokens, C.int(len(tokens)), cBoxes, C.int(maxObjects)))
	if count < 0 {
		return nil, count
	}
	return outBoxes[:count*4], count
}
