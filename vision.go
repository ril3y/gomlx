package gomlx

import (
	"context"
	"fmt"
)

// PrepareVisionInput validates and prepares images for a vision model.
// Each image is loaded into the model's vision pipeline via the bridge.
func (m *Model) PrepareVisionInput(images []Image) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.supportsVis {
		return ErrVisionNotSupported
	}
	for _, img := range images {
		if len(img.Data) > 0 {
			if rc := m.bridge.SetImageFromBytes(img.Data); rc < 0 {
				return fmt.Errorf("%w: %s", ErrImageLoadFailed, bridgeError("failed to load image from bytes"))
			}
		} else if img.Path != "" {
			if rc := m.bridge.SetImageFromFile(img.Path); rc < 0 {
				return fmt.Errorf("%w: %s", ErrImageLoadFailed, bridgeError("failed to load image: "+img.Path))
			}
		} else {
			return fmt.Errorf("%w: image has neither path nor data", ErrImageLoadFailed)
		}
	}

	return nil
}

// Point detects the location of the specified object in the image,
// returning normalized [0,1] coordinates. Only supported for Moondream models.
func (m *Model) Point(ctx context.Context, image Image, object string) (*PointResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.supportsVis {
		return nil, ErrVisionNotSupported
	}

	select {
	case <-ctx.Done():
		return nil, ErrContextCanceled
	default:
	}

	// Load the image
	if len(image.Data) > 0 {
		if rc := m.bridge.SetImageFromBytes(image.Data); rc < 0 {
			return nil, fmt.Errorf("%w: %s", ErrImageLoadFailed, bridgeError("failed to load image from bytes"))
		}
	} else if image.Path != "" {
		if rc := m.bridge.SetImageFromFile(image.Path); rc < 0 {
			return nil, fmt.Errorf("%w: %s", ErrImageLoadFailed, bridgeError("failed to load image: "+image.Path))
		}
	} else {
		return nil, fmt.Errorf("%w: image has neither path nor data", ErrImageLoadFailed)
	}

	// Build point detection tokens
	tokens := MoondreamPointTokens(object, m.tokenizer.Encode)

	// Call bridge
	coords, count := m.bridge.DetectPoints(tokens, 50)
	if count < 0 {
		return nil, fmt.Errorf("%w: %s", ErrInferenceFailed, bridgeError("point detection failed"))
	}

	// Convert to typed result
	result := &PointResult{Points: make([]Point, count)}
	for i := 0; i < count; i++ {
		result.Points[i] = Point{
			X: float64(coords[i*2]),
			Y: float64(coords[i*2+1]),
		}
	}

	return result, nil
}

// Detect finds all instances of the specified object in the image,
// returning normalized [0,1] bounding boxes. Only supported for Moondream models.
func (m *Model) Detect(ctx context.Context, image Image, object string) (*DetectResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.supportsVis {
		return nil, ErrVisionNotSupported
	}

	select {
	case <-ctx.Done():
		return nil, ErrContextCanceled
	default:
	}

	// Load the image
	if len(image.Data) > 0 {
		if rc := m.bridge.SetImageFromBytes(image.Data); rc < 0 {
			return nil, fmt.Errorf("%w: %s", ErrImageLoadFailed, bridgeError("failed to load image from bytes"))
		}
	} else if image.Path != "" {
		if rc := m.bridge.SetImageFromFile(image.Path); rc < 0 {
			return nil, fmt.Errorf("%w: %s", ErrImageLoadFailed, bridgeError("failed to load image: "+image.Path))
		}
	} else {
		return nil, fmt.Errorf("%w: image has neither path nor data", ErrImageLoadFailed)
	}

	// Build detect tokens
	tokens := MoondreamDetectTokens(object, m.tokenizer.Encode)

	// Call bridge
	boxes, count := m.bridge.DetectObjects(tokens, 50)
	if count < 0 {
		return nil, fmt.Errorf("%w: %s", ErrInferenceFailed, bridgeError("object detection failed"))
	}

	// Convert to typed result
	result := &DetectResult{Objects: make([]BoundingBox, count)}
	for i := 0; i < count; i++ {
		result.Objects[i] = BoundingBox{
			XMin: float64(boxes[i*4]),
			YMin: float64(boxes[i*4+1]),
			XMax: float64(boxes[i*4+2]),
			YMax: float64(boxes[i*4+3]),
		}
	}

	return result, nil
}
