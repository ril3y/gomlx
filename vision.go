package gomlx

import "fmt"

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
