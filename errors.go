package gomlx

import (
	"errors"
	"fmt"

	"github.com/ril3y/gomlx/internal/bridge"
)

var (
	// ErrModelNotFound indicates the model path does not exist or is not a valid model directory.
	ErrModelNotFound = errors.New("gomlx: model not found")

	// ErrModelLoadFailed indicates the model could not be loaded.
	ErrModelLoadFailed = errors.New("gomlx: model load failed")

	// ErrInferenceFailed indicates an error during token generation.
	ErrInferenceFailed = errors.New("gomlx: inference failed")

	// ErrVisionNotSupported indicates the model does not support vision inputs.
	ErrVisionNotSupported = errors.New("gomlx: vision not supported by this model")

	// ErrImageLoadFailed indicates an image could not be loaded for vision processing.
	ErrImageLoadFailed = errors.New("gomlx: image load failed")

	// ErrContextCanceled indicates the operation was canceled via context.
	ErrContextCanceled = errors.New("gomlx: context canceled")
)

// bridgeError wraps an error message with the last error from the C bridge.
func bridgeError(msg string) error {
	lastErr := bridge.LastError()
	if lastErr != "" {
		return fmt.Errorf("%s: %s", msg, lastErr)
	}
	return fmt.Errorf("%s", msg)
}
