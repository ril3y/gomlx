package gomlx

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/ril3y/gomlx/internal/bridge"
	"github.com/ril3y/gomlx/internal/tokenizer"
)

// Model represents a loaded MLX language model ready for inference.
type Model struct {
	mu           sync.Mutex
	bridge       *bridge.BridgeModel
	tokenizer    *tokenizer.Tokenizer
	architecture string
	supportsVis  bool
	vocabSize    int
	stopTokens   map[int32]struct{}
	opts         modelOptions
}

// LoadModel loads an MLX model from the given directory path.
func LoadModel(ctx context.Context, modelPath string, opts ...ModelOption) (*Model, error) {
	// Verify model path exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("%w: %s", ErrModelNotFound, modelPath)
	}

	// Apply options
	var options modelOptions
	for _, opt := range opts {
		opt(&options)
	}

	// Check context before starting
	select {
	case <-ctx.Done():
		return nil, ErrContextCanceled
	default:
	}

	// Register progress callback if provided
	var callbackHandle uintptr
	if options.progressFn != nil {
		callbackHandle = bridge.RegisterCallback(func(progress float32, message string) {
			options.progressFn(progress, message)
		})
		defer bridge.UnregisterCallback(callbackHandle)
	}

	// Create the C bridge model
	bm := bridge.CreateModel(modelPath, callbackHandle)
	if bm == nil {
		return nil, fmt.Errorf("%w: %s", ErrModelLoadFailed, bridgeError("failed to create model"))
	}

	// Load tokenizer
	tok, err := tokenizer.New(modelPath)
	if err != nil {
		bm.Free()
		return nil, fmt.Errorf("%w: %v", ErrModelLoadFailed, err)
	}

	m := &Model{
		bridge:       bm,
		tokenizer:    tok,
		architecture: bm.Architecture(),
		supportsVis:  bm.SupportsVision(),
		vocabSize:    bm.VocabSize(),
		opts:         options,
	}

	// Cache stop tokens from the model
	stopToks := bm.GetStopTokens()
	m.stopTokens = make(map[int32]struct{}, len(stopToks))
	for _, id := range stopToks {
		m.stopTokens[id] = struct{}{}
	}

	return m, nil
}

// Close releases all resources associated with the model.
// It is safe to call Close concurrently with other Model methods.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.tokenizer != nil {
		m.tokenizer.Close()
		m.tokenizer = nil
	}
	if m.bridge != nil {
		m.bridge.Free()
		m.bridge = nil
	}
	return nil
}

// Architecture returns the model architecture name (e.g., "llama").
func (m *Model) Architecture() string {
	return m.architecture
}

// SupportsVision returns true if the model supports image inputs.
func (m *Model) SupportsVision() bool {
	return m.supportsVis
}

// VocabSize returns the number of tokens in the model's vocabulary.
func (m *Model) VocabSize() int {
	return m.vocabSize
}

// isStopToken checks if a token ID is a stop token for this model.
func (m *Model) isStopToken(id int32) bool {
	_, ok := m.stopTokens[id]
	return ok
}
