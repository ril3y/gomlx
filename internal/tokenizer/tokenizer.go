package tokenizer

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/daulet/tokenizers"
)

// Tokenizer wraps a HuggingFace tokenizer loaded from a model directory.
type Tokenizer struct {
	tk *tokenizers.Tokenizer
}

// New creates a Tokenizer by loading tokenizer.json from the given model directory.
func New(modelPath string) (*Tokenizer, error) {
	tokPath := filepath.Join(modelPath, "tokenizer.json")
	if _, err := os.Stat(tokPath); err != nil {
		return nil, fmt.Errorf("tokenizer not found at %s: %w", tokPath, err)
	}

	tk, err := tokenizers.FromFile(tokPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return &Tokenizer{tk: tk}, nil
}

// Encode tokenizes text into a slice of token IDs.
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) []int32 {
	encoding, _ := t.tk.Encode(text, addSpecialTokens)
	ids := make([]int32, len(encoding))
	for i, id := range encoding {
		ids[i] = int32(id)
	}
	return ids
}

// Decode converts token IDs back into text.
func (t *Tokenizer) Decode(ids []int32, skipSpecialTokens bool) string {
	uids := make([]uint32, len(ids))
	for i, id := range ids {
		uids[i] = uint32(id)
	}
	return t.tk.Decode(uids, skipSpecialTokens)
}

// Close releases the tokenizer resources.
func (t *Tokenizer) Close() {
	if t.tk != nil {
		t.tk.Close()
		t.tk = nil
	}
}
