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

// EncodeWithError tokenizes text into a slice of token IDs.
// It returns an error if the underlying tokenizer produces no tokens for
// non-empty input, which indicates a tokenization failure.
func (t *Tokenizer) EncodeWithError(text string, addSpecialTokens bool) ([]int32, error) {
	encoding, _ := t.tk.Encode(text, addSpecialTokens)
	if len(encoding) == 0 && len(text) > 0 {
		return nil, fmt.Errorf("tokenizer encode produced no tokens for non-empty input")
	}
	ids := make([]int32, len(encoding))
	for i, id := range encoding {
		ids[i] = int32(id)
	}
	return ids, nil
}

// Encode tokenizes text into a slice of token IDs.
// It silently returns an empty/nil slice if encoding fails.
// Prefer EncodeWithError when error handling is needed.
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) []int32 {
	ids, _ := t.EncodeWithError(text, addSpecialTokens)
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
