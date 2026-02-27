package gomlx

import (
	"context"
	"fmt"
	"strings"
	"time"
)

const (
	defaultMaxTokens   = 2048
	defaultTemperature = 0.7
	defaultTopP        = 0.9
)

// hasImages returns true if any message contains images.
func hasImages(messages []Message) bool {
	for _, msg := range messages {
		if len(msg.Images) > 0 {
			return true
		}
	}
	return false
}

// applyDefaults fills in default values for GenerateInput fields.
func (m *Model) applyDefaults(input *GenerateInput) {
	if input.MaxTokens <= 0 {
		if m.opts.maxTokens > 0 {
			input.MaxTokens = m.opts.maxTokens
		} else {
			input.MaxTokens = defaultMaxTokens
		}
	}
	if input.Temperature <= 0 {
		input.Temperature = defaultTemperature
	}
	if input.TopP <= 0 {
		input.TopP = defaultTopP
	}
}

// Generate produces a complete text response for the given input.
func (m *Model) Generate(ctx context.Context, input GenerateInput) (*GenerateOutput, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.applyDefaults(&input)

	// Reset model state for new generation
	m.bridge.Reset()

	// For vision models, load images BEFORE templating so the chat template
	// can insert the correct number of image placeholder tokens.
	isVision := hasImages(input.Messages) && m.supportsVis
	if isVision {
		for _, msg := range input.Messages {
			for _, img := range msg.Images {
				var rc int
				if len(img.Data) > 0 {
					rc = m.bridge.SetImageFromBytes(img.Data)
				} else if img.Path != "" {
					rc = m.bridge.SetImageFromFile(img.Path)
				}
				if rc < 0 {
					return nil, fmt.Errorf("%w: %s", ErrImageLoadFailed, bridgeError("failed to set image"))
				}
			}
		}
	}

	// Get vision token count (0 for non-vision models/paths)
	visionTokenCount := 0
	if isVision {
		visionTokenCount = m.bridge.GetVisionTokenCount()
	}

	// Format messages using the appropriate chat template
	templateFn := GetTemplate(m.architecture)
	prompt := templateFn(input.Messages, TemplateContext{VisionTokenCount: visionTokenCount})

	// Tokenize the prompt (don't add special tokens - template already has them)
	tokens := m.tokenizer.Encode(prompt, false)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("%w: empty prompt after tokenization", ErrInferenceFailed)
	}

	start := time.Now()

	// Prefill with prompt tokens
	firstToken := m.bridge.Prefill(tokens)
	if firstToken < 0 {
		return nil, fmt.Errorf("%w: %s", ErrInferenceFailed, bridgeError("prefill failed"))
	}

	ttft := time.Since(start)
	var generated []int32
	currentToken := int32(firstToken)

	for len(generated) < input.MaxTokens {
		// Check for context cancellation
		select {
		case <-ctx.Done():
			return nil, ErrContextCanceled
		default:
		}

		// Check for stop tokens
		if m.isStopToken(currentToken) {
			break
		}

		generated = append(generated, currentToken)

		// Check stop strings
		if len(input.Stop) > 0 {
			partial := m.tokenizer.Decode(generated, true)
			shouldStop := false
			for _, stop := range input.Stop {
				if strings.Contains(partial, stop) {
					// Trim the stop string from output
					partial = strings.SplitN(partial, stop, 2)[0]
					generated = m.tokenizer.Encode(partial, false)
					shouldStop = true
					break
				}
			}
			if shouldStop {
				break
			}
		}

		// Generate next token
		nextToken := m.bridge.NextToken(input.Temperature, input.TopP)
		if nextToken < 0 {
			// Return partial output on error instead of failing completely
			elapsed := time.Since(start)
			content := m.tokenizer.Decode(generated, true)
			tokenCount := len(generated)
			tokPerSec := 0.0
			if elapsed > 0 {
				tokPerSec = float64(tokenCount) / elapsed.Seconds()
			}
			return &GenerateOutput{
				Content:            content,
				TokenCount:         tokenCount,
				TokensPerSecond:    tokPerSec,
				TimeToFirstTokenMs: float64(ttft.Milliseconds()),
				TotalTimeMs:        float64(elapsed.Milliseconds()),
			}, fmt.Errorf("%w: %s (partial output returned)", ErrInferenceFailed, bridgeError("next_token failed"))
		}
		currentToken = int32(nextToken)
	}

	elapsed := time.Since(start)
	content := m.tokenizer.Decode(generated, true)
	tokenCount := len(generated)
	tokPerSec := 0.0
	if elapsed > 0 {
		tokPerSec = float64(tokenCount) / elapsed.Seconds()
	}

	return &GenerateOutput{
		Content:            content,
		TokenCount:         tokenCount,
		TokensPerSecond:    tokPerSec,
		TimeToFirstTokenMs: float64(ttft.Milliseconds()),
		TotalTimeMs:        float64(elapsed.Milliseconds()),
	}, nil
}

// GenerateStream produces a streaming text response, calling the callback
// with each new text fragment as it is generated.
func (m *Model) GenerateStream(ctx context.Context, input GenerateInput, callback func(string)) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.applyDefaults(&input)

	// Reset model state for new generation
	m.bridge.Reset()

	// For vision models, load images BEFORE templating so the chat template
	// can insert the correct number of image placeholder tokens.
	isVision := hasImages(input.Messages) && m.supportsVis
	if isVision {
		for _, msg := range input.Messages {
			for _, img := range msg.Images {
				var rc int
				if len(img.Data) > 0 {
					rc = m.bridge.SetImageFromBytes(img.Data)
				} else if img.Path != "" {
					rc = m.bridge.SetImageFromFile(img.Path)
				}
				if rc < 0 {
					return fmt.Errorf("%w: %s", ErrImageLoadFailed, bridgeError("failed to set image"))
				}
			}
		}
	}

	// Get vision token count (0 for non-vision models/paths)
	visionTokenCount := 0
	if isVision {
		visionTokenCount = m.bridge.GetVisionTokenCount()
	}

	// Format messages using the appropriate chat template
	templateFn := GetTemplate(m.architecture)
	prompt := templateFn(input.Messages, TemplateContext{VisionTokenCount: visionTokenCount})

	// Tokenize the prompt
	tokens := m.tokenizer.Encode(prompt, false)
	if len(tokens) == 0 {
		return fmt.Errorf("%w: empty prompt after tokenization", ErrInferenceFailed)
	}

	// Prefill with prompt tokens
	firstToken := m.bridge.Prefill(tokens)
	if firstToken < 0 {
		return fmt.Errorf("%w: %s", ErrInferenceFailed, bridgeError("prefill failed"))
	}

	var generated []int32
	var prevText string
	currentToken := int32(firstToken)

	for len(generated) < input.MaxTokens {
		select {
		case <-ctx.Done():
			return ErrContextCanceled
		default:
		}

		if m.isStopToken(currentToken) {
			break
		}

		generated = append(generated, currentToken)

		// Decode all tokens so far and emit the new delta
		fullText := m.tokenizer.Decode(generated, true)

		// Check stop strings
		if len(input.Stop) > 0 {
			shouldStop := false
			for _, stop := range input.Stop {
				if strings.Contains(fullText, stop) {
					// Emit up to the stop string
					trimmed := strings.SplitN(fullText, stop, 2)[0]
					if len(trimmed) > len(prevText) {
						callback(trimmed[len(prevText):])
					}
					shouldStop = true
					break
				}
			}
			if shouldStop {
				break
			}
		}

		if len(fullText) > len(prevText) {
			delta := fullText[len(prevText):]
			callback(delta)
			prevText = fullText
		}

		nextToken := m.bridge.NextToken(input.Temperature, input.TopP)
		if nextToken < 0 {
			// Flush any remaining partial text before returning the error
			if len(generated) > 0 {
				fullText := m.tokenizer.Decode(generated, true)
				if len(fullText) > len(prevText) {
					callback(fullText[len(prevText):])
				}
			}
			return fmt.Errorf("%w: %s (partial output returned)", ErrInferenceFailed, bridgeError("next_token failed"))
		}
		currentToken = int32(nextToken)
	}

	return nil
}
