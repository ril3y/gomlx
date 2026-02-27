package gomlx

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetTemplate_RegisteredArchitectures(t *testing.T) {
	archs := []string{"llama", "llama3", "mllama", "mistral", "qwen2", "qwen2_5_vl", "gemma2"}
	for _, arch := range archs {
		fn := GetTemplate(arch)
		assert.NotNil(t, fn, "template for %q should not be nil", arch)
	}
}

func TestGetTemplate_UnknownFallsBackToLlama(t *testing.T) {
	fn := GetTemplate("unknown_model")
	assert.NotNil(t, fn)
	// Should produce Llama-style output
	result := fn([]Message{{Role: RoleUser, Content: "hi"}}, TemplateContext{})
	assert.Contains(t, result, "<|begin_of_text|>")
}

func TestFormatLlama3(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "You are helpful."},
		{Role: RoleUser, Content: "Hello"},
	}
	result := FormatLlama3(msgs, TemplateContext{})

	assert.Contains(t, result, "<|begin_of_text|>")
	assert.Contains(t, result, "<|start_header_id|>system<|end_header_id|>")
	assert.Contains(t, result, "You are helpful.")
	assert.Contains(t, result, "<|start_header_id|>user<|end_header_id|>")
	assert.Contains(t, result, "Hello")
	assert.True(t, strings.HasSuffix(result, "<|start_header_id|>assistant<|end_header_id|>\n\n"))
}

func TestFormatMllamaVision_WithImage(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "Describe this", Images: []Image{{Path: "test.png"}}},
	}
	result := FormatMllamaVision(msgs, TemplateContext{})

	assert.Contains(t, result, "<|image|>")
	assert.Contains(t, result, "Describe this")
}

func TestFormatMllamaVision_WithoutImage(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "Hello"},
	}
	result := FormatMllamaVision(msgs, TemplateContext{})

	assert.NotContains(t, result, "<|image|>")
	assert.Contains(t, result, "Hello")
}

func TestFormatMistral(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "What is 2+2?"},
	}
	result := FormatMistral(msgs, TemplateContext{})

	assert.True(t, strings.HasPrefix(result, "<s>"))
	assert.Contains(t, result, "[INST]")
	assert.Contains(t, result, "What is 2+2?")
	assert.Contains(t, result, "[/INST]")
}

func TestFormatMistral_MultiTurn(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "Hi"},
		{Role: RoleAssistant, Content: "Hello!"},
		{Role: RoleUser, Content: "How are you?"},
	}
	result := FormatMistral(msgs, TemplateContext{})

	assert.Contains(t, result, "[INST] Hi [/INST]")
	assert.Contains(t, result, "Hello!</s>")
	assert.Contains(t, result, "[INST] How are you? [/INST]")
}

func TestFormatQwen2(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "Hello"},
	}
	result := FormatQwen2(msgs, TemplateContext{})

	assert.Contains(t, result, "<|im_start|>system")
	assert.Contains(t, result, "You are a helpful assistant.")
	assert.Contains(t, result, "<|im_start|>user")
	assert.Contains(t, result, "Hello")
	assert.Contains(t, result, "<|im_end|>")
	assert.True(t, strings.HasSuffix(result, "<|im_start|>assistant\n"))
}

func TestFormatQwen2_CustomSystem(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "You are a pirate."},
		{Role: RoleUser, Content: "Hello"},
	}
	result := FormatQwen2(msgs, TemplateContext{})

	assert.Contains(t, result, "You are a pirate.")
	// Should NOT have default system message
	assert.Equal(t, 1, strings.Count(result, "<|im_start|>system"))
}

func TestFormatQwen25VL_WithVisionTokens(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "Describe", Images: []Image{{Path: "test.png"}}},
	}
	result := FormatQwen25VL(msgs, TemplateContext{VisionTokenCount: 3})

	assert.Contains(t, result, "<|vision_start|>")
	assert.Contains(t, result, "<|vision_end|>")
	assert.Equal(t, 3, strings.Count(result, "<|image_pad|>"))
}

func TestFormatQwen25VL_NoVisionTokens(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "Hello"},
	}
	result := FormatQwen25VL(msgs, TemplateContext{VisionTokenCount: 0})

	assert.NotContains(t, result, "<|vision_start|>")
	assert.NotContains(t, result, "<|image_pad|>")
}

func TestFormatGemma2(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "What is Go?"},
	}
	result := FormatGemma2(msgs, TemplateContext{})

	assert.Contains(t, result, "<start_of_turn>user")
	assert.Contains(t, result, "What is Go?")
	assert.Contains(t, result, "<end_of_turn>")
	assert.True(t, strings.HasSuffix(result, "<start_of_turn>model\n"))
}

func TestFormatGemma2_MultiTurn(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "Hi"},
		{Role: RoleAssistant, Content: "Hello!"},
		{Role: RoleUser, Content: "Bye"},
	}
	result := FormatGemma2(msgs, TemplateContext{})

	assert.Equal(t, 2, strings.Count(result, "<start_of_turn>user"))
	assert.Equal(t, 2, strings.Count(result, "<start_of_turn>model")) // 1 for assistant + 1 for prompt
}

func TestTemplateContext_ZeroValueSafe(t *testing.T) {
	// All templates should handle zero-value TemplateContext without panicking
	msgs := []Message{{Role: RoleUser, Content: "test"}}
	ctx := TemplateContext{}

	assert.NotPanics(t, func() { FormatLlama3(msgs, ctx) })
	assert.NotPanics(t, func() { FormatMllamaVision(msgs, ctx) })
	assert.NotPanics(t, func() { FormatMistral(msgs, ctx) })
	assert.NotPanics(t, func() { FormatQwen2(msgs, ctx) })
	assert.NotPanics(t, func() { FormatQwen25VL(msgs, ctx) })
	assert.NotPanics(t, func() { FormatGemma2(msgs, ctx) })
}
