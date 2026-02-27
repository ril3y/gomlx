package gomlx

import "strings"

// TemplateContext provides model-specific context to chat template functions.
type TemplateContext struct {
	VisionTokenCount int // 0 if no image or model doesn't need placeholders
}

// ChatTemplateFunc formats a slice of messages into a prompt string.
type ChatTemplateFunc func(messages []Message, ctx TemplateContext) string

// templateRegistry maps architecture names to their chat template functions.
var templateRegistry = map[string]ChatTemplateFunc{
	"llama":      FormatLlama3,
	"llama3":     FormatLlama3,
	"mllama":     FormatMllamaVision,
	"mistral":    FormatMistral,
	"qwen2":      FormatQwen2,
	"qwen2_5_vl": FormatQwen25VL,
	"gemma2":     FormatGemma2,
	"moondream1": FormatMoondream,
}

// GetTemplate returns the chat template function for the given architecture.
// Falls back to Llama 3 format if the architecture is not recognized.
func GetTemplate(arch string) ChatTemplateFunc {
	arch = strings.ToLower(arch)
	if fn, ok := templateRegistry[arch]; ok {
		return fn
	}
	// Default to Llama 3 format
	return FormatLlama3
}

// FormatMllamaVision formats messages using the Llama 3.2 Vision (mllama) chat template.
// When a user message has images, an <|image|> token is inserted before the text content.
func FormatMllamaVision(messages []Message, ctx TemplateContext) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|>")

	for _, msg := range messages {
		b.WriteString("<|start_header_id|>")
		b.WriteString(string(msg.Role))
		b.WriteString("<|end_header_id|>\n\n")
		if msg.Role == RoleUser && len(msg.Images) > 0 {
			b.WriteString("<|image|>")
		}
		b.WriteString(msg.Content)
		b.WriteString("<|eot_id|>")
	}

	b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return b.String()
}

// FormatQwen25VL formats messages using the Qwen2.5-VL chat template.
//
// Format:
//
//	<|im_start|>system
//	You are a helpful assistant.<|im_end|>
//	<|im_start|>user
//	<|vision_start|><|image_pad|>...<|vision_end|>{user_content}<|im_end|>
//	<|im_start|>assistant
func FormatQwen25VL(messages []Message, ctx TemplateContext) string {
	var b strings.Builder

	// Add default system message if none provided
	hasSystem := false
	for _, msg := range messages {
		if msg.Role == RoleSystem {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		b.WriteString("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
	}

	for _, msg := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(string(msg.Role))
		b.WriteString("\n")
		if msg.Role == RoleUser && len(msg.Images) > 0 && ctx.VisionTokenCount > 0 {
			b.WriteString("<|vision_start|>")
			for i := 0; i < ctx.VisionTokenCount; i++ {
				b.WriteString("<|image_pad|>")
			}
			b.WriteString("<|vision_end|>")
		}
		b.WriteString(msg.Content)
		b.WriteString("<|im_end|>\n")
	}

	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

// FormatLlama3 formats messages using the Llama 3 Instruct chat template.
//
// Format:
//
//	<|begin_of_text|><|start_header_id|>system<|end_header_id|>
//
//	{system_message}<|eot_id|>
//	<|start_header_id|>user<|end_header_id|>
//
//	{user_message}<|eot_id|>
//	<|start_header_id|>assistant<|end_header_id|>
func FormatLlama3(messages []Message, ctx TemplateContext) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|>")

	for _, msg := range messages {
		b.WriteString("<|start_header_id|>")
		b.WriteString(string(msg.Role))
		b.WriteString("<|end_header_id|>\n\n")
		b.WriteString(msg.Content)
		b.WriteString("<|eot_id|>")
	}

	// Add the assistant header to prompt generation
	b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return b.String()
}

// FormatMistral formats messages using the Mistral Instruct chat template.
//
// Format:
//
//	<s>[INST] {user_message} [/INST]
func FormatMistral(messages []Message, ctx TemplateContext) string {
	var b strings.Builder
	b.WriteString("<s>")

	for i, msg := range messages {
		switch msg.Role {
		case RoleSystem:
			// System message is prepended to the first user message
			b.WriteString("[INST] ")
			b.WriteString(msg.Content)
			b.WriteString("\n\n")
		case RoleUser:
			if i > 0 || msg.Role == RoleUser {
				b.WriteString("[INST] ")
			}
			b.WriteString(msg.Content)
			b.WriteString(" [/INST]")
		case RoleAssistant:
			b.WriteString(msg.Content)
			b.WriteString("</s>")
		}
	}

	return b.String()
}

// FormatQwen2 formats messages using the Qwen2 chat template.
// Same format as Qwen2.5-VL but without vision token support.
//
// Format:
//
//	<|im_start|>system
//	You are a helpful assistant.<|im_end|>
//	<|im_start|>user
//	{user_message}<|im_end|>
//	<|im_start|>assistant
func FormatQwen2(messages []Message, ctx TemplateContext) string {
	var b strings.Builder

	hasSystem := false
	for _, msg := range messages {
		if msg.Role == RoleSystem {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		b.WriteString("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
	}

	for _, msg := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(string(msg.Role))
		b.WriteString("\n")
		b.WriteString(msg.Content)
		b.WriteString("<|im_end|>\n")
	}

	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

// FormatGemma2 formats messages using the Gemma 2 chat template.
//
// Format:
//
//	<start_of_turn>user
//	{user_message}<end_of_turn>
//	<start_of_turn>model
func FormatGemma2(messages []Message, ctx TemplateContext) string {
	var b strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case RoleSystem:
			b.WriteString("<start_of_turn>user\n")
			b.WriteString(msg.Content)
			b.WriteString("<end_of_turn>\n")
		case RoleUser:
			b.WriteString("<start_of_turn>user\n")
			b.WriteString(msg.Content)
			b.WriteString("<end_of_turn>\n")
		case RoleAssistant:
			b.WriteString("<start_of_turn>model\n")
			b.WriteString(msg.Content)
			b.WriteString("<end_of_turn>\n")
		}
	}

	b.WriteString("<start_of_turn>model\n")
	return b.String()
}

// MoondreamPointTokens builds token array for point detection.
// Template: <|md_reserved_0|>point<|md_reserved_1|> {object}<|md_reserved_2|>
// Token IDs: [1, 2581, 2] + tokenize(" " + object) + [3]
func MoondreamPointTokens(object string, encode func(string, bool) []int32) []int32 {
	tokens := []int32{1, 2581, 2} // <|md_reserved_0|>, "point", <|md_reserved_1|>
	objTokens := encode(" "+object, false)
	tokens = append(tokens, objTokens...)
	tokens = append(tokens, 3) // <|md_reserved_2|>
	return tokens
}

// MoondreamDetectTokens builds token array for object detection.
// Template: <|md_reserved_0|>detect all<|md_reserved_1|> {object}<|md_reserved_2|>
// Token IDs: [1, 7235, 476, 2] + tokenize(" " + object) + [3]
func MoondreamDetectTokens(object string, encode func(string, bool) []int32) []int32 {
	tokens := []int32{1, 7235, 476, 2} // <|md_reserved_0|>, "detect", "all", <|md_reserved_1|>
	objTokens := encode(" "+object, false)
	tokens = append(tokens, objTokens...)
	tokens = append(tokens, 3) // <|md_reserved_2|>
	return tokens
}

// FormatMoondream formats messages using the Moondream 2 chat template.
// Moondream uses special template tokens from the starmie-v1 tokenizer:
//   - <|endoftext|> (token 0) = BOS/EOS
//   - <|md_reserved_0|> (token 1) = query start marker
//   - "query" (token 15381) = query type indicator
//   - <|md_reserved_1|> (token 2) = separator
//   - <|md_reserved_2|> (token 3) = answer marker
//
// Text-only: <|endoftext|><|md_reserved_0|>query<|md_reserved_1|>{question}<|md_reserved_2|><|md_reserved_2|>
// With image: <|md_reserved_0|>query<|md_reserved_1|>{question}<|md_reserved_2|><|md_reserved_2|>
//
// (BOS is prepended by the vision prefill for image queries)
func FormatMoondream(messages []Message, ctx TemplateContext) string {
	var b strings.Builder

	// For vision queries, BOS is added by the C++ vision prefill path
	hasImage := false
	for _, m := range messages {
		if len(m.Images) > 0 {
			hasImage = true
			break
		}
	}

	if !hasImage {
		b.WriteString("<|endoftext|>")
	}

	b.WriteString("<|md_reserved_0|>query<|md_reserved_1|>")

	// Use the last user message as the question
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == RoleUser {
			b.WriteString(messages[i].Content)
			break
		}
	}

	b.WriteString("<|md_reserved_2|><|md_reserved_2|>")
	return b.String()
}
