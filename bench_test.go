package gomlx

import (
	"context"
	"os"
	"testing"
)

// Benchmarks measure end-to-end generation performance per model architecture.
// Run with: go test -bench=. -benchtime=1x -timeout=600s
// The -benchtime=1x flag runs each benchmark exactly once (model loading is expensive).

func benchmarkGenerate(b *testing.B, modelPath string, prompt string, maxTokens int) {
	b.Helper()
	if !modelExists(modelPath) {
		b.Skipf("model not found at %s", modelPath)
	}

	ctx := context.Background()
	m, err := LoadModel(ctx, modelPath)
	if err != nil {
		b.Fatalf("failed to load model: %v", err)
	}
	defer m.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := m.Generate(ctx, GenerateInput{
			Messages:    []Message{{Role: RoleUser, Content: prompt}},
			MaxTokens:   maxTokens,
			Temperature: 0.0,
			TopP:        0.9,
		})
		if err != nil {
			b.Fatalf("generation failed: %v", err)
		}
		b.ReportMetric(out.TokensPerSecond, "tok/s")
		b.ReportMetric(float64(out.TokenCount), "tokens")
		b.ReportMetric(out.TimeToFirstTokenMs, "ttft_ms")
		b.ReportMetric(out.TotalTimeMs, "total_ms")
	}
}

func benchmarkVisionGenerate(b *testing.B, modelPath string, imagePath string, prompt string, maxTokens int) {
	b.Helper()
	if !modelExists(modelPath) {
		b.Skipf("model not found at %s", modelPath)
	}

	ctx := context.Background()
	m, err := LoadModel(ctx, modelPath)
	if err != nil {
		b.Fatalf("failed to load model: %v", err)
	}
	defer m.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := m.Generate(ctx, GenerateInput{
			Messages: []Message{
				{
					Role:    RoleUser,
					Content: prompt,
					Images:  []Image{{Path: imagePath}},
				},
			},
			MaxTokens:   maxTokens,
			Temperature: 0.0,
			TopP:        0.9,
		})
		if err != nil {
			b.Fatalf("generation failed: %v", err)
		}
		b.ReportMetric(out.TokensPerSecond, "tok/s")
		b.ReportMetric(float64(out.TokenCount), "tokens")
		b.ReportMetric(out.TimeToFirstTokenMs, "ttft_ms")
		b.ReportMetric(out.TotalTimeMs, "total_ms")
	}
}

// --- Text Generation Benchmarks (32 tokens) ---

func BenchmarkGenerate_Llama_32tok(b *testing.B) {
	benchmarkGenerate(b, llamaTextPath, "Explain what a compiler does in one paragraph.", 32)
}

func BenchmarkGenerate_Mistral_32tok(b *testing.B) {
	benchmarkGenerate(b, mistralPath, "Explain what a compiler does in one paragraph.", 32)
}

func BenchmarkGenerate_Qwen2_32tok(b *testing.B) {
	benchmarkGenerate(b, qwen2Path, "Explain what a compiler does in one paragraph.", 32)
}

func BenchmarkGenerate_Gemma2_32tok(b *testing.B) {
	benchmarkGenerate(b, gemma2Path, "Explain what a compiler does in one paragraph.", 32)
}

// --- Text Generation Benchmarks (128 tokens) ---

func BenchmarkGenerate_Llama_128tok(b *testing.B) {
	benchmarkGenerate(b, llamaTextPath, "Write a short essay about the history of computing.", 128)
}

func BenchmarkGenerate_Mistral_128tok(b *testing.B) {
	benchmarkGenerate(b, mistralPath, "Write a short essay about the history of computing.", 128)
}

func BenchmarkGenerate_Qwen2_128tok(b *testing.B) {
	benchmarkGenerate(b, qwen2Path, "Write a short essay about the history of computing.", 128)
}

func BenchmarkGenerate_Gemma2_128tok(b *testing.B) {
	benchmarkGenerate(b, gemma2Path, "Write a short essay about the history of computing.", 128)
}

// --- Vision Benchmarks ---

func BenchmarkVision_LlamaVision_32tok(b *testing.B) {
	benchmarkVisionGenerate(b, llamaVisionPath, testImagePath,
		"What do you see in this image? Be brief.", 32)
}

func BenchmarkVision_Qwen25VL_32tok(b *testing.B) {
	benchmarkVisionGenerate(b, qwen25vlPath, testImagePath,
		"What do you see in this image? Be brief.", 32)
}

func BenchmarkVision_LlamaVision_64tok(b *testing.B) {
	benchmarkVisionGenerate(b, llamaVisionPath, testImagePath,
		"Describe this image in detail.", 64)
}

func BenchmarkVision_Qwen25VL_64tok(b *testing.B) {
	benchmarkVisionGenerate(b, qwen25vlPath, testImagePath,
		"Describe this image in detail.", 64)
}

// --- Image-from-Bytes Benchmark ---

func BenchmarkVision_Qwen25VL_Bytes_32tok(b *testing.B) {
	if !modelExists(qwen25vlPath) {
		b.Skipf("model not found at %s", qwen25vlPath)
	}
	imgData, err := os.ReadFile(testImagePath)
	if err != nil {
		b.Skipf("test image not found: %v", err)
	}

	ctx := context.Background()
	m, err := LoadModel(ctx, qwen25vlPath)
	if err != nil {
		b.Fatalf("failed to load model: %v", err)
	}
	defer m.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := m.Generate(ctx, GenerateInput{
			Messages: []Message{
				{
					Role:    RoleUser,
					Content: "What do you see? Be brief.",
					Images:  []Image{{Data: imgData}},
				},
			},
			MaxTokens:   32,
			Temperature: 0.0,
			TopP:        0.9,
		})
		if err != nil {
			b.Fatalf("generation failed: %v", err)
		}
		b.ReportMetric(out.TokensPerSecond, "tok/s")
		b.ReportMetric(float64(out.TokenCount), "tokens")
		b.ReportMetric(out.TimeToFirstTokenMs, "ttft_ms")
	}
}
