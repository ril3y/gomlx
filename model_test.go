package gomlx

import (
	"context"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Model paths — tests skip if not present.
const (
	llamaTextPath    = "models/Llama-3.2-3B-Instruct-4bit"
	llamaVisionPath  = "models/Llama-3.2-11B-Vision-Instruct-4bit"
	mistralPath      = "models/Mistral-7B-Instruct-v0.3-4bit"
	qwen2Path        = "models/Qwen2.5-3B-Instruct-4bit"
	qwen25vlPath     = "models/Qwen2.5-VL-7B-Instruct-4bit"
	gemma2Path       = "models/gemma-2-2b-it-4bit"
	testImagePath    = "testdata/nozzle.png"
)

func modelExists(path string) bool {
	_, err := os.Stat(path + "/config.json")
	return err == nil
}

func skipIfNoModel(t *testing.T, path string) {
	if !modelExists(path) {
		t.Skipf("model not found at %s, skipping", path)
	}
}

// --- Model Loading Tests ---

func TestLoadModel_InvalidPath(t *testing.T) {
	ctx := context.Background()
	_, err := LoadModel(ctx, "/nonexistent/path")
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrModelNotFound)
}

func TestLoadModel_Llama(t *testing.T) {
	skipIfNoModel(t, llamaTextPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, llamaTextPath)
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, "llama", m.Architecture())
	assert.False(t, m.SupportsVision())
	assert.Greater(t, m.VocabSize(), 0)
}

func TestLoadModel_Mistral(t *testing.T) {
	skipIfNoModel(t, mistralPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, mistralPath)
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, "mistral", m.Architecture())
	assert.False(t, m.SupportsVision())
	assert.Greater(t, m.VocabSize(), 0)
}

func TestLoadModel_Qwen2(t *testing.T) {
	skipIfNoModel(t, qwen2Path)
	ctx := context.Background()
	m, err := LoadModel(ctx, qwen2Path)
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, "qwen2", m.Architecture())
	assert.False(t, m.SupportsVision())
	assert.Greater(t, m.VocabSize(), 0)
}

func TestLoadModel_Gemma2(t *testing.T) {
	skipIfNoModel(t, gemma2Path)
	ctx := context.Background()
	m, err := LoadModel(ctx, gemma2Path)
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, "gemma2", m.Architecture())
	assert.False(t, m.SupportsVision())
	assert.Greater(t, m.VocabSize(), 0)
}

func TestLoadModel_LlamaVision(t *testing.T) {
	skipIfNoModel(t, llamaVisionPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, llamaVisionPath)
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, "mllama", m.Architecture())
	assert.True(t, m.SupportsVision())
}

func TestLoadModel_Qwen25VL(t *testing.T) {
	skipIfNoModel(t, qwen25vlPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, qwen25vlPath)
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, "qwen2_5_vl", m.Architecture())
	assert.True(t, m.SupportsVision())
}

// --- Text Generation Tests ---

func generateText(t *testing.T, modelPath string, prompt string) string {
	t.Helper()
	skipIfNoModel(t, modelPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, modelPath)
	require.NoError(t, err)
	defer m.Close()

	out, err := m.Generate(ctx, GenerateInput{
		Messages:    []Message{{Role: RoleUser, Content: prompt}},
		MaxTokens:   32,
		Temperature: 0.0,
		TopP:        0.9,
	})
	require.NoError(t, err)
	require.NotNil(t, out)
	assert.Greater(t, out.TokenCount, 0)
	assert.Greater(t, len(out.Content), 0)
	return out.Content
}

func TestGenerate_Llama(t *testing.T) {
	content := generateText(t, llamaTextPath, "What is 2+2? Answer with just the number.")
	assert.Contains(t, content, "4")
}

func TestGenerate_Mistral(t *testing.T) {
	content := generateText(t, mistralPath, "What is 2+2? Answer with just the number.")
	assert.Contains(t, content, "4")
}

func TestGenerate_Qwen2(t *testing.T) {
	content := generateText(t, qwen2Path, "What is 2+2? Answer with just the number.")
	assert.Contains(t, content, "4")
}

func TestGenerate_Gemma2(t *testing.T) {
	content := generateText(t, gemma2Path, "What is 2+2? Answer with just the number.")
	assert.Contains(t, content, "4")
}

// --- Streaming Tests ---

func TestGenerateStream_Llama(t *testing.T) {
	skipIfNoModel(t, llamaTextPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, llamaTextPath)
	require.NoError(t, err)
	defer m.Close()

	var chunks []string
	err = m.GenerateStream(ctx, GenerateInput{
		Messages:    []Message{{Role: RoleUser, Content: "Say hello."}},
		MaxTokens:   16,
		Temperature: 0.0,
	}, func(token string) {
		chunks = append(chunks, token)
	})
	require.NoError(t, err)
	assert.Greater(t, len(chunks), 0, "should have received streaming tokens")

	full := strings.Join(chunks, "")
	assert.Greater(t, len(full), 0)
}

// --- Vision Tests ---

func TestVision_LlamaVision(t *testing.T) {
	skipIfNoModel(t, llamaVisionPath)
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("test image not found")
	}

	ctx := context.Background()
	m, err := LoadModel(ctx, llamaVisionPath)
	require.NoError(t, err)
	defer m.Close()

	out, err := m.Generate(ctx, GenerateInput{
		Messages: []Message{
			{
				Role:    RoleUser,
				Content: "What do you see in this image? Be brief.",
				Images:  []Image{{Path: testImagePath}},
			},
		},
		MaxTokens:   64,
		Temperature: 0.0,
	})
	require.NoError(t, err)
	assert.Greater(t, out.TokenCount, 0)
	assert.Greater(t, len(out.Content), 0)
	t.Logf("Llama Vision output: %s", out.Content)
}

func TestVision_Qwen25VL(t *testing.T) {
	skipIfNoModel(t, qwen25vlPath)
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("test image not found")
	}

	ctx := context.Background()
	m, err := LoadModel(ctx, qwen25vlPath)
	require.NoError(t, err)
	defer m.Close()

	out, err := m.Generate(ctx, GenerateInput{
		Messages: []Message{
			{
				Role:    RoleUser,
				Content: "What do you see in this image? Be brief.",
				Images:  []Image{{Path: testImagePath}},
			},
		},
		MaxTokens:   64,
		Temperature: 0.0,
	})
	require.NoError(t, err)
	assert.Greater(t, out.TokenCount, 0)
	assert.Greater(t, len(out.Content), 0)
	t.Logf("Qwen2.5-VL output: %s", out.Content)
}

func TestVision_Qwen25VL_NozzleCrosshair(t *testing.T) {
	skipIfNoModel(t, qwen25vlPath)
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("test image not found")
	}

	ctx := context.Background()
	m, err := LoadModel(ctx, qwen25vlPath)
	require.NoError(t, err)
	defer m.Close()

	out, err := m.Generate(ctx, GenerateInput{
		Messages: []Message{
			{
				Role:    RoleUser,
				Content: "Is the crosshair centered? Answer YES or NO only.",
				Images:  []Image{{Path: testImagePath}},
			},
		},
		MaxTokens:   8,
		Temperature: 0.0,
	})
	require.NoError(t, err)
	assert.Greater(t, out.TokenCount, 0)
	t.Logf("Nozzle crosshair answer: %s", out.Content)
	// Should contain YES or NO
	upper := strings.ToUpper(out.Content)
	assert.True(t, strings.Contains(upper, "YES") || strings.Contains(upper, "NO"),
		"expected YES or NO, got: %s", out.Content)
}

// --- Text model rejects vision ---

func TestVision_TextModelRejectsImages(t *testing.T) {
	skipIfNoModel(t, llamaTextPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, llamaTextPath)
	require.NoError(t, err)
	defer m.Close()

	assert.False(t, m.SupportsVision())
	err = m.PrepareVisionInput([]Image{{Path: testImagePath}})
	assert.ErrorIs(t, err, ErrVisionNotSupported)
}

// --- Context Cancellation ---

func TestGenerate_CanceledContext(t *testing.T) {
	skipIfNoModel(t, llamaTextPath)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	m, err := LoadModel(context.Background(), llamaTextPath)
	require.NoError(t, err)
	defer m.Close()

	_, err = m.Generate(ctx, GenerateInput{
		Messages: []Message{{Role: RoleUser, Content: "Hello"}},
	})
	assert.ErrorIs(t, err, ErrContextCanceled)
}

// --- Memory Stats ---

func TestMemoryStats(t *testing.T) {
	stats := GetMemoryStats()
	// Just verify it doesn't panic and returns something
	assert.GreaterOrEqual(t, stats.PeakBytes, stats.ActiveBytes)
}

// --- Stop Tokens ---

func TestStopTokens_Llama(t *testing.T) {
	skipIfNoModel(t, llamaTextPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, llamaTextPath)
	require.NoError(t, err)
	defer m.Close()

	assert.True(t, m.isStopToken(128001), "Llama EOS should be stop token")
	assert.True(t, m.isStopToken(128009), "Llama EOT should be stop token")
	assert.False(t, m.isStopToken(0), "token 0 should not be stop token")
}

func TestStopTokens_Mistral(t *testing.T) {
	skipIfNoModel(t, mistralPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, mistralPath)
	require.NoError(t, err)
	defer m.Close()

	assert.True(t, m.isStopToken(2), "Mistral </s> should be stop token")
	assert.False(t, m.isStopToken(128001), "Llama EOS should not be Mistral stop token")
}

func TestStopTokens_Gemma2(t *testing.T) {
	skipIfNoModel(t, gemma2Path)
	ctx := context.Background()
	m, err := LoadModel(ctx, gemma2Path)
	require.NoError(t, err)
	defer m.Close()

	assert.True(t, m.isStopToken(1), "Gemma EOS should be stop token")
	assert.True(t, m.isStopToken(107), "Gemma end_of_turn should be stop token")
	assert.False(t, m.isStopToken(128001), "Llama EOS should not be Gemma stop token")
}

// --- Thread Safety ---

func TestGenerate_ConcurrentSafe(t *testing.T) {
	skipIfNoModel(t, llamaTextPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, llamaTextPath)
	require.NoError(t, err)
	defer m.Close()

	var wg sync.WaitGroup
	errors := make(chan error, 4)

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := m.Generate(ctx, GenerateInput{
				Messages:    []Message{{Role: RoleUser, Content: "Say hi"}},
				MaxTokens:   4,
				Temperature: 0.0,
			})
			if err != nil {
				errors <- err
			}
		}()
	}

	wg.Wait()
	close(errors)
	for err := range errors {
		t.Errorf("concurrent generate error: %v", err)
	}
}

// --- Latency Tracking ---

func TestGenerate_LatencyTracking(t *testing.T) {
	skipIfNoModel(t, llamaTextPath)
	ctx := context.Background()
	m, err := LoadModel(ctx, llamaTextPath)
	require.NoError(t, err)
	defer m.Close()

	out, err := m.Generate(ctx, GenerateInput{
		Messages:    []Message{{Role: RoleUser, Content: "What is 2+2?"}},
		MaxTokens:   16,
		Temperature: 0.0,
	})
	require.NoError(t, err)

	assert.Greater(t, out.TimeToFirstTokenMs, 0.0, "TTFT should be positive")
	assert.Greater(t, out.TotalTimeMs, 0.0, "TotalTime should be positive")
	assert.GreaterOrEqual(t, out.TotalTimeMs, out.TimeToFirstTokenMs, "TotalTime should be >= TTFT")
	t.Logf("TTFT: %.1fms, Total: %.1fms, Tokens: %d", out.TimeToFirstTokenMs, out.TotalTimeMs, out.TokenCount)
}

// --- Multi-Image Vision ---

func TestVision_MultipleImages(t *testing.T) {
	skipIfNoModel(t, qwen25vlPath)
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("test image not found")
	}
	if _, err := os.Stat("testdata/test.png"); os.IsNotExist(err) {
		t.Skip("second test image not found")
	}

	ctx := context.Background()
	m, err := LoadModel(ctx, qwen25vlPath)
	require.NoError(t, err)
	defer m.Close()

	// Send multiple images in one message — should not error
	out, err := m.Generate(ctx, GenerateInput{
		Messages: []Message{
			{
				Role:    RoleUser,
				Content: "What do you see?",
				Images: []Image{
					{Path: "testdata/test.png"},
					{Path: testImagePath},
				},
			},
		},
		MaxTokens:   32,
		Temperature: 0.0,
	})
	require.NoError(t, err)
	assert.Greater(t, out.TokenCount, 0)
	t.Logf("Multi-image output: %s", out.Content)
}
