# gomlx

Go bindings for [Apple MLX](https://github.com/ml-explore/mlx) — run vision and text LLMs natively on Apple Silicon.

## Why

I built this for my Go-based pick-and-place (PnP) machine vision system. I needed a way to run vision LLMs locally on a Mac to analyze camera feeds — things like checking nozzle alignment, verifying component placement, reading part markings — without depending on cloud APIs or Python. Nothing existed in Go for MLX, so I wrote one.

gomlx gives you a clean Go API to load quantized models and run inference with full Metal GPU acceleration. It supports both text-only and vision models, with streaming generation and conversation history.

## Supported Models

| Architecture | Type | Example Model |
|-------------|------|---------------|
| Llama 3.x | Text | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Llama 3.2 Vision (mllama) | Vision | `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit` |
| Mistral | Text | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| Qwen 2 | Text | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Qwen 2.5-VL | Vision | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| Gemma 2 | Text | `mlx-community/gemma-2-2b-it-4bit` |

The architecture is model-agnostic — adding a new model only requires a C++ `BaseModel` subclass, a registry entry, and a chat template function. No changes to the bridge or generation layer.

## Performance

Benchmarked on a **Mac Mini M4** (10-core, 16 GB RAM) with 4-bit quantized models.

### Text Generation

| Model | Parameters | tok/s (32 tok) | tok/s (128 tok) | Memory |
|-------|-----------|----------------|-----------------|--------|
| Qwen 2 | 3B | **47.0** | **45.7** | 2.3 GB |
| Llama 3.2 | 3B | 39.0 | 40.3 | 2.5 GB |
| Gemma 2 | 2B | 33.3 | 33.1 | 2.5 GB |
| Mistral | 7B | 21.7 | 20.9 | 4.1 GB |

### Vision

| Model | Parameters | tok/s (32 tok) | tok/s (64 tok) |
|-------|-----------|----------------|----------------|
| Qwen 2.5-VL | 7B | **13.7** | **13.9** |
| Llama 3.2 Vision | 11B | 9.8 | 10.8 |

All models produce correct outputs and scale linearly with token count. Run benchmarks yourself with:

```bash
go test -bench=. -benchtime=1x -timeout=600s -run=^$
```

## Requirements

- macOS with Apple Silicon (M1+)
- Go 1.23+
- Xcode Command Line Tools (for Metal framework)
- CMake

## Build

```bash
# Clone
git clone https://github.com/ril3y/gomlx.git
cd gomlx

# Build everything (fetches tokenizer lib, builds C++ via CMake, builds Go)
make build

# Download a model for testing
make download-model
```

## Quick Start

### Text Chat

```go
package main

import (
    "context"
    "fmt"
    "github.com/ril3y/gomlx"
)

func main() {
    ctx := context.Background()
    model, _ := gomlx.LoadModel(ctx, "./models/Llama-3.2-3B-Instruct-4bit")
    defer model.Close()

    output, _ := model.Generate(ctx, gomlx.GenerateInput{
        Messages: []gomlx.Message{
            {Role: gomlx.RoleUser, Content: "What is 2+2?"},
        },
    })
    fmt.Println(output.Content)
}
```

### Vision — Analyze an Image

```go
model, _ := gomlx.LoadModel(ctx, "./models/Qwen2.5-VL-7B-Instruct-4bit")
defer model.Close()

output, _ := model.Generate(ctx, gomlx.GenerateInput{
    Messages: []gomlx.Message{
        {
            Role:    gomlx.RoleUser,
            Content: "Is the crosshair centered on the nozzle? Say YES or NO.",
            Images:  []gomlx.Image{{Path: "nozzle.png"}},
        },
    },
    MaxTokens:   64,
    Temperature: 0.0,
})
fmt.Println(output.Content)
```

### Streaming

```go
model.GenerateStream(ctx, gomlx.GenerateInput{
    Messages: messages,
}, func(token string) {
    fmt.Print(token)
})
```

## Examples

```bash
# Interactive text chat
go run ./examples/text-chat ./models/Llama-3.2-3B-Instruct-4bit

# Describe an image
go run ./examples/vision-describe ./models/Qwen2.5-VL-7B-Instruct-4bit photo.jpg

# Vision with a specific question
go run ./examples/vision-describe ./models/Qwen2.5-VL-7B-Instruct-4bit nozzle.png \
  "Is the crosshair centered? Say YES or NO."
```

## Testing

```bash
# Run all tests (requires models in ./models/)
make test

# Run just unit tests (no models needed)
go test -run "TestFormat|TestGetTemplate|TestTemplateContext" -v

# Run benchmarks
go test -bench=. -benchtime=1x -timeout=600s -run=^$
```

## API

| Function | Description |
|----------|-------------|
| `LoadModel(ctx, path, opts...)` | Load a quantized MLX model from a directory |
| `Model.Generate(ctx, input)` | Generate a complete response |
| `Model.GenerateStream(ctx, input, callback)` | Stream tokens as they're generated |
| `Model.SupportsVision()` | Check if model handles image inputs |
| `Model.Close()` | Free model resources |
| `GetMemoryStats()` | Get active/peak Metal memory usage |

### Options

```go
gomlx.WithProgress(func(progress float32, message string) { ... })
gomlx.WithMaxTokens(1024)
```

### Input

```go
gomlx.GenerateInput{
    Messages:    []gomlx.Message{...},
    MaxTokens:   2048,        // default
    Temperature: 0.7,         // default
    TopP:        0.9,         // default
    Stop:        []string{},  // optional stop sequences
}
```

## Architecture

```
Go application
  └── gomlx (Go API)
        ├── chat.go        — chat templates per model architecture
        ├── generate.go    — model-agnostic generation loop
        ├── model.go       — model loading, stop token caching
        └── internal/bridge/
              └── bridge.go — CGo FFI to C API
                    └── mlxlib/
                          ├── mlx_bridge.cpp  — C API (model-agnostic virtual dispatch)
                          ├── base_model.h    — abstract interface
                          └── models/
                                ├── llama/    — LlamaText, LlamaVision
                                ├── mistral/  — MistralText
                                ├── qwen/     — Qwen2Text, Qwen2.5-VL
                                └── gemma/    — Gemma2Text
```

All model-specific behavior lives behind `BaseModel` virtual methods. The bridge and generation layer have zero knowledge of concrete model types.

## License

MIT
