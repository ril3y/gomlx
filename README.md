# gomlx

Go bindings for [Apple MLX](https://github.com/ml-explore/mlx) — run vision and text LLMs natively on Apple Silicon.

## Why

I built this for my Go-based pick-and-place (PnP) machine vision system. I needed a way to run vision LLMs locally on a Mac to analyze camera feeds — things like checking nozzle alignment, verifying component placement, reading part markings — without depending on cloud APIs or Python. Nothing existed in Go for MLX, so I wrote one.

gomlx gives you a clean Go API to load quantized models and run inference with full Metal GPU acceleration. It supports text generation, vision describe, streaming, and ML-guided coordinate/bounding-box detection for machine vision workflows.

## Supported Models

| Architecture | Type | Example Model |
|-------------|------|---------------|
| Llama 3.x | Text | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Llama 3.2 Vision (mllama) | Vision | `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit` |
| Mistral | Text | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| Qwen 2 | Text | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Qwen 2.5-VL | Vision | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| Gemma 2 | Text | `mlx-community/gemma-2-2b-it-4bit` |
| Moondream 2 | Vision + Point/Detect | `vikhyatk/moondream2` |

The architecture is model-agnostic — adding a new model only requires a C++ `BaseModel` subclass, a registry entry, and a chat template function. No changes to the bridge or generation layer.

### Moondream 2

[Moondream 2](https://github.com/vikhyat/moondream) is a lightweight vision-language model (1.9B parameters) built on SigLIP + Phi-1.5. In addition to standard vision describe, gomlx exposes Moondream's **region model** for coordinate-level detection:

- **`Point()`** — returns normalized (x, y) coordinates for a named object. Use this to locate the center of a part, hole, fiducial, or any visual feature.
- **`Detect()`** — returns bounding boxes (x_min, y_min, x_max, y_max) for all instances of a named object.

All coordinates are normalized to [0, 1]. Multiply by your image dimensions to get pixel coordinates.

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

## Install

```bash
go get github.com/ril3y/gomlx
```

## Build from Source

```bash
# Clone
git clone https://github.com/ril3y/gomlx.git
cd gomlx

# Build everything (fetches tokenizer lib, builds C++ via CMake, builds Go)
make build

# Download a model for testing
make download-model
```

## API Reference

### Loading a Model

```go
model, err := gomlx.LoadModel(ctx, "./models/Llama-3.2-3B-Instruct-4bit",
    gomlx.WithProgress(func(progress float32, message string) {
        fmt.Printf("[%.0f%%] %s\n", progress*100, message)
    }),
    gomlx.WithMaxTokens(1024),
)
if err != nil {
    log.Fatal(err)
}
defer model.Close()
```

`LoadModel` loads a model directory containing `config.json`, safetensors weights, and `tokenizer.json`. It returns a thread-safe `*Model` that can be used for all inference operations.

**Options:**

| Option | Description |
|--------|-------------|
| `WithProgress(fn)` | Callback for loading progress (0.0 to 1.0) |
| `WithMaxTokens(n)` | Default max tokens for generation (default: 2048) |

### Text Generation

```go
output, err := model.Generate(ctx, gomlx.GenerateInput{
    Messages: []gomlx.Message{
        {Role: gomlx.RoleSystem, Content: "You are a helpful assistant."},
        {Role: gomlx.RoleUser, Content: "What is 2+2?"},
    },
    MaxTokens:   256,
    Temperature: 0.7,
    TopP:        0.9,
    Stop:        []string{"\n\n"},
})
fmt.Println(output.Content)
fmt.Printf("%.1f tok/s\n", output.TokensPerSecond)
```

`Generate` runs a complete generation and returns the full response. Pass `context.Context` for cancellation support.

**`GenerateInput` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Messages` | `[]Message` | required | Conversation history |
| `MaxTokens` | `int` | 2048 | Maximum tokens to generate |
| `Temperature` | `float32` | 0.7 | Sampling temperature (0 = greedy) |
| `TopP` | `float32` | 0.9 | Nucleus sampling threshold |
| `Stop` | `[]string` | none | Stop generation on these strings |

**`GenerateOutput` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Content` | `string` | Generated text |
| `TokenCount` | `int` | Number of tokens generated |
| `TokensPerSecond` | `float64` | Generation throughput |
| `TimeToFirstTokenMs` | `float64` | Prefill latency |
| `TotalTimeMs` | `float64` | Total wall-clock time |

### Streaming Generation

```go
err := model.GenerateStream(ctx, gomlx.GenerateInput{
    Messages: []gomlx.Message{
        {Role: gomlx.RoleUser, Content: "Write a haiku about Go."},
    },
}, func(token string) {
    fmt.Print(token) // called with each new text fragment
})
```

`GenerateStream` calls the callback with each new text fragment as it is generated. Same input options as `Generate`.

### Vision — Describe an Image

Any vision-capable model can analyze images. Attach images to a user message:

```go
model, _ := gomlx.LoadModel(ctx, "./models/Qwen2.5-VL-7B-Instruct-4bit")
defer model.Close()

output, err := model.Generate(ctx, gomlx.GenerateInput{
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
```

Images can be loaded from a file path or raw bytes:

```go
gomlx.Image{Path: "/path/to/photo.jpg"}   // from file
gomlx.Image{Data: jpegBytes}               // from bytes (Data takes precedence)
```

### Point Detection (Moondream 2)

`Point` locates the center of a named object in an image and returns normalized [0,1] coordinates.

```go
model, _ := gomlx.LoadModel(ctx, "./models/moondream2")
defer model.Close()

result, err := model.Point(ctx, gomlx.Image{Path: "nozzle.png"}, "nozzle hole")
if err != nil {
    log.Fatal(err)
}

for _, pt := range result.Points {
    fmt.Printf("x=%.3f y=%.3f\n", pt.X, pt.Y)
    // Convert to pixels: px_x = pt.X * imageWidth, px_y = pt.Y * imageHeight
}
// Output: x=0.496 y=0.522
```

**`PointResult`:**

```go
type PointResult struct {
    Points []Point `json:"points"`
}

type Point struct {
    X float64 `json:"x"` // Normalized [0,1]
    Y float64 `json:"y"` // Normalized [0,1]
}
```

The model can return multiple points if more than one instance of the object is found.

### Object Detection (Moondream 2)

`Detect` finds all instances of a named object and returns bounding boxes.

```go
result, err := model.Detect(ctx, gomlx.Image{Path: "board.png"}, "capacitor")
if err != nil {
    log.Fatal(err)
}

for _, box := range result.Objects {
    fmt.Printf("bbox: (%.3f,%.3f)-(%.3f,%.3f)\n", box.XMin, box.YMin, box.XMax, box.YMax)
}
```

**`DetectResult`:**

```go
type DetectResult struct {
    Objects []BoundingBox `json:"objects"`
}

type BoundingBox struct {
    XMin float64 `json:"x_min"` // Normalized [0,1]
    YMin float64 `json:"y_min"`
    XMax float64 `json:"x_max"`
    YMax float64 `json:"y_max"`
}
```

### Model Info

```go
model.Architecture()   // "llama", "moondream1", "qwen2_5_vl", etc.
model.SupportsVision() // true if the model accepts image inputs
model.VocabSize()      // number of tokens in the vocabulary
gomlx.GetMemoryStats() // returns MemoryStats{ActiveBytes, PeakBytes}
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

# Point detection — get xy coordinates of an object
go run ./examples/vision-point ./models/moondream2 nozzle.png "nozzle hole"

# Object detection — get bounding boxes
go run ./examples/vision-detect ./models/moondream2 board.png "capacitor"
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

### Test Coverage

| Test Suite | What it covers |
|-----------|---------------|
| `TestFormat*` | Chat template formatting for all 7 architectures |
| `TestGetTemplate*` | Template registry lookup and fallback behavior |
| `TestLoadModel*` | Model loading for Llama, Mistral, Qwen2, Gemma2, LlamaVision, Qwen2.5-VL |
| `TestGenerate*` | Text generation, streaming, cancellation, concurrency, latency |
| `TestVision*` | Vision inference with LlamaVision and Qwen2.5-VL |
| `TestStopTokens*` | Stop token detection for Llama, Mistral, Gemma2 |
| `TestMemoryStats` | Metal memory reporting |

## Architecture

```
Go application
  └── gomlx (Go API)
        ├── chat.go        — chat templates per model architecture
        ├── generate.go    — model-agnostic generation loop
        ├── vision.go      — Point() and Detect() for region detection
        ├── model.go       — model loading, stop token caching
        └── internal/bridge/
              ├── bridge.go        — CGo FFI to C API
              └── bridge_region.go — CGo FFI for point/detect
                    └── mlxlib/
                          ├── mlx_bridge.cpp  — C API (model-agnostic virtual dispatch)
                          ├── base_model.h    — abstract interface
                          └── models/
                                ├── llama/      — LlamaText, LlamaVision
                                ├── mistral/    — MistralText
                                ├── qwen/       — Qwen2Text, Qwen2.5-VL
                                ├── gemma/      — Gemma2Text
                                └── moondream/  — Moondream2 (SigLIP + Phi-1.5 + Region)
```

All model-specific behavior lives behind `BaseModel` virtual methods. The bridge and generation layer have zero knowledge of concrete model types.

## License

MIT
