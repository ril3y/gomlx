package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/ril3y/gomlx"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model-path> <image-path> [prompt]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nDescribe an image using a vision-capable LLM.\n")
		fmt.Fprintf(os.Stderr, "Example: %s ./models/Llama-3.2-11B-Vision-Instruct-4bit photo.jpg\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Example: %s ./models/Llama-3.2-11B-Vision-Instruct-4bit photo.jpg \"What is in this image?\"\n", os.Args[0])
		os.Exit(1)
	}
	modelPath := os.Args[1]
	imagePath := os.Args[2]
	prompt := "Describe this image in detail."
	if len(os.Args) >= 4 {
		prompt = os.Args[3]
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	fmt.Printf("Loading model from %s...\n", modelPath)
	model, err := gomlx.LoadModel(ctx, modelPath, gomlx.WithProgress(func(progress float32, message string) {
		fmt.Printf("\r  [%.0f%%] %s", progress*100, message)
	}))
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()
	fmt.Println()

	if !model.SupportsVision() {
		fmt.Fprintf(os.Stderr, "Error: model %q does not support vision\n", model.Architecture())
		os.Exit(1)
	}

	fmt.Printf("Describing image: %s\n\n", imagePath)

	messages := []gomlx.Message{
		{
			Role:    gomlx.RoleUser,
			Content: prompt,
			Images:  []gomlx.Image{{Path: imagePath}},
		},
	}

	maxTokens := 512
	temperature := float32(0.7)
	if len(os.Args) >= 4 {
		// Custom prompt mode: use lower temperature for precision
		maxTokens = 64
		temperature = 0.0
	}

	err = model.GenerateStream(ctx, gomlx.GenerateInput{
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        0.9,
	}, func(token string) {
		fmt.Print(token)
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}
	fmt.Println()

	stats := gomlx.GetMemoryStats()
	fmt.Printf("\n[memory: active=%.1f MB, peak=%.1f MB]\n",
		float64(stats.ActiveBytes)/(1024*1024),
		float64(stats.PeakBytes)/(1024*1024))
}
