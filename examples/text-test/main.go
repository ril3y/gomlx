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
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model-path> [prompt]\n", os.Args[0])
		os.Exit(1)
	}
	modelPath := os.Args[1]
	prompt := "What is 2+2? Answer with just the number."
	if len(os.Args) >= 3 {
		prompt = os.Args[2]
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

	fmt.Printf("Prompt: %s\n\n", prompt)

	messages := []gomlx.Message{
		{Role: gomlx.RoleUser, Content: prompt},
	}

	// Debug: show formatted prompt
	templateFn := gomlx.GetTemplate(model.Architecture())
	formatted := templateFn(messages, gomlx.TemplateContext{})
	fmt.Printf("Formatted: %q\n\n", formatted)

	err = model.GenerateStream(ctx, gomlx.GenerateInput{
		Messages:    messages,
		MaxTokens:   32,
		Temperature: 0.0,
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
