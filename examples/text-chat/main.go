package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/ril3y/gomlx"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model-path>\n", os.Args[0])
		os.Exit(1)
	}
	modelPath := os.Args[1]

	// Set up context with Ctrl+C handling
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	fmt.Printf("Loading model from %s...\n", modelPath)

	model, err := gomlx.LoadModel(ctx, modelPath, gomlx.WithProgress(func(progress float32, message string) {
		fmt.Printf("\r  [%.0f%%] %s", progress*100, message)
	}))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	fmt.Printf("\nModel loaded: arch=%s vocab=%d vision=%v\n", model.Architecture(), model.VocabSize(), model.SupportsVision())
	fmt.Println("Type your message (or 'quit' to exit):")

	var history []gomlx.Message
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "quit" || input == "exit" {
			break
		}

		history = append(history, gomlx.Message{
			Role:    gomlx.RoleUser,
			Content: input,
		})

		// Create a cancelable context for this generation
		genCtx, genCancel := context.WithCancel(ctx)

		err := model.GenerateStream(genCtx, gomlx.GenerateInput{
			Messages: history,
		}, func(token string) {
			fmt.Print(token)
		})
		genCancel()

		if err != nil {
			fmt.Fprintf(os.Stderr, "\nGeneration error: %v\n", err)
			// Remove the failed user message from history
			history = history[:len(history)-1]
			continue
		}
		fmt.Println()

		// Print memory stats
		stats := gomlx.GetMemoryStats()
		fmt.Printf("[memory: active=%.1f MB, peak=%.1f MB]\n",
			float64(stats.ActiveBytes)/(1024*1024),
			float64(stats.PeakBytes)/(1024*1024))
	}

	fmt.Println("Goodbye!")
}
