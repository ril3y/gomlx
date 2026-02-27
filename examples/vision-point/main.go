package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/ril3y/gomlx"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model-path> <image-path> <object>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nDetect the point location of an object in an image.\n")
		fmt.Fprintf(os.Stderr, "Example: %s ./models/moondream2 ./testdata/nozzle.png \"nozzle hole\"\n", os.Args[0])
		os.Exit(1)
	}
	modelPath := os.Args[1]
	imagePath := os.Args[2]
	object := os.Args[3]

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

	fmt.Printf("Detecting point: %q in %s\n", object, imagePath)

	result, err := model.Point(ctx, gomlx.Image{Path: imagePath}, object)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	out, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(out))

	stats := gomlx.GetMemoryStats()
	fmt.Printf("\n[memory: active=%.1f MB, peak=%.1f MB]\n",
		float64(stats.ActiveBytes)/(1024*1024),
		float64(stats.PeakBytes)/(1024*1024))
}
