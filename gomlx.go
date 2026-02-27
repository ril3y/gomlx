// Package gomlx provides native MLX inference on Apple Silicon from Go.
//
// gomlx wraps Apple's MLX framework via a CGo bridge, enabling efficient
// large language model inference directly from Go programs. It supports
// text generation with streaming, chat templates, and vision models.
//
// Basic usage:
//
//	model, err := gomlx.LoadModel(ctx, "/path/to/model")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	output, err := model.Generate(ctx, gomlx.GenerateInput{
//	    Messages: []gomlx.Message{
//	        {Role: gomlx.RoleUser, Content: "Hello!"},
//	    },
//	})
package gomlx
