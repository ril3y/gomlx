package gomlx

// Role represents a participant in a chat conversation.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Image represents an image input for vision models.
type Image struct {
	// Path is the filesystem path to an image file.
	Path string
	// Data is raw image bytes. If both Path and Data are set, Data takes precedence.
	Data []byte
}

// Message represents a single message in a chat conversation.
type Message struct {
	Role    Role
	Content string
	Images  []Image
}

// GenerateInput configures a text generation request.
type GenerateInput struct {
	Messages    []Message
	MaxTokens   int
	Temperature float32
	TopP        float32
	Stop        []string
}

// GenerateOutput contains the result of a text generation request.
type GenerateOutput struct {
	Content            string
	TokenCount         int
	TokensPerSecond    float64
	TimeToFirstTokenMs float64
	TotalTimeMs        float64
}

// ProgressFunc is called during model loading to report progress.
type ProgressFunc func(progress float32, message string)

// ModelOption configures model loading behavior.
type ModelOption func(*modelOptions)

type modelOptions struct {
	progressFn ProgressFunc
	maxTokens  int
}

// WithProgress sets a callback for model loading progress updates.
func WithProgress(fn ProgressFunc) ModelOption {
	return func(o *modelOptions) {
		o.progressFn = fn
	}
}

// WithMaxTokens sets the default maximum tokens for generation.
func WithMaxTokens(n int) ModelOption {
	return func(o *modelOptions) {
		o.maxTokens = n
	}
}

// Point represents a detected point with normalized [0,1] coordinates.
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// BoundingBox represents a detected object's bounding box with normalized [0,1] coordinates.
type BoundingBox struct {
	XMin float64 `json:"x_min"`
	YMin float64 `json:"y_min"`
	XMax float64 `json:"x_max"`
	YMax float64 `json:"y_max"`
}

// PointResult contains the results of a point detection query.
type PointResult struct {
	Points []Point `json:"points"`
}

// DetectResult contains the results of an object detection query.
type DetectResult struct {
	Objects []BoundingBox `json:"objects"`
}
