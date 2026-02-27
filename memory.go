package gomlx

import "github.com/ril3y/gomlx/internal/bridge"

// MemoryStats contains MLX memory usage information.
type MemoryStats struct {
	ActiveBytes uint64
	PeakBytes   uint64
}

// GetMemoryStats returns current MLX memory usage statistics.
func GetMemoryStats() MemoryStats {
	return MemoryStats{
		ActiveBytes: bridge.GetActiveMemory(),
		PeakBytes:   bridge.GetPeakMemory(),
	}
}
