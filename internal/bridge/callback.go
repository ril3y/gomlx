package bridge

/*
#include <stdlib.h>
*/
import "C"
import (
	"sync"
	"sync/atomic"
	"unsafe"
)

// ProgressCallback is the Go-side callback type for model loading progress.
type ProgressCallback func(progress float32, message string)

var (
	callbackMu     sync.Mutex
	callbackMap    = make(map[uintptr]ProgressCallback)
	nextCallbackID uint64 = 1
)

// RegisterCallback stores a Go callback and returns a handle that can be passed
// as user_data through the C bridge.
func RegisterCallback(fn ProgressCallback) uintptr {
	callbackMu.Lock()
	defer callbackMu.Unlock()
	id := atomic.AddUint64(&nextCallbackID, 1) - 1
	// Skip ID 0 which means "no callback"
	if id == 0 {
		id = atomic.AddUint64(&nextCallbackID, 1) - 1
	}
	handle := uintptr(id)
	callbackMap[handle] = fn
	return handle
}

// UnregisterCallback removes a previously registered callback.
func UnregisterCallback(handle uintptr) {
	callbackMu.Lock()
	defer callbackMu.Unlock()
	delete(callbackMap, handle)
}

//export goProgressCallback
func goProgressCallback(progress C.float, message *C.char, userData unsafe.Pointer) {
	handle := uintptr(userData)
	callbackMu.Lock()
	fn, ok := callbackMap[handle]
	callbackMu.Unlock()
	if ok && fn != nil {
		fn(float32(progress), C.GoString(message))
	}
}
