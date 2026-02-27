package bridge

/*
#include <stdlib.h>
*/
import "C"
import (
	"sync"
	"unsafe"
)

// ProgressCallback is the Go-side callback type for model loading progress.
type ProgressCallback func(progress float32, message string)

var (
	callbackMu    sync.Mutex
	callbackMap   = make(map[uintptr]ProgressCallback)
	nextCallbackID uintptr = 1
)

// RegisterCallback stores a Go callback and returns a handle that can be passed
// as user_data through the C bridge.
func RegisterCallback(fn ProgressCallback) uintptr {
	callbackMu.Lock()
	defer callbackMu.Unlock()
	id := nextCallbackID
	nextCallbackID++
	callbackMap[id] = fn
	return id
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
