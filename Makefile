# gomlx - Go bindings for Apple MLX
# Build system for CGo + C++ static libraries

# macOS SDK
MACOS_SDK := $(shell xcrun --show-sdk-path 2>/dev/null)

# Directories
ROOT_DIR     := $(shell pwd)
BUILD_DIR    := $(ROOT_DIR)/build
LIB_DIR      := $(ROOT_DIR)/lib
MLXLIB_DIR   := $(ROOT_DIR)/mlxlib
SCRIPTS_DIR  := $(ROOT_DIR)/scripts

# CGo flags
export CGO_CFLAGS  := -I$(MLXLIB_DIR) -I$(ROOT_DIR)/third_party/mlx
export CGO_LDFLAGS := \
	-L$(BUILD_DIR) -lmlxbridge \
	-L$(BUILD_DIR)/mlx_build -lmlx \
	-L$(LIB_DIR) -ltokenizers \
	-framework Metal -framework Foundation -framework Accelerate \
	-lc++

.PHONY: all cmake build-cpp fetch-tokenizer-lib build test clean

all: fetch-tokenizer-lib build-cpp
	@echo "Build complete. Run 'make test' or 'go build ./...' to verify."

# Build C++ libraries via CMake
cmake:
	@mkdir -p $(BUILD_DIR)
	cmake -S $(MLXLIB_DIR) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_OSX_SYSROOT=$(MACOS_SDK)

build-cpp: cmake
	cmake --build $(BUILD_DIR) --config Release -j$(shell sysctl -n hw.ncpu)

# Fetch pre-built tokenizer library
fetch-tokenizer-lib:
	@$(SCRIPTS_DIR)/fetch-tokenizer-lib.sh

# Go build and test
build: all
	go build ./...

test: all
	go test ./...

# Download a model for local testing
download-model:
	@$(SCRIPTS_DIR)/download-model.sh

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(LIB_DIR)
