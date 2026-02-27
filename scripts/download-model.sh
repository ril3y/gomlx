#!/usr/bin/env bash
set -euo pipefail

# Supported models:
#   mlx-community/Llama-3.2-3B-Instruct-4bit (text-only, ~2.5GB)
#   mlx-community/Llama-3.2-11B-Vision-Instruct-4bit (vision, ~6GB)
MODEL_REPO="${1:-mlx-community/Llama-3.2-3B-Instruct-4bit}"
MODELS_DIR="${MODELS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
MODEL_NAME="$(basename "$MODEL_REPO")"
MODEL_DIR="$MODELS_DIR/$MODEL_NAME"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Model already exists at $MODEL_DIR, skipping."
    exit 0
fi

mkdir -p "$MODELS_DIR"

# Prefer huggingface-cli if available
if command -v huggingface-cli &>/dev/null; then
    echo "Downloading $MODEL_REPO via huggingface-cli ..."
    huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"
elif command -v git &>/dev/null && command -v git-lfs &>/dev/null; then
    echo "Downloading $MODEL_REPO via git clone (LFS) ..."
    GIT_LFS_SKIP_SMUDGE=0 git clone "https://huggingface.co/$MODEL_REPO" "$MODEL_DIR"
else
    echo "Error: Neither huggingface-cli nor git-lfs found." >&2
    echo "Install one of:" >&2
    echo "  pip install huggingface_hub" >&2
    echo "  brew install git-lfs && git lfs install" >&2
    exit 1
fi

echo "Model downloaded to $MODEL_DIR"
