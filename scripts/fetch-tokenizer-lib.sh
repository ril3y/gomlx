#!/usr/bin/env bash
set -euo pipefail

LIB_DIR="${LIB_DIR:-$(cd "$(dirname "$0")/.." && pwd)/lib}"
TOKENIZER_LIB="$LIB_DIR/libtokenizers.a"

# Skip if already downloaded
if [ -f "$TOKENIZER_LIB" ]; then
    echo "libtokenizers.a already exists, skipping download."
    exit 0
fi

# Determine architecture
ARCH="$(uname -m)"
case "$ARCH" in
    arm64) PLATFORM="darwin-arm64" ;;
    x86_64) PLATFORM="darwin-amd64" ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

VERSION="v1.20.2"
URL="https://github.com/daulet/tokenizers/releases/download/${VERSION}/libtokenizers.${PLATFORM}.tar.gz"

echo "Downloading libtokenizers ($PLATFORM) from $URL ..."
mkdir -p "$LIB_DIR"

TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

curl -fsSL "$URL" -o "$TMPFILE"
tar -xzf "$TMPFILE" -C "$LIB_DIR"

if [ -f "$TOKENIZER_LIB" ]; then
    echo "Successfully installed libtokenizers.a to $LIB_DIR/"
else
    echo "Error: libtokenizers.a not found after extraction." >&2
    ls -la "$LIB_DIR/" >&2
    exit 1
fi
