#!/bin/bash
set -e

VERSION="nightly-2026-04-20"
DEPS_DIR="deps/pjrt"

# Detect OS
OS="linux"
case "$(uname -s)" in
    Linux*)     OS="linux";;
    Darwin*)    OS="darwin";;
    CYGWIN*|MINGW*|MSYS*) OS="windows";;
    *)          echo "Unsupported OS: $(uname -s)"; exit 1;;
esac

# Detect architecture and map to artifact naming
ARCH="amd64"
case "$(uname -m)" in
    x86_64|amd64)  ARCH="amd64";;
    aarch64|arm64) ARCH="arm64";;
    *)             echo "Unsupported architecture: $(uname -m)"; exit 1;;
esac

mkdir -p "$DEPS_DIR"

echo "Detected platform: ${OS}-${ARCH}"
echo "Downloading PJRT plugins from zml/pjrt-artifacts (${VERSION})..."

# CPU plugin (always needed)
CPU_TAR="pjrt-cpu_${OS}-${ARCH}.tar.gz"
CPU_URL="https://github.com/zml/pjrt-artifacts/releases/download/${VERSION}/${CPU_TAR}"

echo "  → CPU plugin: ${CPU_URL}"
curl -fsSL -o "/tmp/${CPU_TAR}" "$CPU_URL"
tar -xzf "/tmp/${CPU_TAR}" -C "$DEPS_DIR" --strip-components=1
rm -f "/tmp/${CPU_TAR}"

# CUDA plugin (optional, only on Linux amd64 with nvidia-smi)
if [ "$OS" = "linux" ] && [ "$ARCH" = "amd64" ] && command -v nvidia-smi &> /dev/null; then
    CUDA_TAR="pjrt-cuda_${OS}-${ARCH}.tar.gz"
    CUDA_URL="https://github.com/zml/pjrt-artifacts/releases/download/${VERSION}/${CUDA_TAR}"
    
    echo "  → CUDA plugin: ${CUDA_URL}"
    curl -fsSL -o "/tmp/${CUDA_TAR}" "$CUDA_URL"
    tar -xzf "/tmp/${CUDA_TAR}" -C "$DEPS_DIR" --strip-components=1
    rm -f "/tmp/${CUDA_TAR}"
fi

# ROCm plugin (optional, only on Linux amd64 with rocminfo)
if [ "$OS" = "linux" ] && [ "$ARCH" = "amd64" ] && command -v rocminfo &> /dev/null; then
    ROCM_TAR="pjrt-rocm_${OS}-${ARCH}.tar.gz"
    ROCM_URL="https://github.com/zml/pjrt-artifacts/releases/download/${VERSION}/${ROCM_TAR}"
    
    echo "  → ROCm plugin: ${ROCM_URL}"
    curl -fsSL -o "/tmp/${ROCM_TAR}" "$ROCM_URL"
    tar -xzf "/tmp/${ROCM_TAR}" -C "$DEPS_DIR" --strip-components=1
    rm -f "/tmp/${ROCM_TAR}"
fi

echo ""
echo "PJRT plugins installed to $DEPS_DIR:"
ls -la "$DEPS_DIR"
