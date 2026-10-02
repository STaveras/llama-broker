#!/bin/bash
# Local discovery by default. Pass --download to opt into Hugging Face access.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${LLAMA_BROKER_MODELS_DIR:-${MODELS_DIR:-${HOME}/.local/share/models}}"

if [[ "${1:-}" != "--download" ]]; then
    exec env LLAMA_BROKER_MODELS_DIR="$MODELS_DIR" MODELS_DIR="$MODELS_DIR" python3 "$ROOT/sync-models.py"
fi

echo "=== Gemma Model Downloader ==="
echo ""
echo "Available Gemma 4 variants:"
echo ""
echo "1. Gemma 4 12B-it-GGUF (Unsloth quantization, recommended)"
echo "   Quantizations available: Q4_K_M, Q5_K_M, UD-Q5_K_XL, Q6_K, F16"
echo "   Context: 262,144 tokens (256K)"
echo ""
echo "2. Gemma 4 E2B-it-GGUF (MoE variant, smaller)"
echo "   Context: 262,144 tokens (256K)"
echo ""
echo "3. Gemma 4 E4B-it-GGUF (MoE variant, medium)"
echo "   Context: 262,144 tokens (256K)"
echo ""

echo "Which model to download? (1-3):"
read -r choice

case "$choice" in
    1)
        HF_REPO="unsloth/gemma-4-12b-it-GGUF"
        MODEL_PREFIX="gemma-4-12b-it"
        echo "Selecting quantization:"
        echo "  1) Q4_K_M.gguf (~7GB) - Recommended for testing"
        echo "  2) Q5_K_M.gguf (~8GB) - Better quality"
        echo "  3) UD-Q5_K_XL.gguf (~8GB) - Better quality, larger cache"
        echo "  4) Q6_K.gguf (~9GB) - High quality"
        echo "  5) F16.gguf (~14GB) - Full precision"
        read -r quant_choice
        case "$quant_choice" in
            1) QUANT="Q4_K_M" ;;
            2) QUANT="Q5_K_M" ;;
            3) QUANT="UD-Q5_K_XL" ;;
            4) QUANT="Q6_K" ;;
            5) QUANT="F16" ;;
            *)
                echo "Invalid quantization choice: $quant_choice" >&2
                exit 2
                ;;
        esac
        ;;
    2)
        HF_REPO="unsloth/gemma-4-E2B-it-GGUF"
        MODEL_PREFIX="gemma-4-E2B-it"
        QUANT="UD-Q5_K_XL"
        ;;
    3)
        HF_REPO="unsloth/gemma-4-E4B-it-GGUF"
        MODEL_PREFIX="gemma-4-E4B-it"
        QUANT="UD-Q5_K_XL"
        ;;
    *)
        echo "Invalid model choice: $choice" >&2
        exit 2
        ;;
esac

MODEL="${MODEL_PREFIX}-${QUANT}.gguf"
case "$choice" in
    1) MODEL_RELATIVE_DIR="gemma-4/12b" ;;
    2) MODEL_RELATIVE_DIR="gemma-4/e2b" ;;
    3) MODEL_RELATIVE_DIR="gemma-4/e4b" ;;
esac
CANONICAL_QUANT="${QUANT,,}"
CANONICAL_QUANT="${CANONICAL_QUANT//_/-}"
OUTPUT_FILE="$MODELS_DIR/$MODEL_RELATIVE_DIR/$CANONICAL_QUANT.gguf"
DOWNLOAD_DIR="$(dirname "$OUTPUT_FILE")"
DOWNLOAD_FILE="$DOWNLOAD_DIR/$MODEL"

echo ""
echo "Downloading: $HF_REPO/$MODEL"
echo "Output: $OUTPUT_FILE"
echo "Estimated size: $(case $MODEL in *Q4*) echo "~7GB" ;; *UD-Q5*) echo "~8GB" ;; *Q5*) echo "~8GB" ;; *Q6*) echo "~9GB" ;; *) echo "~14GB" ;; esac)"
echo ""
echo "This will take several minutes..."
echo "Press Ctrl+C to cancel"
echo ""

mkdir -p "$DOWNLOAD_DIR"

# Check if hf CLI is available (new huggingface_hub CLI)
if ! command -v hf &> /dev/null; then
    echo "error: the Hugging Face CLI ('hf') is not installed" >&2
    echo "Install it with pipx install huggingface_hub, then rerun this script." >&2
    exit 1
fi

# Download using new hf CLI
echo "Fetching weights from HuggingFace..."
hf download \
    --repo-type model \
    "$HF_REPO" \
    "$MODEL" \
    --local-dir "$DOWNLOAD_DIR"

if [ "$DOWNLOAD_FILE" != "$OUTPUT_FILE" ] && [ -f "$DOWNLOAD_FILE" ]; then
    mv -- "$DOWNLOAD_FILE" "$OUTPUT_FILE"
fi

echo ""
echo "Download complete!"
echo "Model location: $OUTPUT_FILE"
echo ""

# Verify download
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "✓ Model downloaded successfully"
    echo "  Size: $SIZE"
    echo ""
else
    echo "✗ Download failed or file not found"
    exit 1
fi
