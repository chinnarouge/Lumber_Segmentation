#!/bin/bash
set -e

if [ $# -lt 2 ]; then
    echo "Usage: ./run_inference.sh <input_dir> <output_dir>"
    echo ""
    echo "  input_dir   Directory containing NIfTI CT scans (.nii.gz or .nii)"
    echo "  output_dir  Directory for output segmentations and centroid JSONs"
    echo ""
    echo "Option 1 (Docker - recommended):"
    echo "  docker build -t lumbar-seg ."
    echo "  ./run_inference.sh <input_dir> <output_dir>"
    echo ""
    echo "Option 2 (local Python 3.10+, CUDA GPU recommended):"
    echo "  pip install -r requirements.txt"
    echo "  ./run_inference.sh <input_dir> <output_dir>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="$(cd "$1" && pwd)"
OUTPUT_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"

if docker image inspect lumbar-seg >/dev/null 2>&1; then
    echo "Running with Docker..."
    docker run --rm --gpus all \
        -v "$INPUT_DIR":/input:ro \
        -v "$OUTPUT_DIR":/output \
        lumbar-seg /input /output
else
    echo "Docker image not found, running with local Python..."
    python "$SCRIPT_DIR/src/inference.py" "$INPUT_DIR" "$OUTPUT_DIR"
fi
