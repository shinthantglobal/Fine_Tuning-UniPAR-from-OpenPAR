#!/bin/bash

# Batch inference script for UniPAR
# Usage: ./run_inference.sh <checkpoint_path> <input_folder> <output_folder>

CHECKPOINT_PATH=$1
INPUT_FOLDER=$2
OUTPUT_FOLDER=$3

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$INPUT_FOLDER" ] || [ -z "$OUTPUT_FOLDER" ]; then
    echo "Usage: $0 <checkpoint_path> <input_folder> <output_folder>"
    echo "Example: $0 logs/pa100k_uniPAR_v2/2026-05-12_15_22_24/ckpt_2026-05-12_16_45_35_5.pth /path/to/images ./output"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

# Find all image files
find "$INPUT_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | while read -r img_path; do
    echo "Processing: $img_path"
    python inference.py --checkpoint "$CHECKPOINT_PATH" --image "$img_path" --output "$OUTPUT_FOLDER"
done

echo "Batch inference completed. Results saved to: $OUTPUT_FOLDER"