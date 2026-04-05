#!/bin/bash
set -e

OUTPUT_DIR="./data_2"
ROWS=40
COLS=40
COUNT=100000
WORKERS=8
SEED=0
RESOLUTION=512
FPS=24
DURATION=10.0

echo "Generating ${COUNT} mazes (${ROWS}x${COLS}) with ${WORKERS} workers"
echo "Output: ${OUTPUT_DIR}"
echo ""

uv run python main.py \
    --rows "$ROWS" \
    --cols "$COLS" \
    --count "$COUNT" \
    --output "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --resolution "$RESOLUTION" \
    --fps "$FPS" \
    --duration "$DURATION"

echo ""
echo "Done! Output in ${OUTPUT_DIR}/"
echo "  Videos: $(ls ${OUTPUT_DIR}/walk_videos/ | wc -l) walk + $(ls ${OUTPUT_DIR}/bfs_videos/ | wc -l) bfs"
echo "  Images: $(ls ${OUTPUT_DIR}/solution_images/ | wc -l) solutions"
echo "  Metadata: ${OUTPUT_DIR}/metadata.jsonl"
