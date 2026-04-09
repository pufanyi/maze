#!/bin/bash
set -e

OUTPUT_DIR="./data_easy"
ROWS=5
COLS=5
COUNT=100000
WORKERS=32
SEED=0
RESOLUTION=256
FPS=24
DURATION=10.0
SEARCH_VIDEO_MODES="bidirectional"

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
    --duration "$DURATION" \
    --search-video-modes "$SEARCH_VIDEO_MODES"

echo ""
echo "Done! Output in ${OUTPUT_DIR}/"
echo "  Videos: $(ls ${OUTPUT_DIR}/walk_videos/ | wc -l) walk"
if [ -d "${OUTPUT_DIR}/bfs_videos" ]; then
    echo "          $(ls ${OUTPUT_DIR}/bfs_videos/ | wc -l) bidirectional bfs"
fi
if [ -d "${OUTPUT_DIR}/unidirectional_bfs_videos" ]; then
    echo "          $(ls ${OUTPUT_DIR}/unidirectional_bfs_videos/ | wc -l) unidirectional bfs"
fi
echo "  Images: $(ls ${OUTPUT_DIR}/solution_images/ | wc -l) solutions"
echo "  Metadata: ${OUTPUT_DIR}/metadata.jsonl"
