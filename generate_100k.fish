#!/usr/bin/env fish

set -e

set output_base ./data_100k
set count 100000
set duration 10
set fps 24
set resolution 256
set workers (nproc)

echo "Workers: $workers"

echo "=== Generating easy (5x5) ==="
uv run python main.py \
    --rows 5 --cols 5 \
    --count $count \
    --output $output_base/easy_5x5 \
    --duration $duration \
    --fps $fps \
    --resolution $resolution \
    --workers $workers

echo "=== Generating medium (10x10) ==="
uv run python main.py \
    --rows 10 --cols 10 \
    --count $count \
    --output $output_base/medium_10x10 \
    --duration $duration \
    --fps $fps \
    --resolution $resolution \
    --workers $workers

echo "=== Generating hard (20x20) ==="
uv run python main.py \
    --rows 20 --cols 20 \
    --count $count \
    --output $output_base/hard_20x20 \
    --duration $duration \
    --fps $fps \
    --resolution $resolution \
    --workers $workers

echo "=== Generating xhard (30x30) ==="
uv run python main.py \
    --rows 30 --cols 30 \
    --count $count \
    --output $output_base/xhard_30x30 \
    --duration $duration \
    --fps $fps \
    --resolution $resolution \
    --workers $workers

echo "=== All done ==="
