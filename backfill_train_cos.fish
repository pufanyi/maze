#!/usr/bin/env fish

function usage
    echo "Usage: backfill_train_cos.fish [options]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE           Search video mode: unidirectional | bidirectional (default: unidirectional)"
    echo "  -d, --search-dir DIR      Relative output directory for search videos"
    echo "                           Default: unidirectional_bfs_videos or bidirectional_bfs_videos"
    echo "  -j, --train-json FILE     Output train json filename in each dataset dir"
    echo "                           Default: train_cos_unidirectional.json or train_cos_bidirectional.json"
    echo "  -w, --workers N           Parallel workers for video generation (default: CPU count)"
    echo "  -s, --datasets LIST       Comma-separated dataset directories"
    echo "  -p, --prompt TEXT         Prompt text for train json"
    echo "  -n, --no-overwrite        Skip overwriting existing search videos"
    echo "  -h, --help                Show this help"
end

argparse h/help m/mode= d/search-dir= j/train-json= w/workers= s/datasets= p/prompt= n/no-overwrite -- $argv
or begin
    usage
    exit 1
end

if set -q _flag_help
    usage
    exit 0
end

set -l script_dir (cd (dirname (status --current-filename)); and pwd)

set -l mode unidirectional
set -l search_dir ""
set -l train_json ""
set -l prompt "A ball navigates through a maze from the top-left corner to the bottom-right corner."
set -l overwrite 1
set -l datasets data

set -l workers 8
if command -q nproc
    set workers (nproc)
else if command -q getconf
    set workers (getconf _NPROCESSORS_ONLN)
end

if set -q _flag_mode
    set mode $_flag_mode
end
if set -q _flag_search_dir
    set search_dir $_flag_search_dir
end
if set -q _flag_train_json
    set train_json $_flag_train_json
end
if set -q _flag_workers
    set workers $_flag_workers
end
if set -q _flag_prompt
    set prompt $_flag_prompt
end
if set -q _flag_datasets
    set datasets (string split ',' -- $_flag_datasets)
end
if set -q _flag_no_overwrite
    set overwrite 0
end

if not contains -- $mode unidirectional bidirectional
    echo "Unsupported mode: $mode" >&2
    usage
    exit 1
end

set -l search_video_field unidirectional_bfs_video
if test "$mode" = "bidirectional"
    set search_video_field bidirectional_bfs_video
end

if test -z "$search_dir"
    set search_dir "$mode"_bfs_videos
end
if test -z "$train_json"
    set train_json "train_cos_$mode.json"
end

for dataset in $datasets
    set -l metadata "$dataset/metadata.jsonl"
    if not test -f "$metadata"
        echo "Missing metadata file: $metadata" >&2
        exit 1
    end
end

echo "Mode: $mode"
echo "Search dir: $search_dir"
echo "Train json: $train_json"
echo "Workers: $workers"
echo "Datasets: "(string join ', ' -- $datasets)
echo ""

for dataset in $datasets
    set -l metadata "$dataset/metadata.jsonl"
    set -l train_json_path "$dataset/$train_json"

    echo "[$dataset] Backfilling search videos"
    set -l backfill_cmd env UV_CACHE_DIR=/tmp/uv-cache uv run python "$script_dir/backfill_search_videos.py" \
        --metadata "$metadata" \
        --workers "$workers" \
        --search-video-modes "$mode" \
        --output-subdir "$search_dir"
    if test $overwrite -eq 1
        set backfill_cmd $backfill_cmd --overwrite
    end
    $backfill_cmd
    or exit $status

    echo "[$dataset] Building $train_json"
    env UV_CACHE_DIR=/tmp/uv-cache uv run python "$script_dir/build_train_cos.py" \
        --metadata "$metadata" \
        --output "$train_json_path" \
        --search-video-field "$search_video_field" \
        --prompt "$prompt"
    or exit $status

    echo "[$dataset] Done"
    echo ""
end

echo "All datasets finished."
