import argparse
import os

from dataset import generate_search_videos_from_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Backfill BFS search videos for an existing dataset using metadata.jsonl",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata.jsonl from an existing dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Dataset output directory. Defaults to the metadata file's parent directory",
    )
    parser.add_argument(
        "--search-video-modes",
        type=str,
        default="unidirectional",
        help="Comma-separated search video modes: bidirectional,unidirectional",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Optional relative output subdirectory for generated videos, e.g. bfs_videos",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing search videos instead of skipping them",
    )
    parser.add_argument(
        "--default-resolution",
        type=int,
        default=256,
        help="Fallback resolution when metadata is missing the resolution field",
    )
    parser.add_argument(
        "--default-fps",
        type=int,
        default=24,
        help="Fallback FPS when metadata is missing the fps field",
    )
    parser.add_argument(
        "--default-duration",
        type=float,
        default=3.0,
        help="Fallback duration when metadata is missing the duration field",
    )
    args = parser.parse_args()

    search_video_modes = [mode.strip() for mode in args.search_video_modes.split(",")]

    print(f"Backfilling search videos from {args.metadata}")
    generate_search_videos_from_metadata(
        metadata_path=args.metadata,
        output_dir=args.output,
        search_video_modes=search_video_modes,
        workers=args.workers,
        overwrite=args.overwrite,
        default_resolution=args.default_resolution,
        default_fps=args.default_fps,
        default_duration=args.default_duration,
        output_subdir=args.output_subdir,
    )
    print("Done!")


if __name__ == "__main__":
    main()
