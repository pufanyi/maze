import argparse
import os

from dataset import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate maze video dataset")
    parser.add_argument("--rows", type=int, default=10, help="Maze logical rows")
    parser.add_argument("--cols", type=int, default=10, help="Maze logical cols")
    parser.add_argument("--count", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--resolution", type=int, default=256, help="Video resolution (square)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--duration", type=float, default=3.0, help="Video duration in seconds")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed")
    args = parser.parse_args()

    print(f"Generating {args.count} mazes ({args.rows}x{args.cols}) -> {args.output}")
    generate_dataset(
        rows=args.rows,
        cols=args.cols,
        count=args.count,
        output_dir=args.output,
        resolution=args.resolution,
        fps=args.fps,
        duration=args.duration,
        workers=args.workers,
        seed=args.seed,
    )
    print("Done!")


if __name__ == "__main__":
    main()
