"""Generate 128 test samples: first frame + solution image only (no videos)."""

import json
import os
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from generator import generate_maze
from renderer import render_maze_base, render_walk_frame, render_solution_image
from solver import solve_bfs


def _generate_single(args: dict) -> dict:
    sample_id: int = args["id"]
    rows: int = args["rows"]
    cols: int = args["cols"]
    seed: int = args["seed"]
    resolution: int = args["resolution"]
    output_dir: str = args["output_dir"]

    maze = generate_maze(rows, cols, seed)
    path = solve_bfs(maze.grid, maze.start, maze.end)

    name = f"{sample_id:06d}"

    # First frame: maze base with ball at start
    base, layout = render_maze_base(maze.grid, resolution, maze.start, maze.end)
    first_frame = render_walk_frame(base, layout, (float(maze.start[0]), float(maze.start[1])))
    first_frame_path = os.path.join(output_dir, "first_frames", f"{name}.png")
    first_frame.convert("RGB").save(first_frame_path)

    # Solution image
    sol_img = render_solution_image(maze.grid, path, maze.start, maze.end, resolution)
    sol_path = os.path.join(output_dir, "solution_images", f"{name}.png")
    sol_img.convert("RGB").save(sol_path)

    return {
        "id": sample_id,
        "maze_size": [rows, cols],
        "grid": maze.grid.tolist(),
        "start": list(maze.start),
        "end": list(maze.end),
        "path": [list(p) for p in path],
        "path_length": len(path),
        "seed": seed,
        "first_frame": f"first_frames/{name}.png",
        "solution_image": f"solution_images/{name}.png",
        "resolution": resolution,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate test data (first frame + solution only)")
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=40)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--output", type=str, default="./test_data_2")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=1000000)
    args = parser.parse_args()

    output = Path(args.output)
    (output / "first_frames").mkdir(parents=True, exist_ok=True)
    (output / "solution_images").mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "id": i,
            "rows": args.rows,
            "cols": args.cols,
            "seed": args.seed + i,
            "resolution": args.resolution,
            "output_dir": str(output),
        }
        for i in range(args.count)
    ]

    results: list[dict] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="Generating test data"):
            results.append(_generate_single(task))
    else:
        with Pool(processes=args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_generate_single, tasks),
                total=args.count,
                desc="Generating test data",
            ):
                results.append(result)

    results.sort(key=lambda x: x["id"])
    metadata_path = output / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for meta in results:
            f.write(json.dumps(meta) + "\n")

    print(f"Done! {len(results)} samples in {args.output}/")
    print(f"  First frames: {output / 'first_frames'}")
    print(f"  Solutions:     {output / 'solution_images'}")
    print(f"  Metadata:      {metadata_path}")


if __name__ == "__main__":
    main()
