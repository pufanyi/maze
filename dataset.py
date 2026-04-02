import json
import os
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from generator import Maze, generate_maze
from renderer import render_bfs_frames, render_solution_image, render_walk_frames
from solver import solve_bfs, solve_bidirectional_bfs
from video import encode_video


def _generate_single(args: dict) -> dict:
    """Generate a single maze sample. Called by each worker."""
    sample_id: int = args["id"]
    rows: int = args["rows"]
    cols: int = args["cols"]
    seed: int = args["seed"]
    resolution: int = args["resolution"]
    fps: int = args["fps"]
    duration: float = args["duration"]
    output_dir: str = args["output_dir"]

    # Generate maze
    maze = generate_maze(rows, cols, seed)

    # Solve
    path = solve_bfs(maze.grid, maze.start, maze.end)
    bfs_result = solve_bidirectional_bfs(maze.grid, maze.start, maze.end)

    name = f"{sample_id:06d}"

    # Render and save walk video
    walk_frames = render_walk_frames(
        maze.grid, path, maze.start, maze.end, resolution, fps, duration,
    )
    walk_path = os.path.join(output_dir, "walk_videos", f"{name}.mp4")
    encode_video(walk_frames, walk_path, fps)

    # Render and save BFS video
    bfs_frames = render_bfs_frames(
        maze.grid, bfs_result.steps, bfs_result.path,
        maze.start, maze.end, resolution, fps, duration,
    )
    bfs_path = os.path.join(output_dir, "bfs_videos", f"{name}.mp4")
    encode_video(bfs_frames, bfs_path, fps)

    # Render and save solution image
    sol_img = render_solution_image(maze.grid, path, maze.start, maze.end, resolution)
    sol_path = os.path.join(output_dir, "solution_images", f"{name}.png")
    sol_img.convert("RGB").save(sol_path)

    # Build metadata
    metadata = {
        "id": sample_id,
        "maze_size": [rows, cols],
        "grid": maze.grid.tolist(),
        "start": list(maze.start),
        "end": list(maze.end),
        "path": [list(p) for p in path],
        "path_length": len(path),
        "seed": seed,
        "walk_video": f"walk_videos/{name}.mp4",
        "bfs_video": f"bfs_videos/{name}.mp4",
        "solution_image": f"solution_images/{name}.png",
        "fps": fps,
        "duration": duration,
        "resolution": resolution,
    }
    return metadata


def generate_dataset(
    rows: int,
    cols: int,
    count: int,
    output_dir: str,
    resolution: int = 256,
    fps: int = 24,
    duration: float = 3.0,
    workers: int = 1,
    seed: int = 0,
) -> None:
    """Generate the full maze dataset."""
    output = Path(output_dir)
    (output / "walk_videos").mkdir(parents=True, exist_ok=True)
    (output / "bfs_videos").mkdir(parents=True, exist_ok=True)
    (output / "solution_images").mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "id": i,
            "rows": rows,
            "cols": cols,
            "seed": seed + i,
            "resolution": resolution,
            "fps": fps,
            "duration": duration,
            "output_dir": str(output),
        }
        for i in range(count)
    ]

    metadata_path = output / "metadata.jsonl"
    results: list[dict] = []

    if workers <= 1:
        for task in tqdm(tasks, desc="Generating mazes"):
            results.append(_generate_single(task))
    else:
        with Pool(processes=workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_generate_single, tasks),
                total=count,
                desc="Generating mazes",
            ):
                results.append(result)

    # Sort by id and write metadata
    results.sort(key=lambda x: x["id"])
    with open(metadata_path, "w") as f:
        for meta in results:
            f.write(json.dumps(meta) + "\n")
