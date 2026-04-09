import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from generator import generate_maze
from renderer import (
    render_bfs_frames,
    render_distance_bfs_frames,
    render_pruning_frames,
    render_solution_image,
    render_walk_frames,
)
from solver import (
    compute_pruning_order,
    solve_bfs,
    solve_bfs_with_steps,
    solve_bidirectional_bfs,
)
from video import encode_video


SEARCH_VIDEO_MODES = {"bidirectional", "unidirectional"}


def _normalize_search_video_modes(search_video_modes: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for mode in search_video_modes:
        value = mode.strip().lower()
        if not value:
            continue
        if value not in SEARCH_VIDEO_MODES:
            raise ValueError(
                f"Unsupported search video mode: {mode}. "
                f"Expected one of: {', '.join(sorted(SEARCH_VIDEO_MODES))}"
            )
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one search video mode must be provided.")
    return normalized


def _search_video_output_subdirs(
    search_video_modes: list[str],
    output_subdir: str | None = None,
) -> dict[str, str]:
    if output_subdir is None:
        return {
            "bidirectional": "bfs_videos",
            "unidirectional": "unidirectional_bfs_videos",
        }

    if len(search_video_modes) != 1:
        raise ValueError("Custom output_subdir is only supported when generating exactly one mode.")

    mode = search_video_modes[0]
    return {mode: output_subdir}


def _ensure_search_video_dirs(
    output_dir: Path,
    search_video_modes: list[str],
    output_subdir: str | None = None,
) -> None:
    output_subdirs = _search_video_output_subdirs(search_video_modes, output_subdir=output_subdir)
    for mode in search_video_modes:
        (output_dir / output_subdirs[mode]).mkdir(parents=True, exist_ok=True)


def _search_video_metadata_fields(
    name: str,
    search_video_modes: list[str],
    output_subdir: str | None = None,
) -> dict[str, str]:
    output_subdirs = _search_video_output_subdirs(search_video_modes, output_subdir=output_subdir)
    fields: dict[str, str] = {}
    if "bidirectional" in search_video_modes:
        video_path = f"{output_subdirs['bidirectional']}/{name}.mp4"
        fields["bfs_video"] = video_path
        fields["bidirectional_bfs_video"] = video_path
    if "unidirectional" in search_video_modes:
        fields["unidirectional_bfs_video"] = f"{output_subdirs['unidirectional']}/{name}.mp4"
    return fields


def _merge_search_video_modes(
    existing_modes: list[str] | tuple[str, ...] | str | None,
    new_modes: list[str] | tuple[str, ...],
) -> list[str]:
    merged: list[str] = []
    for source in (existing_modes, new_modes):
        if source is None:
            continue
        if isinstance(source, str):
            items = [source]
        else:
            items = list(source)
        for mode in items:
            value = mode.strip().lower()
            if value and value in SEARCH_VIDEO_MODES and value not in merged:
                merged.append(value)
    return merged


def _sample_name(metadata: dict) -> str:
    for key in (
        "walk_video",
        "solution_image",
        "bfs_video",
        "bidirectional_bfs_video",
        "unidirectional_bfs_video",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem

    sample_id = metadata.get("id")
    if sample_id is None:
        raise ValueError("Metadata record is missing both asset paths and id.")
    return f"{int(sample_id):06d}"


def _render_search_videos(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    resolution: int,
    fps: int,
    duration: float,
    output_dir: str,
    name: str,
    search_video_modes: list[str],
    final_path: list[tuple[int, int]] | None = None,
    overwrite: bool = True,
    output_subdir: str | None = None,
) -> dict[str, str]:
    metadata_fields = _search_video_metadata_fields(
        name,
        search_video_modes,
        output_subdir=output_subdir,
    )
    output = Path(output_dir)

    if "bidirectional" in search_video_modes:
        bfs_path = output / metadata_fields["bfs_video"]
        if overwrite or not bfs_path.exists():
            bfs_result = solve_bidirectional_bfs(grid, start, end)
            bfs_frames = render_bfs_frames(
                grid,
                bfs_result.steps,
                final_path or bfs_result.path,
                start,
                end,
                resolution,
                fps,
                duration,
            )
            encode_video(bfs_frames, str(bfs_path), fps)

    if "unidirectional" in search_video_modes:
        unidirectional_path = output / metadata_fields["unidirectional_bfs_video"]
        if overwrite or not unidirectional_path.exists():
            unidirectional_result = solve_bfs_with_steps(grid, start, end)
            unidirectional_frames = render_bfs_frames(
                grid,
                unidirectional_result.steps,
                final_path or unidirectional_result.path,
                start,
                end,
                resolution,
                fps,
                duration,
            )
            encode_video(unidirectional_frames, str(unidirectional_path), fps)

    return metadata_fields


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

    name = f"{sample_id:06d}"

    # --- Phase 1: Distance-colored BFS ---
    bfs_result = solve_bfs_with_steps(maze.grid, maze.start, maze.end)
    dist_bfs_frames = render_distance_bfs_frames(
        maze.grid, bfs_result.steps, maze.start, maze.end, resolution, fps, duration,
    )
    dist_bfs_path = os.path.join(output_dir, "distance_bfs_videos", f"{name}.mp4")
    encode_video(dist_bfs_frames, dist_bfs_path, fps)

    # --- Phase 2: BFS + Pruning animation ---
    final_step = bfs_result.steps[-1]
    pruning_layers = compute_pruning_order(final_step.forward_parent, path)
    prune_only_frames = render_pruning_frames(
        maze.grid,
        final_step.forward_parent,
        final_step.forward_dist,
        pruning_layers,
        path,
        maze.start,
        maze.end,
        resolution,
        fps,
        duration,
    )
    pruning_path = os.path.join(output_dir, "pruning_videos", f"{name}.mp4")
    encode_video(dist_bfs_frames + prune_only_frames, pruning_path, fps)

    # --- Phase 3: Ball walk video ---
    walk_frames = render_walk_frames(
        maze.grid, path, maze.start, maze.end, resolution, fps, duration,
    )
    walk_path = os.path.join(output_dir, "walk_videos", f"{name}.mp4")
    encode_video(walk_frames, walk_path, fps)

    # --- Solution image ---
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
        "distance_bfs_video": f"distance_bfs_videos/{name}.mp4",
        "pruning_video": f"pruning_videos/{name}.mp4",
        "walk_video": f"walk_videos/{name}.mp4",
        "solution_image": f"solution_images/{name}.png",
        "fps": fps,
        "duration": duration,
        "resolution": resolution,
    }
    return metadata


def _backfill_search_video_single(args: dict) -> dict:
    metadata: dict = dict(args["metadata"])
    output_dir = args["output_dir"]
    search_video_modes: list[str] = args["search_video_modes"]
    overwrite: bool = args["overwrite"]
    default_resolution: int = args["default_resolution"]
    default_fps: int = args["default_fps"]
    default_duration: float = args["default_duration"]
    output_subdir: str | None = args["output_subdir"]

    grid = np.array(metadata["grid"], dtype=np.uint8)
    start = tuple(metadata["start"])
    end = tuple(metadata["end"])
    path = [tuple(p) for p in metadata.get("path", [])]
    if not path:
        path = solve_bfs(grid, start, end)
        metadata["path"] = [list(p) for p in path]
        metadata["path_length"] = len(path)

    resolution = int(metadata.get("resolution", default_resolution))
    fps = int(metadata.get("fps", default_fps))
    duration = float(metadata.get("duration", default_duration))
    name = _sample_name(metadata)

    metadata.update(_render_search_videos(
        grid,
        start,
        end,
        resolution,
        fps,
        duration,
        output_dir,
        name,
        search_video_modes,
        final_path=path,
        overwrite=overwrite,
        output_subdir=output_subdir,
    ))
    metadata["resolution"] = resolution
    metadata["fps"] = fps
    metadata["duration"] = duration
    metadata["search_video_modes"] = _merge_search_video_modes(
        metadata.get("search_video_modes"),
        search_video_modes,
    )
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
    (output / "distance_bfs_videos").mkdir(parents=True, exist_ok=True)
    (output / "pruning_videos").mkdir(parents=True, exist_ok=True)
    (output / "walk_videos").mkdir(parents=True, exist_ok=True)
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


def generate_search_videos_from_metadata(
    metadata_path: str,
    output_dir: str | None = None,
    search_video_modes: list[str] | tuple[str, ...] = ("unidirectional",),
    workers: int = 1,
    overwrite: bool = False,
    default_resolution: int = 256,
    default_fps: int = 24,
    default_duration: float = 3.0,
    output_subdir: str | None = None,
) -> None:
    """Backfill search videos for an existing dataset using metadata.jsonl."""
    search_video_modes = _normalize_search_video_modes(search_video_modes)
    metadata_file = Path(metadata_path)
    output = Path(output_dir) if output_dir is not None else metadata_file.parent

    _ensure_search_video_dirs(output, search_video_modes, output_subdir=output_subdir)

    with open(metadata_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    tasks = [
        {
            "metadata": record,
            "output_dir": str(output),
            "search_video_modes": search_video_modes,
            "overwrite": overwrite,
            "default_resolution": default_resolution,
            "default_fps": default_fps,
            "default_duration": default_duration,
            "output_subdir": output_subdir,
        }
        for record in records
    ]

    results: list[dict] = []
    if workers <= 1:
        for task in tqdm(tasks, desc="Backfilling search videos"):
            results.append(_backfill_search_video_single(task))
    else:
        with Pool(processes=workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_backfill_search_video_single, tasks),
                total=len(tasks),
                desc="Backfilling search videos",
            ):
                results.append(result)

    results.sort(key=lambda x: x.get("id", 0))
    with open(metadata_file, "w") as f:
        for meta in results:
            f.write(json.dumps(meta) + "\n")
