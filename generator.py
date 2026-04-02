from dataclasses import dataclass

import numpy as np


@dataclass
class Maze:
    grid: np.ndarray  # 2D array, 1=wall, 0=path
    start: tuple[int, int]  # (row, col) in grid coordinates
    end: tuple[int, int]
    rows: int  # logical maze size
    cols: int
    seed: int


def generate_maze(rows: int, cols: int, seed: int) -> Maze:
    """Generate a perfect maze using DFS recursive backtracker.

    The grid has size (2*rows+1, 2*cols+1).
    Logical cells are at odd indices: (2*r+1, 2*c+1) for r in [0, rows), c in [0, cols).
    Walls are at even indices.
    """
    rng = np.random.default_rng(seed)
    h, w = 2 * rows + 1, 2 * cols + 1
    grid = np.ones((h, w), dtype=np.uint8)

    # Open all logical cells
    for r in range(rows):
        for c in range(cols):
            grid[2 * r + 1, 2 * c + 1] = 0

    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Iterative DFS to avoid recursion limit
    stack: list[tuple[int, int]] = []
    sr, sc = 0, 0
    visited[sr, sc] = True
    stack.append((sr, sc))

    while stack:
        cr, cc = stack[-1]
        neighbors = []
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                neighbors.append((nr, nc))

        if neighbors:
            idx = rng.integers(len(neighbors))
            nr, nc = neighbors[idx]
            # Remove wall between current and neighbor
            wall_r = 2 * cr + 1 + (nr - cr)
            wall_c = 2 * cc + 1 + (nc - cc)
            grid[wall_r, wall_c] = 0
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    start = (1, 1)
    end = (2 * rows - 1, 2 * cols - 1)

    return Maze(grid=grid, start=start, end=end, rows=rows, cols=cols, seed=seed)
