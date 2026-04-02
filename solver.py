from collections import deque
from dataclasses import dataclass

import numpy as np


def solve_bfs(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """BFS shortest path on the grid. Returns list of (row, col) from start to end."""
    h, w = grid.shape
    visited = set()
    visited.add(start)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            break
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    # Reconstruct path
    path = []
    node: tuple[int, int] | None = end
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()
    return path


@dataclass
class BfsStep:
    """State at one step of bidirectional BFS."""
    forward_visited: set[tuple[int, int]]
    backward_visited: set[tuple[int, int]]
    forward_frontier: list[tuple[int, int]]
    backward_frontier: list[tuple[int, int]]
    forward_parent: dict[tuple[int, int], tuple[int, int] | None]
    backward_parent: dict[tuple[int, int], tuple[int, int] | None]
    meeting_point: tuple[int, int] | None


@dataclass
class BidirectionalBfsResult:
    steps: list[BfsStep]
    path: list[tuple[int, int]]


def solve_bidirectional_bfs(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> BidirectionalBfsResult:
    """Bidirectional BFS. Records each expansion step for animation."""
    h, w = grid.shape
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    fwd_visited: set[tuple[int, int]] = {start}
    bwd_visited: set[tuple[int, int]] = {end}
    fwd_parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    bwd_parent: dict[tuple[int, int], tuple[int, int] | None] = {end: None}
    fwd_frontier = deque([start])
    bwd_frontier = deque([end])

    steps: list[BfsStep] = []
    meeting: tuple[int, int] | None = None

    # Record initial state
    steps.append(BfsStep(
        forward_visited=set(fwd_visited),
        backward_visited=set(bwd_visited),
        forward_frontier=list(fwd_frontier),
        backward_frontier=list(bwd_frontier),
        forward_parent=dict(fwd_parent),
        backward_parent=dict(bwd_parent),
        meeting_point=None,
    ))

    def _expand_frontier(
        frontier: deque[tuple[int, int]],
        visited: set[tuple[int, int]],
        parent: dict[tuple[int, int], tuple[int, int] | None],
        other_visited: set[tuple[int, int]],
    ) -> tuple[deque[tuple[int, int]], tuple[int, int] | None]:
        """Expand one BFS layer. Returns (new_frontier, meeting_point)."""
        next_frontier: deque[tuple[int, int]] = deque()
        met: tuple[int, int] | None = None
        while frontier:
            r, c = frontier.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    next_frontier.append((nr, nc))
                    if met is None and (nr, nc) in other_visited:
                        met = (nr, nc)
        return next_frontier, met

    while fwd_frontier and bwd_frontier and meeting is None:
        # Expand both frontiers simultaneously (one layer each)
        fwd_frontier, fwd_met = _expand_frontier(
            fwd_frontier, fwd_visited, fwd_parent, bwd_visited,
        )
        bwd_frontier, bwd_met = _expand_frontier(
            bwd_frontier, bwd_visited, bwd_parent, fwd_visited,
        )
        meeting = fwd_met or bwd_met

        steps.append(BfsStep(
            forward_visited=set(fwd_visited),
            backward_visited=set(bwd_visited),
            forward_frontier=list(fwd_frontier),
            backward_frontier=list(bwd_frontier),
            forward_parent=dict(fwd_parent),
            backward_parent=dict(bwd_parent),
            meeting_point=meeting,
        ))

    # Reconstruct path through meeting point
    path: list[tuple[int, int]] = []
    if meeting is not None:
        # Forward half: start -> meeting
        fwd_half: list[tuple[int, int]] = []
        node: tuple[int, int] | None = meeting
        while node is not None:
            fwd_half.append(node)
            node = fwd_parent.get(node)
        fwd_half.reverse()

        # Backward half: meeting -> end
        bwd_half: list[tuple[int, int]] = []
        node = bwd_parent.get(meeting)
        while node is not None:
            bwd_half.append(node)
            node = bwd_parent.get(node)

        path = fwd_half + bwd_half

    return BidirectionalBfsResult(steps=steps, path=path)
