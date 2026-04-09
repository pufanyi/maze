import colorsys

import numpy as np
from PIL import Image, ImageDraw

from solver import BfsStep

# Color palette
WALL_COLOR = (44, 62, 80)         # #2C3E50
WALL_SHADOW = (30, 42, 54)       # darker wall for shadow
FLOOR_COLOR = (236, 240, 241)    # #ECF0F1
BALL_COLOR = (231, 76, 60)       # #E74C3C
BALL_HIGHLIGHT = (255, 180, 170) # light highlight
START_COLOR = (46, 204, 113)     # #2ECC71 green
END_COLOR = (241, 196, 15)       # #F1C40F gold
PATH_START_COLOR = (46, 204, 113)
PATH_END_COLOR = (231, 76, 60)
FWD_COLOR = (52, 152, 219)       # #3498DB blue
FWD_FRONTIER_COLOR = (41, 128, 185)
BWD_COLOR = (230, 126, 34)       # #E67E22 orange
BWD_FRONTIER_COLOR = (211, 84, 0)
MEETING_COLOR = (155, 89, 182)   # #9B59B6 purple

WALL_RATIO = 0.25  # wall thickness relative to path cell size


class MazeLayout:
    """Handles coordinate mapping between grid and pixel space with thin walls."""

    def __init__(self, grid_shape: tuple[int, ...], resolution: int, wall_ratio: float = WALL_RATIO):
        h, w = grid_shape
        # h = 2*rows+1, w = 2*cols+1
        rows = (h - 1) // 2
        cols = (w - 1) // 2

        # Compute path_size such that maze fits in resolution
        # total = (rows+1)*wall_ratio*path_size + rows*path_size
        h_units = (rows + 1) * wall_ratio + rows
        w_units = (cols + 1) * wall_ratio + cols
        self.path_size = resolution * 0.95 / max(h_units, w_units)
        self.wall_size = self.path_size * wall_ratio

        # Total maze pixel dimensions
        self.maze_h = (rows + 1) * self.wall_size + rows * self.path_size
        self.maze_w = (cols + 1) * self.wall_size + cols * self.path_size
        self.offset_x = (resolution - self.maze_w) / 2
        self.offset_y = (resolution - self.maze_h) / 2
        self.resolution = resolution
        self.grid_h = h
        self.grid_w = w

        # Precompute row/col pixel positions and sizes
        self._row_pos = []  # (y_start, height) for each grid row
        self._col_pos = []  # (x_start, width) for each grid col
        y = self.offset_y
        for r in range(h):
            sz = self.wall_size if r % 2 == 0 else self.path_size
            self._row_pos.append((y, sz))
            y += sz
        x = self.offset_x
        for c in range(w):
            sz = self.wall_size if c % 2 == 0 else self.path_size
            self._col_pos.append((x, sz))
            x += sz

    def cell_rect(self, r: int, c: int) -> tuple[float, float, float, float]:
        """Return (x0, y0, x1, y1) pixel rect for grid cell (r, c)."""
        y0, h = self._row_pos[r]
        x0, w = self._col_pos[c]
        return x0, y0, x0 + w, y0 + h

    def cell_center(self, r: float, c: float) -> tuple[float, float]:
        """Return pixel center for a (possibly fractional) grid position."""
        # Integer part
        ri, ci = int(r), int(c)
        fr, fc = r - ri, c - ci

        # Clamp
        ri = max(0, min(ri, self.grid_h - 1))
        ci = max(0, min(ci, self.grid_w - 1))

        y0, h0 = self._row_pos[ri]
        x0, w0 = self._col_pos[ci]
        cy = y0 + h0 / 2
        cx = x0 + w0 / 2

        # Fractional interpolation toward next cell
        if fr > 0 and ri + 1 < self.grid_h:
            y1, h1 = self._row_pos[ri + 1]
            cy_next = y1 + h1 / 2
            cy = cy + fr * (cy_next - cy)
        if fc > 0 and ci + 1 < self.grid_w:
            x1, w1 = self._col_pos[ci + 1]
            cx_next = x1 + w1 / 2
            cx = cx + fc * (cx_next - cx)

        return cx, cy


def render_maze_base(
    grid: np.ndarray,
    resolution: int,
    start: tuple[int, int] | None = None,
    end: tuple[int, int] | None = None,
) -> tuple[Image.Image, MazeLayout]:
    """Render the maze base (walls + floor + start/end markers). Returns (image, layout)."""
    h, w = grid.shape
    layout = MazeLayout(grid.shape, resolution)

    img = Image.new("RGBA", (resolution, resolution), FLOOR_COLOR + (255,))
    draw = ImageDraw.Draw(img)

    shadow_offset = max(1, layout.wall_size * 0.15)

    # Draw wall shadows first
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 1:
                x0, y0, x1, y1 = layout.cell_rect(r, c)
                draw.rectangle(
                    [x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset],
                    fill=WALL_SHADOW + (60,),
                )

    # Draw walls
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 1:
                x0, y0, x1, y1 = layout.cell_rect(r, c)
                draw.rectangle([x0, y0, x1, y1], fill=WALL_COLOR + (255,))

    # Draw start/end markers
    marker_radius = layout.path_size * 0.3
    if start is not None:
        sx, sy = layout.cell_center(start[0], start[1])
        draw.ellipse(
            [sx - marker_radius, sy - marker_radius, sx + marker_radius, sy + marker_radius],
            fill=START_COLOR + (150,),
        )
    if end is not None:
        ex, ey = layout.cell_center(end[0], end[1])
        draw.ellipse(
            [ex - marker_radius, ey - marker_radius, ex + marker_radius, ey + marker_radius],
            fill=END_COLOR + (150,),
        )

    return img, layout


def _draw_ball(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float) -> None:
    """Draw a ball with gradient-like effect and shadow."""
    # Shadow
    sr = radius * 0.9
    sx, sy = cx + radius * 0.2, cy + radius * 0.3
    draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(40, 40, 40, 60))

    # Main ball
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=BALL_COLOR + (255,))

    # Inner gradient ring (lighter)
    inner_r = radius * 0.7
    draw.ellipse(
        [cx - inner_r, cy - inner_r - radius * 0.1, cx + inner_r, cy + inner_r - radius * 0.1],
        fill=(241, 106, 90, 120),
    )

    # Highlight
    hr = radius * 0.3
    hx, hy = cx - radius * 0.25, cy - radius * 0.25
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=BALL_HIGHLIGHT + (180,))


def render_walk_frame(
    base_img: Image.Image,
    layout: MazeLayout,
    ball_pos: tuple[float, float],
) -> Image.Image:
    """Render a single frame with the ball at a given grid position (can be fractional)."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    bx, by = layout.cell_center(ball_pos[0], ball_pos[1])
    radius = layout.path_size * 0.35
    _draw_ball(draw, bx, by, radius)

    return img


def render_walk_frames(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    resolution: int,
    fps: int,
    duration: float,
) -> list[Image.Image]:
    """Render all frames for the ball walking along the path."""
    total_frames = int(fps * duration)
    if total_frames < 2:
        total_frames = 2

    base, layout = render_maze_base(grid, resolution, start, end)

    num_steps = len(path) - 1
    if num_steps <= 0:
        return [render_walk_frame(base, layout, (float(path[0][0]), float(path[0][1])))]

    frames: list[Image.Image] = []
    for f in range(total_frames):
        t = f / (total_frames - 1)  # 0.0 to 1.0
        step_float = t * num_steps
        step_idx = min(int(step_float), num_steps - 1)
        frac = step_float - step_idx

        r0, c0 = path[step_idx]
        r1, c1 = path[step_idx + 1]
        ball_r = r0 + frac * (r1 - r0)
        ball_c = c0 + frac * (c1 - c0)

        frame = render_walk_frame(base, layout, (ball_r, ball_c))
        frames.append(frame)

    return frames


def render_solution_image(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    resolution: int,
) -> Image.Image:
    """Render the maze with the solution path drawn as a gradient line."""
    img, layout = render_maze_base(grid, resolution, start, end)
    draw = ImageDraw.Draw(img)

    if len(path) < 2:
        return img

    line_width = max(2, int(layout.path_size * 0.35))
    n = len(path) - 1
    for i in range(n):
        t = i / n
        r = int(PATH_START_COLOR[0] * (1 - t) + PATH_END_COLOR[0] * t)
        g = int(PATH_START_COLOR[1] * (1 - t) + PATH_END_COLOR[1] * t)
        b = int(PATH_START_COLOR[2] * (1 - t) + PATH_END_COLOR[2] * t)

        x0, y0 = layout.cell_center(path[i][0], path[i][1])
        x1, y1 = layout.cell_center(path[i + 1][0], path[i + 1][1])
        draw.line([(x0, y0), (x1, y1)], fill=(r, g, b, 220), width=line_width)

    dot_r = layout.path_size * 0.25
    sx, sy = layout.cell_center(path[0][0], path[0][1])
    draw.ellipse([sx - dot_r, sy - dot_r, sx + dot_r, sy + dot_r], fill=START_COLOR + (255,))
    ex, ey = layout.cell_center(path[-1][0], path[-1][1])
    draw.ellipse([ex - dot_r, ey - dot_r, ex + dot_r, ey + dot_r], fill=END_COLOR + (255,))

    return img


def _render_bfs_overlay(
    base_img: Image.Image,
    layout: MazeLayout,
    step: BfsStep,
    final_path: list[tuple[int, int]] | None = None,
) -> Image.Image:
    """Render one frame of the BFS exploration."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    line_width = max(2, int(layout.path_size * 0.4))
    dot_radius = layout.path_size * 0.2

    # Draw forward explored edges (cell -> parent)
    for node, parent in step.forward_parent.items():
        if parent is None:
            continue
        x0, y0 = layout.cell_center(parent[0], parent[1])
        x1, y1 = layout.cell_center(node[0], node[1])
        draw.line([(x0, y0), (x1, y1)], fill=FWD_COLOR + (180,), width=line_width)

    # Draw backward explored edges
    for node, parent in step.backward_parent.items():
        if parent is None:
            continue
        x0, y0 = layout.cell_center(parent[0], parent[1])
        x1, y1 = layout.cell_center(node[0], node[1])
        draw.line([(x0, y0), (x1, y1)], fill=BWD_COLOR + (180,), width=line_width)

    # Highlight forward frontier
    for node in step.forward_frontier:
        x, y = layout.cell_center(node[0], node[1])
        draw.ellipse(
            [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
            fill=FWD_FRONTIER_COLOR + (220,),
        )

    # Highlight backward frontier
    for node in step.backward_frontier:
        x, y = layout.cell_center(node[0], node[1])
        draw.ellipse(
            [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
            fill=BWD_FRONTIER_COLOR + (220,),
        )

    # If meeting point found, highlight final path
    if final_path and step.meeting_point is not None:
        path_width = max(3, int(layout.path_size * 0.45))
        for i in range(len(final_path) - 1):
            x0, y0 = layout.cell_center(final_path[i][0], final_path[i][1])
            x1, y1 = layout.cell_center(final_path[i + 1][0], final_path[i + 1][1])
            draw.line([(x0, y0), (x1, y1)], fill=MEETING_COLOR + (240,), width=path_width)

        mx, my = layout.cell_center(step.meeting_point[0], step.meeting_point[1])
        mr = layout.path_size * 0.3
        draw.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=MEETING_COLOR + (255,))

    return img


def render_bfs_frames(
    grid: np.ndarray,
    steps: list[BfsStep],
    path: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    resolution: int,
    fps: int,
    duration: float,
) -> list[Image.Image]:
    """Render all frames for a BFS exploration video."""
    total_frames = int(fps * duration)
    if total_frames < 2:
        total_frames = 2

    base, layout = render_maze_base(grid, resolution, start, end)
    num_steps = len(steps)

    if num_steps == 0:
        return [base.copy()]

    frames: list[Image.Image] = []

    # Reserve last 20% of frames for showing the final path
    explore_frames = int(total_frames * 0.8)
    final_frames = total_frames - explore_frames

    for f in range(explore_frames):
        t = f / max(1, explore_frames - 1)
        step_idx = min(int(t * (num_steps - 1)), num_steps - 1)
        step = steps[step_idx]
        show_path = step.meeting_point is not None and step_idx == num_steps - 1
        frame = _render_bfs_overlay(
            base, layout, step,
            final_path=path if show_path else None,
        )
        frames.append(frame)

    if steps:
        last_step = steps[-1]
        for _ in range(final_frames):
            frame = _render_bfs_overlay(base, layout, last_step, final_path=path)
            frames.append(frame)

    return frames


# --- Distance-colored BFS and pruning ---

def _distance_color(dist: int, max_dist: int) -> tuple[int, int, int]:
    """Map BFS distance to a color via HSV hue rotation (blue -> red)."""
    if max_dist <= 0:
        return (52, 152, 219)
    t = min(dist / max_dist, 1.0)
    hue = 0.66 * (1.0 - t)  # 0.66 (blue) -> 0.0 (red)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _render_distance_bfs_overlay(
    base_img: Image.Image,
    layout: MazeLayout,
    step: BfsStep,
    max_dist: int,
) -> Image.Image:
    """Render one frame of distance-colored BFS exploration."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    line_width = max(2, int(layout.path_size * 0.4))
    dot_radius = layout.path_size * 0.2

    for node, parent in step.forward_parent.items():
        if parent is None:
            continue
        dist = step.forward_dist.get(node, 0)
        color = _distance_color(dist, max_dist)
        x0, y0 = layout.cell_center(parent[0], parent[1])
        x1, y1 = layout.cell_center(node[0], node[1])
        draw.line([(x0, y0), (x1, y1)], fill=color + (200,), width=line_width)

    for node in step.forward_frontier:
        dist = step.forward_dist.get(node, 0)
        color = _distance_color(dist, max_dist)
        x, y = layout.cell_center(node[0], node[1])
        draw.ellipse(
            [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
            fill=color + (255,),
        )

    return img


def render_distance_bfs_frames(
    grid: np.ndarray,
    steps: list[BfsStep],
    start: tuple[int, int],
    end: tuple[int, int],
    resolution: int,
    fps: int,
    duration: float,
) -> list[Image.Image]:
    """Render BFS exploration with distance-based coloring."""
    total_frames = int(fps * duration)
    if total_frames < 2:
        total_frames = 2

    base, layout = render_maze_base(grid, resolution, start, end)
    num_steps = len(steps)
    if num_steps == 0:
        return [base.copy()]

    # Compute max distance from the final step
    final_dist = steps[-1].forward_dist
    max_dist = max(final_dist.values()) if final_dist else 1

    # 80% exploration, 20% hold final state
    explore_frames = int(total_frames * 0.8)
    hold_frames = total_frames - explore_frames

    frames: list[Image.Image] = []
    for f in range(explore_frames):
        t = f / max(1, explore_frames - 1)
        step_idx = min(int(t * (num_steps - 1)), num_steps - 1)
        frame = _render_distance_bfs_overlay(base, layout, steps[step_idx], max_dist)
        frames.append(frame)

    # Hold final state
    final_frame = _render_distance_bfs_overlay(base, layout, steps[-1], max_dist)
    for _ in range(hold_frames):
        frames.append(final_frame)

    return frames


def render_pruning_frames(
    grid: np.ndarray,
    full_parent: dict[tuple[int, int], tuple[int, int] | None],
    full_dist: dict[tuple[int, int], int],
    pruning_layers: list[list[tuple[int, int]]],
    path: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    resolution: int,
    fps: int,
    duration: float,
) -> list[Image.Image]:
    """Render pruning animation: gradually remove non-path branches."""
    total_frames = int(fps * duration)
    if total_frames < 2:
        total_frames = 2

    base, layout = render_maze_base(grid, resolution, start, end)
    max_dist = max(full_dist.values()) if full_dist else 1

    line_width = max(2, int(layout.path_size * 0.4))
    path_width = max(3, int(layout.path_size * 0.45))

    # Build cumulative removal sets
    num_layers = len(pruning_layers)
    removed_cumulative: list[set[tuple[int, int]]] = [set()]
    removed = set()
    for layer in pruning_layers:
        removed = removed | set(layer)
        removed_cumulative.append(set(removed))

    # Frame allocation: 10% hold full, 80% prune, 10% hold result
    hold_start = max(1, int(total_frames * 0.1))
    hold_end = max(1, int(total_frames * 0.1))
    prune_frames = total_frames - hold_start - hold_end

    path_set = set(path)

    def _draw_state(removed_now: set[tuple[int, int]]) -> Image.Image:
        img = base.copy()
        draw = ImageDraw.Draw(img)

        # Draw remaining non-path edges
        for node, par in full_parent.items():
            if par is None or node in removed_now or node in path_set:
                continue
            dist = full_dist.get(node, 0)
            color = _distance_color(dist, max_dist)
            x0, y0 = layout.cell_center(par[0], par[1])
            x1, y1 = layout.cell_center(node[0], node[1])
            draw.line([(x0, y0), (x1, y1)], fill=color + (160,), width=line_width)

        # Draw solution path on top (thicker, brighter)
        for i in range(len(path) - 1):
            dist = full_dist.get(path[i + 1], 0)
            color = _distance_color(dist, max_dist)
            x0, y0 = layout.cell_center(path[i][0], path[i][1])
            x1, y1 = layout.cell_center(path[i + 1][0], path[i + 1][1])
            draw.line([(x0, y0), (x1, y1)], fill=color + (240,), width=path_width)

        return img

    frames: list[Image.Image] = []

    # Hold full tree
    full_frame = _draw_state(set())
    for _ in range(hold_start):
        frames.append(full_frame)

    # Pruning animation
    for f in range(prune_frames):
        t = f / max(1, prune_frames - 1)
        stage_idx = min(int(t * num_layers), num_layers)
        frame = _draw_state(removed_cumulative[stage_idx])
        frames.append(frame)

    # Hold final (path only)
    final_frame = _draw_state(removed_cumulative[-1])
    for _ in range(hold_end):
        frames.append(final_frame)

    return frames
