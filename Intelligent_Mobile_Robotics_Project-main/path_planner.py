"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
import heapq
import math
import numpy as np


def _segment_is_free(env, p0, p1, sample_step):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if env.is_outside(p0) or env.is_outside(p1):
        return False
    if env.is_collide(p0) or env.is_collide(p1):
        return False
    dist = float(np.linalg.norm(p1 - p0))
    if dist == 0.0:
        return True
    steps = int(math.ceil(dist / sample_step))
    for i in range(1, steps):
        t = i / steps
        p = p0 + t * (p1 - p0)
        if env.is_outside(p) or env.is_collide(p):
            return False
    return True


def _shortcut_path(env, path, sample_step):
    path = np.asarray(path, dtype=float)
    if len(path) <= 2:
        return path.copy()
    shortened = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _segment_is_free(env, path[i], path[j], sample_step):
                break
            j -= 1
        shortened.append(path[j])
        i = j
    return np.asarray(shortened, dtype=float)


def _prune_by_angle(path, angle_deg, min_dist):
    path = np.asarray(path, dtype=float)
    if len(path) <= 2:
        return path.copy()
    pruned = [path[0]]
    angle_threshold = math.radians(angle_deg)
    for i in range(1, len(path) - 1):
        prev_point = pruned[-1]
        current = path[i]
        next_point = path[i + 1]
        if np.linalg.norm(current - prev_point) < min_dist:
            continue
        v1 = current - prev_point
        v2 = next_point - current
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = math.acos(cosang)
        if angle >= angle_threshold:
            pruned.append(current)
    pruned.append(path[-1])
    return np.asarray(pruned, dtype=float)


def _compute_bounds(path, start, goal, margin, env, resolution):
    points = np.vstack((path, np.asarray(start, dtype=float), np.asarray(goal, dtype=float)))
    min_xyz = np.min(points, axis=0) - margin
    max_xyz = np.max(points, axis=0) + margin
    min_xyz = np.maximum(min_xyz, [0.0, 0.0, 0.0])
    max_xyz = np.minimum(max_xyz, [env.env_width, env.env_length, env.env_height])

    min_idx = (
        int(math.floor(min_xyz[0] / resolution)),
        int(math.floor(min_xyz[1] / resolution)),
        int(math.floor(min_xyz[2] / resolution)),
    )
    max_idx = (
        int(math.ceil(max_xyz[0] / resolution)),
        int(math.ceil(max_xyz[1] / resolution)),
        int(math.ceil(max_xyz[2] / resolution)),
    )
    return min_idx, max_idx


def _path_length(path):
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _plan_astar(env, start, goal, resolution, weight=1.0, bounds=None, upper_bound=None):
    start = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)

    if env.is_outside(start) or env.is_outside(goal):
        raise ValueError("Start or goal is outside the environment.")

    x_count = int(round(env.env_width / resolution)) + 1
    y_count = int(round(env.env_length / resolution)) + 1
    z_count = int(round(env.env_height / resolution)) + 1

    grid_min = (0, 0, 0)
    grid_max = (x_count - 1, y_count - 1, z_count - 1)

    if bounds is None:
        min_idx, max_idx = grid_min, grid_max
    else:
        min_idx = tuple(max(bounds[0][i], grid_min[i]) for i in range(3))
        max_idx = tuple(min(bounds[1][i], grid_max[i]) for i in range(3))

    def idx_in_bounds(idx):
        return all(min_idx[i] <= idx[i] <= max_idx[i] for i in range(3))

    def point_to_idx(point):
        return (
            int(round(point[0] / resolution)),
            int(round(point[1] / resolution)),
            int(round(point[2] / resolution)),
        )

    def idx_to_point(idx):
        return np.array([idx[0] * resolution, idx[1] * resolution, idx[2] * resolution], dtype=float)

    free_cache = {}

    def is_free_idx(idx):
        if idx in free_cache:
            return free_cache[idx]
        point = idx_to_point(idx)
        free = (not env.is_outside(point)) and (not env.is_collide(point))
        free_cache[idx] = free
        return free

    sample_step = min(0.2, resolution / 2.0)

    def nearest_free_idx(target_idx, anchor_point):
        if idx_in_bounds(target_idx) and is_free_idx(target_idx):
            if _segment_is_free(env, anchor_point, idx_to_point(target_idx), sample_step):
                return target_idx
        max_radius = max(x_count, y_count, z_count)
        for radius in range(1, max_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != radius:
                            continue
                        candidate = (target_idx[0] + dx, target_idx[1] + dy, target_idx[2] + dz)
                        if not idx_in_bounds(candidate):
                            continue
                        if not is_free_idx(candidate):
                            continue
                        candidate_point = idx_to_point(candidate)
                        if _segment_is_free(env, anchor_point, candidate_point, sample_step):
                            return candidate
        raise RuntimeError("Unable to find a collision-free grid point near the anchor.")

    start_idx = nearest_free_idx(point_to_idx(start), start)
    goal_idx = nearest_free_idx(point_to_idx(goal), goal)
    goal_point = idx_to_point(goal_idx)

    neighbor_offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor_offsets.append((dx, dy, dz))

    def heuristic(point):
        return float(np.linalg.norm(point - goal_point))

    open_heap = []
    g_cost = {start_idx: 0.0}
    came_from = {}
    start_point = idx_to_point(start_idx)
    heapq.heappush(open_heap, (heuristic(start_point) * weight, start_idx))
    closed = set()

    while open_heap:
        _, current_idx = heapq.heappop(open_heap)
        if current_idx in closed:
            continue
        if current_idx == goal_idx:
            break
        closed.add(current_idx)

        current_point = idx_to_point(current_idx)
        current_g = g_cost[current_idx]
        for dx, dy, dz in neighbor_offsets:
            neighbor_idx = (current_idx[0] + dx, current_idx[1] + dy, current_idx[2] + dz)
            if not idx_in_bounds(neighbor_idx):
                continue
            if not is_free_idx(neighbor_idx):
                continue
            neighbor_point = idx_to_point(neighbor_idx)
            if not _segment_is_free(env, current_point, neighbor_point, sample_step):
                continue
            step_cost = float(np.linalg.norm(neighbor_point - current_point))
            tentative_g = current_g + step_cost
            if upper_bound is not None:
                lower_bound = tentative_g + heuristic(neighbor_point)
                if lower_bound > upper_bound:
                    continue
            if neighbor_idx not in g_cost or tentative_g < g_cost[neighbor_idx]:
                g_cost[neighbor_idx] = tentative_g
                came_from[neighbor_idx] = current_idx
                f_cost = tentative_g + weight * heuristic(neighbor_point)
                heapq.heappush(open_heap, (f_cost, neighbor_idx))

    if goal_idx not in g_cost:
        raise RuntimeError("Path not found. Try a smaller resolution.")

    path_indices = [goal_idx]
    current = goal_idx
    while current != start_idx:
        current = came_from[current]
        path_indices.append(current)
    path_indices.reverse()

    path_points = [idx_to_point(idx) for idx in path_indices]
    if not np.allclose(path_points[0], start):
        if _segment_is_free(env, start, path_points[0], sample_step):
            path_points.insert(0, start)
    if not np.allclose(path_points[-1], goal):
        if _segment_is_free(env, path_points[-1], goal, sample_step):
            path_points.append(goal)

    return np.asarray(path_points, dtype=float)


def plan_path(
    env,
    start,
    goal,
    resolution=0.5,
    weight=1.0,
    refine=True,
    coarse_resolution=None,
    refine_margin=1.0,
    use_bounds=False,
    smooth=True,
    angle_deg=10.0,
):
    """
    Plan a collision-free path with optional coarse-to-fine planning and smoothing.
    """
    if coarse_resolution is None:
        coarse_resolution = resolution * 2.0

    upper_bound = None
    bounds = None
    if refine and coarse_resolution > resolution:
        try:
            coarse_path = _plan_astar(env, start, goal, coarse_resolution, weight=weight)
            upper_bound = _path_length(coarse_path)
            if use_bounds:
                bounds = _compute_bounds(coarse_path, start, goal, refine_margin, env, resolution)
        except RuntimeError:
            upper_bound = None
            bounds = None

    if refine:
        path = _plan_astar(env, start, goal, resolution, weight=1.0, bounds=bounds, upper_bound=upper_bound)
    else:
        path = _plan_astar(env, start, goal, resolution, weight=weight, bounds=bounds, upper_bound=upper_bound)

    if smooth:
        sample_step = min(0.2, resolution / 2.0)
        path = _shortcut_path(env, path, sample_step)
        path = _prune_by_angle(path, angle_deg=angle_deg, min_dist=resolution * 0.5)

    return path
