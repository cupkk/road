"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""
import math
import numpy as np
import matplotlib.pyplot as plt


def _remove_duplicate_points(path, min_dist=1e-6):
    cleaned = [path[0]]
    for point in path[1:]:
        if np.linalg.norm(point - cleaned[-1]) > min_dist:
            cleaned.append(point)
    return np.asarray(cleaned, dtype=float)


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


def _densify_path(path, max_segment):
    path = np.asarray(path, dtype=float)
    if len(path) <= 1:
        return path.copy()
    densified = [path[0]]
    for i in range(1, len(path)):
        p0 = path[i - 1]
        p1 = path[i]
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len <= max_segment:
            densified.append(p1)
            continue
        steps = int(math.ceil(seg_len / max_segment))
        for step in range(1, steps + 1):
            t = step / steps
            densified.append(p0 + t * (p1 - p0))
    return np.asarray(densified, dtype=float)


def _segment_is_free(env, p0, p1, sample_step):
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


def _trajectory_is_safe(env, traj, sample_step):
    for i in range(1, len(traj)):
        if not _segment_is_free(env, traj[i - 1], traj[i], sample_step):
            return False
    return True


def _cubic_spline_coeffs(t, y):
    """
    Natural cubic spline coefficients for 1D data.
    Returns arrays a, b, c, d for each interval.
    """
    n = len(t)
    a = y.astype(float).copy()
    h = np.diff(t)

    if np.any(h <= 0.0):
        raise ValueError("Time values must be strictly increasing.")

    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2.0 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    return a, b, c, d


def _evaluate_spline(t_query, t, a, b, c, d):
    idx = np.searchsorted(t, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 2)
    dt = t_query - t[idx]
    return a[idx] + b[idx] * dt + c[idx] * dt ** 2 + d[idx] * dt ** 3


def _time_from_distance(s, v_max, a_max):
    total = float(s[-1])
    if total <= 0.0:
        return np.zeros_like(s)

    t_acc = v_max / a_max
    s_acc = 0.5 * a_max * t_acc * t_acc

    if 2.0 * s_acc >= total:
        t_acc = math.sqrt(total / a_max)
        t_total = 2.0 * t_acc
        times = np.zeros_like(s)
        half = total / 2.0
        for i, si in enumerate(s):
            if si <= half:
                times[i] = math.sqrt(2.0 * si / a_max)
            else:
                times[i] = t_total - math.sqrt(2.0 * (total - si) / a_max)
        return times

    s_const = total - 2.0 * s_acc
    t_total = 2.0 * t_acc + s_const / v_max
    times = np.zeros_like(s)
    for i, si in enumerate(s):
        if si <= s_acc:
            times[i] = math.sqrt(2.0 * si / a_max)
        elif si <= s_acc + s_const:
            times[i] = t_acc + (si - s_acc) / v_max
        else:
            times[i] = t_total - math.sqrt(2.0 * (total - si) / a_max)
    return times


def generate_trajectory(
    path,
    v_max=1.0,
    a_max=2.0,
    dt=0.1,
    max_segment=0.5,
    simplify_angle_deg=5.0,
    simplify_min_dist=0.05,
    env=None,
    max_refine=2,
    collision_sample=0.2,
):
    """
    Generate a smooth trajectory using a natural cubic spline with time scaling.

    Args:
        path: N x 3 numpy array of discrete points.
        v_max: Maximum velocity (m/s).
        a_max: Maximum acceleration (m/s^2).
        dt: Time step for sampling the trajectory.
        max_segment: Max segment length for spline knot densification.
        simplify_angle_deg: Angle threshold (deg) for optional path pruning.
        simplify_min_dist: Min distance to keep points during pruning.
        env: FlightEnvironment instance for collision checking (optional).
        max_refine: Maximum refinement attempts on collision.
        collision_sample: Sample step for collision checking.

    Returns:
        t_samples: 1D array of time stamps.
        traj: M x 3 array of trajectory points.
        path_times: 1D array of time stamps for the (cleaned) path points.
        plot_path: Path points used for plotting.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("Path must be an N x 3 array.")
    if len(path) < 2:
        raise ValueError("Path must contain at least two points.")

    if v_max <= 0.0:
        raise ValueError("v_max must be positive.")
    if a_max <= 0.0:
        raise ValueError("a_max must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if max_segment <= 0.0:
        raise ValueError("max_segment must be positive.")

    path_clean = _remove_duplicate_points(path)
    if len(path_clean) < 2:
        raise ValueError("Path must contain at least two distinct points.")

    path_plot = path_clean.copy()
    path_fit_base = _prune_by_angle(path_clean, simplify_angle_deg, simplify_min_dist)
    if len(path_fit_base) < 2:
        path_fit_base = path_clean.copy()

    deltas = np.diff(path_plot, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    if cumulative[-1] <= 0.0:
        raise ValueError("Path length must be positive.")

    path_times = _time_from_distance(cumulative, v_max, a_max)

    segment_limit = max_segment
    sample_dt = dt
    last_result = None

    for _ in range(max_refine + 1):
        path_fit = _densify_path(path_fit_base, segment_limit)
        fit_deltas = np.diff(path_fit, axis=0)
        fit_lengths = np.linalg.norm(fit_deltas, axis=1)
        fit_cumulative = np.concatenate(([0.0], np.cumsum(fit_lengths)))
        fit_times = _time_from_distance(fit_cumulative, v_max, a_max)
        total_time = fit_times[-1]

        t_samples = np.arange(0.0, total_time, sample_dt)
        if len(t_samples) == 0 or t_samples[-1] < total_time:
            t_samples = np.append(t_samples, total_time)

        ax, bx, cx, dx = _cubic_spline_coeffs(fit_times, path_fit[:, 0])
        ay, by, cy, dy = _cubic_spline_coeffs(fit_times, path_fit[:, 1])
        az, bz, cz, dz = _cubic_spline_coeffs(fit_times, path_fit[:, 2])

        traj_x = _evaluate_spline(t_samples, fit_times, ax, bx, cx, dx)
        traj_y = _evaluate_spline(t_samples, fit_times, ay, by, cy, dy)
        traj_z = _evaluate_spline(t_samples, fit_times, az, bz, cz, dz)

        traj = np.column_stack((traj_x, traj_y, traj_z))
        last_result = (t_samples, traj, path_times, path_plot)

        if env is None:
            return last_result
        if _trajectory_is_safe(env, traj, collision_sample):
            return last_result

        segment_limit = max(0.1, segment_limit * 0.5)
        sample_dt = max(0.02, sample_dt * 0.5)

    print("Warning: trajectory still collides after refinements.")
    return last_result


def plot_trajectory(t_samples, traj, path, path_times):
    """
    Plot x, y, z trajectories and overlay discrete path points.
    """
    path = np.asarray(path, dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    labels = ["x [m]", "y [m]", "z [m]"]
    for idx, ax in enumerate(axes):
        ax.plot(t_samples, traj[:, idx], linewidth=2)
        ax.scatter(path_times, path[:, idx], s=30, marker="x")
        ax.set_ylabel(labels[idx])
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    plt.show()
