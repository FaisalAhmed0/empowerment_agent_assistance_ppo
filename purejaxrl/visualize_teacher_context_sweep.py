"""Visualize teacher goal distributions across progressive competence contexts.

Loads a teacher checkpoint from an experiment directory and renders subplot
grids of goal-distribution heatmaps (softmax probs and empirical sample
frequencies). By default, observation and student-action slots match training
episode reset: env.reset obs (normalized if applicable) plus a student-policy
bootstrap action with a zero goal. Only the competence context varies across
panels.

When the experiment was trained with ``TEACHER_CONDITION_ONLY_ON_COMPETENCE``,
inputs are competence-only (no student bootstrap). Use ``--use_teacher_ema`` to
load ``checkpoints/teacher_ema`` instead of ``checkpoints/teacher``.

Example:
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id>
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> --use_teacher_ema
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> --num_samples 1000
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tyro

try:
    from envs.ant_maze import U_MAZE_PATH_CELLS, all_possible_goals
except ImportError:
    from purejaxrl.envs.ant_maze import U_MAZE_PATH_CELLS, all_possible_goals

try:
    from purejaxrl.load_teacher_model import (
        _make_env,
        load_experiment_config,
        load_obs_norm_stats,
        load_student_model,
        load_teacher_model,
    )
except ImportError:
    from load_teacher_model import (
        _make_env,
        load_experiment_config,
        load_obs_norm_stats,
        load_student_model,
        load_teacher_model,
    )


@dataclass
class ContextSweepConfig:
    exp_dir: str
    checkpoint_dir: str = "checkpoints"
    num_contexts: int = 8
    seed: int = 30
    output_dir: Optional[str] = None
    size_scaling: float = 4.0
    use_teacher_ema: bool = False
    num_samples: int = 1500


def _build_goal_grid(config: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Rebuild the teacher discrete goal grid used during training."""
    num_points = int(config["TEACHER_NUM_GOAL_POINTS"])
    xs = np.linspace(
        float(config["TEACHER_GOAL_X_MIN"]),
        float(config["TEACHER_GOAL_X_MAX"]),
        num_points,
    )
    ys = np.linspace(
        float(config["TEACHER_GOAL_Y_MIN"]),
        float(config["TEACHER_GOAL_Y_MAX"]),
        num_points,
    )
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    goal_grid = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
    custom_goal = np.array([12.0, 8.0], dtype=np.float32)
    replace_idx = int(np.argmin(np.sum((goal_grid - custom_goal) ** 2, axis=-1)))
    goal_grid[replace_idx] = custom_goal
    return goal_grid, num_points


def _path_ordered_competence_ranks(
    competence_goals: np.ndarray,
    size_scaling: float = 4.0,
) -> np.ndarray:
    """Map each competence goal to its rank along the U-corridor path."""
    path_xy = np.asarray(U_MAZE_PATH_CELLS, dtype=np.float32) * float(size_scaling)
    ranks = np.full(competence_goals.shape[0], -1, dtype=np.int32)
    used = set()
    for rank, xy in enumerate(path_xy):
        dists = np.sum((competence_goals - xy) ** 2, axis=-1)
        order = np.argsort(dists)
        for idx in order:
            idx = int(idx)
            if idx not in used:
                ranks[idx] = rank
                used.add(idx)
                break
    if np.any(ranks < 0):
        raise ValueError("Failed to assign path ranks to all competence goals")
    return ranks


def build_progressive_contexts(
    num_competence: int,
    ranks: np.ndarray,
    num_contexts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Progressive mastery: near goals saturate first, far goals last.

    For frontier ``t`` in ``linspace(0, 1, num_contexts)``::
        competence[i] = clip(t * N - rank[i], 0, 1)
    """
    n = float(num_competence)
    ts = np.linspace(0.0, 1.0, num_contexts, dtype=np.float32)
    contexts = np.zeros((num_contexts, num_competence), dtype=np.float32)
    for i, t in enumerate(ts):
        contexts[i] = np.clip(t * n - ranks.astype(np.float32), 0.0, 1.0)
    return contexts, ts


def _build_teacher_input(
    obs: jnp.ndarray,
    competence: jnp.ndarray,
    action: jnp.ndarray,
    *,
    condition_on_competence: bool,
    only_on_competence: bool = False,
) -> jnp.ndarray:
    if only_on_competence:
        return competence
    parts = [obs]
    if condition_on_competence:
        parts.append(competence)
    parts.append(action)
    return jnp.concatenate(parts, axis=-1)


def _normalize_obs(
    obs: jnp.ndarray,
    obs_norm: tuple[np.ndarray, np.ndarray] | None,
) -> jnp.ndarray:
    """Apply saved running-mean/var normalization, matching training."""
    if obs_norm is None:
        return obs
    mean, var = obs_norm
    return (obs - jnp.asarray(mean, dtype=obs.dtype)) / jnp.sqrt(
        jnp.asarray(var, dtype=obs.dtype) + 1e-8
    )


def sample_reset_teacher_inputs(
    env,
    env_params,
    student_train_state,
    config: dict[str, Any],
    *,
    obs_norm: tuple[np.ndarray, np.ndarray] | None,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
    """Match training episode-start teacher inputs.

    Mirrors ``make_train`` reset bootstrap:
      1. ``env.reset``
      2. normalize obs if ``NORMALIZE_ENV``
      3. student samples an action with a zero goal appended (if conditioned)
      4. teacher sees ``(base_obs, competence, bootstrap_action)``

    Returns ``(norm_obs, bootstrap_action, start_xy)`` where ``start_xy`` is the
    denormalized/raw agent xy for plotting.
    """
    condition_on_goal = bool(config.get("CONDITION_ON_GOAL", False))
    normalize_env = bool(config.get("NORMALIZE_ENV", False))
    goal_dim = 2

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng, action_rng = jax.random.split(rng, 3)
    raw_obs, _env_state = env.reset(reset_rng, env_params)
    raw_obs = jnp.asarray(raw_obs, dtype=jnp.float32).reshape(-1)
    start_xy = np.asarray(jax.device_get(raw_obs[:2]), dtype=np.float32)

    if normalize_env:
        if obs_norm is None:
            raise FileNotFoundError(
                "NORMALIZE_ENV is True but obs_norm_stats.npz was not found; "
                "needed to match training reset observations."
            )
        norm_obs = _normalize_obs(raw_obs, obs_norm)
    else:
        norm_obs = raw_obs

    policy_obs = norm_obs
    if condition_on_goal:
        policy_obs = jnp.concatenate(
            [policy_obs, jnp.zeros((goal_dim,), dtype=policy_obs.dtype)],
            axis=-1,
        )
    pi, _ = student_train_state.apply_fn(student_train_state.params, policy_obs)
    bootstrap_action = pi.sample(seed=action_rng)
    bootstrap_action = jnp.asarray(bootstrap_action, dtype=jnp.float32).reshape(-1)
    return norm_obs, bootstrap_action, start_xy


def compute_teacher_probs(
    teacher_train_state,
    config: dict[str, Any],
    contexts: np.ndarray,
    *,
    obs: jnp.ndarray | None = None,
    action: jnp.ndarray | None = None,
) -> np.ndarray:
    """Return teacher P(goal) for each context; shape (num_contexts, num_goals)."""
    only_on_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    condition_on_competence = bool(
        config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
    ) or only_on_competence
    params = teacher_train_state.params
    apply_fn = teacher_train_state.apply_fn

    probs_list = []
    for competence in contexts:
        teacher_obs = _build_teacher_input(
            obs if obs is not None else jnp.zeros((0,), dtype=jnp.float32),
            jnp.asarray(competence, dtype=jnp.float32),
            action if action is not None else jnp.zeros((0,), dtype=jnp.float32),
            condition_on_competence=condition_on_competence,
            only_on_competence=only_on_competence,
        )
        pi, _ = apply_fn(params, teacher_obs)
        probs_list.append(np.asarray(jax.device_get(pi.probs)).reshape(-1))
    return np.stack(probs_list, axis=0)


def sample_teacher_goal_counts(
    teacher_train_state,
    config: dict[str, Any],
    contexts: np.ndarray,
    *,
    num_samples: int,
    seed: int,
    obs: jnp.ndarray | None = None,
    action: jnp.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample goals from the teacher and bin counts per context.

    Returns ``(counts, freqs)`` each of shape ``(num_contexts, num_goals)``,
    where ``freqs = counts / num_samples``.
    """
    only_on_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    condition_on_competence = bool(
        config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
    ) or only_on_competence
    params = teacher_train_state.params
    apply_fn = teacher_train_state.apply_fn
    rng = jax.random.PRNGKey(seed)

    counts_list = []
    for competence in contexts:
        teacher_obs = _build_teacher_input(
            obs if obs is not None else jnp.zeros((0,), dtype=jnp.float32),
            jnp.asarray(competence, dtype=jnp.float32),
            action if action is not None else jnp.zeros((0,), dtype=jnp.float32),
            condition_on_competence=condition_on_competence,
            only_on_competence=only_on_competence,
        )
        pi, _ = apply_fn(params, teacher_obs)
        rng, sample_rng = jax.random.split(rng)
        goal_idxs = pi.sample(seed=sample_rng, sample_shape=(num_samples,))
        goal_idxs_np = np.asarray(jax.device_get(goal_idxs)).reshape(-1).astype(np.int32)
        num_goals = int(np.asarray(jax.device_get(pi.probs)).reshape(-1).shape[0])
        counts = np.bincount(goal_idxs_np, minlength=num_goals).astype(np.float32)
        counts_list.append(counts[:num_goals])
    counts_arr = np.stack(counts_list, axis=0)
    freqs = counts_arr / float(max(num_samples, 1))
    return counts_arr, freqs


def _format_competence_title(competence: np.ndarray) -> str:
    """Format competence as a compact row-vector title string."""
    vals = np.asarray(competence).reshape(-1)
    body = ", ".join(f"{v:.2f}" for v in vals)
    return f"c = [{body}]"


def _plot_probs_on_ax(
    ax,
    goal_grid_xy: np.ndarray,
    probs: np.ndarray,
    num_points: int,
    *,
    cmap,
    vmin: float,
    vmax: float,
    start_xy=None,
    competence_goals=None,
    competence=None,
):
    """Draw a single teacher goal-distribution heatmap on ``ax``."""
    goal_grid_xy = np.asarray(goal_grid_xy)
    probs = np.asarray(probs).reshape(-1)
    gx = goal_grid_xy[:, 0].reshape(num_points, num_points)
    gy = goal_grid_xy[:, 1].reshape(num_points, num_points)
    vgrid = probs.reshape(num_points, num_points)
    mesh = ax.pcolormesh(
        gx, gy, vgrid, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax
    )
    if start_xy is not None:
        start_xy = np.asarray(start_xy).reshape(-1)
        ax.scatter(
            start_xy[0],
            start_xy[1],
            s=160,
            marker="*",
            c="black",
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
            label="start",
        )
    if competence_goals is not None:
        competence_goals = np.asarray(competence_goals)
        # Inactive competence goals: thin open circles.
        ax.scatter(
            competence_goals[:, 0],
            competence_goals[:, 1],
            s=36,
            marker="o",
            facecolors="none",
            edgecolors="0.35",
            linewidths=0.9,
            zorder=3,
            label="competence goals",
        )
        if competence is not None:
            competence = np.asarray(competence).reshape(-1)
            active = competence > 1e-6
            if np.any(active):
                # Highlight goals with non-zero competence; marker size and
                # fill alpha scale with the competence entry.
                active_xy = competence_goals[active]
                active_c = competence[active]
                sizes = 80.0 + 140.0 * active_c
                ax.scatter(
                    active_xy[:, 0],
                    active_xy[:, 1],
                    s=sizes,
                    marker="o",
                    c=active_c,
                    cmap="YlOrRd",
                    vmin=0.0,
                    vmax=1.0,
                    edgecolors="black",
                    linewidths=1.4,
                    zorder=5,
                    label="active competence",
                )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return mesh


def render_context_sweep_figure(
    goal_grid_xy: np.ndarray,
    values: np.ndarray,
    contexts: np.ndarray,
    ts: np.ndarray,
    num_points: int,
    *,
    competence_goals: np.ndarray,
    start_xy=None,
    save_path: Optional[str] = None,
    colorbar_label: str = "P(goal)",
    suptitle: str = "Teacher goal distribution vs progressive competence",
):
    """Subplot grid of per-goal values, blue (low) to red (high) competence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    num_contexts = int(values.shape[0])
    ncols = int(math.ceil(math.sqrt(num_contexts)))
    nrows = int(math.ceil(num_contexts / ncols))
    vmax = float(np.max(values)) if values.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 6.0 * nrows),
        squeeze=False,
    )
    for i in range(num_contexts):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        t = float(ts[i])
        base_color = plt.cm.coolwarm(t)
        cmap = LinearSegmentedColormap.from_list(
            f"ctx_{i}",
            ["white", base_color],
        )
        mesh = _plot_probs_on_ax(
            ax,
            goal_grid_xy,
            values[i],
            num_points,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            start_xy=start_xy,
            competence_goals=competence_goals,
            competence=contexts[i],
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
        ax.set_title(_format_competence_title(contexts[i]), fontsize=10)

    for j in range(num_contexts, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].axis("off")

    fig.suptitle(
        suptitle,
        fontsize=16,
        y=1.01,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def run_context_sweep(args: ContextSweepConfig) -> dict[str, Any]:
    config = load_experiment_config(args.exp_dir)
    teacher_train_state = load_teacher_model(
        exp_dir=args.exp_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        use_ema=args.use_teacher_ema,
    )
    only_on_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )

    norm_obs = None
    bootstrap_action = None
    start_xy = None
    if not only_on_competence:
        student_train_state = load_student_model(
            exp_dir=args.exp_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
        )
        obs_norm = load_obs_norm_stats(args.exp_dir, args.checkpoint_dir)
        env, env_params = _make_env(config)
        norm_obs, bootstrap_action, start_xy = sample_reset_teacher_inputs(
            env,
            env_params,
            student_train_state,
            config,
            obs_norm=obs_norm,
            seed=args.seed,
        )

    competence_goals = np.asarray(all_possible_goals(), dtype=np.float32)
    num_competence = int(competence_goals.shape[0])
    ranks = _path_ordered_competence_ranks(
        competence_goals, size_scaling=args.size_scaling
    )
    contexts, ts = build_progressive_contexts(
        num_competence, ranks, args.num_contexts
    )

    goal_grid, num_points = _build_goal_grid(config)
    probs = compute_teacher_probs(
        teacher_train_state,
        config,
        contexts,
        obs=norm_obs,
        action=bootstrap_action,
    )
    sample_counts, sample_freqs = sample_teacher_goal_counts(
        teacher_train_state,
        config,
        contexts,
        num_samples=args.num_samples,
        seed=args.seed + 1,
        obs=norm_obs,
        action=bootstrap_action,
    )

    output_dir = args.output_dir or os.path.join(
        args.exp_dir, "teacher_context_sweep"
    )
    os.makedirs(output_dir, exist_ok=True)
    ckpt_tag = "ema" if args.use_teacher_ema else "final"
    png_path = os.path.join(
        output_dir, f"teacher_goal_distribution_contexts_{ckpt_tag}.png"
    )
    samples_png_path = os.path.join(
        output_dir, f"teacher_goal_samples_contexts_{ckpt_tag}.png"
    )
    npz_path = os.path.join(
        output_dir, f"teacher_goal_distribution_contexts_{ckpt_tag}.npz"
    )

    fig = render_context_sweep_figure(
        goal_grid,
        probs,
        contexts,
        ts,
        num_points,
        competence_goals=competence_goals,
        start_xy=start_xy,
        save_path=png_path,
        colorbar_label="P(goal)",
        suptitle="Teacher goal distribution vs progressive competence",
    )
    import matplotlib.pyplot as plt

    plt.close(fig)

    samples_fig = render_context_sweep_figure(
        goal_grid,
        sample_freqs,
        contexts,
        ts,
        num_points,
        competence_goals=competence_goals,
        start_xy=start_xy,
        save_path=samples_png_path,
        colorbar_label="sample freq",
        suptitle=(
            f"Teacher goal samples vs progressive competence "
            f"(N={args.num_samples})"
        ),
    )
    plt.close(samples_fig)

    np.savez(
        npz_path,
        probs=probs,
        sample_counts=sample_counts,
        sample_freqs=sample_freqs,
        num_samples=np.asarray(args.num_samples),
        contexts=contexts,
        ts=ts,
        goal_grid=goal_grid,
        competence_goals=competence_goals,
        ranks=ranks,
        reset_obs=(
            np.asarray(jax.device_get(norm_obs)) if norm_obs is not None else np.array([])
        ),
        bootstrap_action=(
            np.asarray(jax.device_get(bootstrap_action))
            if bootstrap_action is not None
            else np.array([])
        ),
        start_xy=start_xy if start_xy is not None else np.array([]),
        use_teacher_ema=np.asarray(args.use_teacher_ema),
        only_on_competence=np.asarray(only_on_competence),
    )
    print(f"[context_sweep] saved figure to {png_path}")
    print(f"[context_sweep] saved samples figure to {samples_png_path}")
    print(f"[context_sweep] saved arrays to {npz_path}")
    if only_on_competence:
        print("[context_sweep] teacher conditioned only on competence vector")
    else:
        print(
            f"[context_sweep] reset start_xy={start_xy.tolist()} "
            f"action_norm={float(np.linalg.norm(jax.device_get(bootstrap_action))):.4f}"
        )
    print(
        f"[context_sweep] using {'EMA' if args.use_teacher_ema else 'final'} teacher checkpoint"
    )
    return {
        "png_path": png_path,
        "samples_png_path": samples_png_path,
        "npz_path": npz_path,
        "probs": probs,
        "sample_counts": sample_counts,
        "sample_freqs": sample_freqs,
        "contexts": contexts,
    }


def main():
    args = tyro.cli(ContextSweepConfig)
    run_context_sweep(args)


if __name__ == "__main__":
    main()
