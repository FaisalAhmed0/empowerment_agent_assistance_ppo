import json
import math
import os
import traceback
import time
os.environ["MUJOCO_GL"] = "osmesa"
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import wandb
from brax import envs as brax_envs
from brax.io import html
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from dataclasses import dataclass, asdict, field
from flax.training.train_state import TrainState
from copy import deepcopy
import distrax
from envs.ant_maze import all_possible_goals
import tyro
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
try:
    from purejaxrl.envs.factory import make_custom_env
except ImportError:
    from envs.factory import make_custom_env

import orbax.checkpoint as ocp

from wonderwords import RandomWord



@dataclass
class TrainConfig:
    LR: float = 3e-4
    NUM_ENVS: int = 256
    NUM_STEPS: int = 64
    TOTAL_TIMESTEPS: int = int(5e7)
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 8
    GAMMA: float = 0.99
    GAMMA_CL: float = 0.99
    CL_BUFFER_SIZE: int = 1000
    GAE_LAMBDA: float = 0.8
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0
    VF_COEF: float = 0.5
    HIDDEN_DIM: int = 256
    MAX_GRAD_NORM: float = 1.0
    ACTIVATION: str = "tanh"
    ENV_NAME: str = "ant_u_maze_single_goal"
    ENV_BACKEND: str | None = None
    EPISODE_LENGTH: int = 1000
    ACTION_REPEAT: int = 1
    ENV_KWARGS: dict[str, Any] = field(default_factory=dict)
    ANNEAL_LR: bool = True
    NORMALIZE_ENV: bool = True
    OBS_NORM_WARMUP_STEPS: int = 5000
    DEBUG: bool = True
    SEED: int = 30
    WANDB_MODE: str = "online"
    ENTITY: str = ""
    PROJECT: str = "purejaxrl"
    EVAL_RENDER_STEPS: int = 1000
    EVAL_RENDER_MAX_FRAMES: int = 1000
    EVAL_RENDER_HEIGHT: int = 360
    EVAL_RENDER_LOG_WANDB_HTML: bool = True
    TRAIN_RENDER_FREQ: int = 200
    EVAL_FREQ: int = 10
    EVAL_NUM_ENVS: int = 50
    COMMENT: str = ""
    ADD_GOAL_REWARD: bool = True
    CONDITION_ON_GOAL: bool = True
    GOAL_REACH_EPSILON: float = 0.5
    TEACHER_GOAL_X_MIN: float = 4.0
    TEACHER_GOAL_X_MAX: float = 12.0
    TEACHER_GOAL_Y_MIN: float = 4.0
    TEACHER_GOAL_Y_MAX: float = 12.0
    TEACHER_NUM_GOAL_POINTS: int = 30
    TEACHER_GOAL_COUNT_VIZ_LOG_WANDB: bool = True
    TEACHER_GOAL_COUNT_VIZ_FREQ: int = 100
    TEACHER_EMPOWERMENT_GRID_VIZ_LOG_WANDB: bool = True
    TEACHER_EMPOWERMENT_GRID_VIZ_FREQ: int = 100
    TEACHER_EMPOWERMENT_GRID_NUM_FUTURES: int = 64
    TEACHER_EMPOWERMENT_GRID_REF_ENV_INDEX: int = 0
    TEACHER_HEATMAP_CMAP: str = "Blues"
    TEACHER_HIDDEN_DIM: int = 256
    SAVE_MODEL: bool = False
    checkpoint_dir: str = "checkpoints"
    GOAL_REWARD_COEF: float = 1.0
    TASK_REWARD_COEF: float = 1.0
    INTERPOLATED_REWARD: bool = False
    NUM_EVAL_ENVS: int = 32
    CONDITION_TEACHER_ON_COMPETENCE: bool = True
    USE_DISTANCE_IN_COMPETENCE: bool = False
    USE_AVERAGE_COMPETENCE_REWARD: bool = False
    USE_LEARNING_PROGRESS_REWARD: bool = False
    USE_TEACHER_EMPOWERMENT_REWARD: bool = True
    # TEACHER_EMPOWERMENT_REWARD_COEF: float = 1.0
    ABSOLUTE_LEARNING_PROGRESS: bool = False
    TEACHER_EMA_COEFF: float = 0.999
    TEACHER_SOFTMAX_VIZ_NUM_SNAPSHOTS: int = 0  # in-training softmax visuals (evenly spaced); 0 disables
    TEACHER_SOFTMAX_VIZ_LOG_WANDB: bool = True
    TEACHER_SOFTMAX_VIZ_REF_ENV_INDEX: int = 0
    TEACHER_COMPETENCE_VIZ_LOG_WANDB: bool = True
    TEACHER_COMPETENCE_VIZ_FREQ: int = 100
    LOG_TEACHER_INPUT_GRADS: bool = True
    LOG_TEACHER_INPUT_GRADS_FREQ: int = 500
    SAVE_AGENT_TRAJECTORY_XY: bool = True
    AGENT_TRAJECTORY_REF_ENV_INDEX: int = 0
    TEACHER_ROLLOUT_BUFFER_SIZE: int = 4
    TEACHER_NUM_MINIBATCHES: int = 8
    TEACHER_UPDATE_EPOCHS: int = 4
    TEACHER_LR: float = 3e-4
    TEACHER_GAMMA: float = 0.99
    TEACHER_GAE_LAMBDA: float = 0.8
    TEACHER_CLIP_EPS: float = 0.2
    TEACHER_ENT_COEF: float = 0.001
    TEACHER_VF_COEF: float = 0.5
    TEACHER_MAX_GRAD_NORM: float = 1.0
    TEACHER_USE_ENCODERS: bool = True
    TEACHER_ACTIVATION: str = "tanh"
    # Empowerment model
    EMPOWERMENT_LR: float = 3e-4
    EMPOWERMENT_MAX_GRAD_NORM: float = 1.0
    EMPOWERMENT_REPR_DIM: int = 64
    EMPOWERMENT_HIDDEN_DIM: int = 256
    EMPOWERMENT_NUM_LAYERS: int = 2
    EMPOWERMENT_UPDATE_EPOCHS: int = 1
    EMPOWERMENT_NUM_MINIBATCHES: int = 128
    EMPOWERMENT_SUBSAMPLE_SIZE: int = 1024
    EMPOWERMENT_ENERGY_FN: str = "l2"
    EMPOWERMENT_CONTRASTIVE_LOSS: str = "fwd_infonce"
    USE_SEPARATE_FUTURE_STATE_ENCODERS: bool = False
    EMP_REWARD_COEF: float = 1.0
    # Agent XY logging
    AGENT_POSITIONS_LOG_FREQ: int = 100
    AGENT_POSITIONS_REF_ENV_INDEX: int = 0
    AGENT_POSITIONS_INJIT_SUBSAMPLE_EVERY: int = 1
    AGENT_POSITIONS_SAVE_DIR: str | None = None
    AGENT_POSITIONS_MAX_POINTS: int = 10000
    AGENT_POSITIONS_ONLY_REF_ENV: bool = False



def _inner_brax_state(state):
    """Walk nested wrappers until the inner Brax ``State`` is reached."""
    current = state
    while hasattr(current, "env_state") and not hasattr(current, "pipeline_state"):
        current = current.env_state
    return current


def _replace_inner_brax_state(wrapped_state, brax_state):
    """Replace the inner Brax state, updating ``org_obs`` when present."""
    if hasattr(wrapped_state, "pipeline_state"):
        return brax_state
    new_inner = _replace_inner_brax_state(wrapped_state.env_state, brax_state)
    updates = {"env_state": new_inner}
    if hasattr(wrapped_state, "org_obs"):
        updates["org_obs"] = brax_state.obs
    return wrapped_state.replace(**updates)


def _success_metric(wrapped_state):
    return _inner_brax_state(wrapped_state).metrics["success"]


def _normalize_xy(goal, mean, var):
    mean_xy = mean[..., :2]
    var_xy = var[..., :2]
    xy = (goal[..., :2] - mean_xy) / jnp.sqrt(var_xy + 1e-8)
    if goal.shape[-1] > 2:
        # Normalize Z with the same obs channel stats as agent torso z.
        z = (goal[..., 2:3] - mean[..., 2:3]) / jnp.sqrt(var[..., 2:3] + 1e-8)
        return jnp.concatenate([xy, z], axis=-1)
    return xy


def _denorm_xy(xy, mean, var):
    mean_xy = mean[..., :2]
    var_xy = var[..., :2]
    denorm_xy = xy[..., :2] * jnp.sqrt(var_xy + 1e-8) + mean_xy
    if xy.shape[-1] > 2:
        denorm_z = xy[..., 2:3] * jnp.sqrt(var[..., 2:3] + 1e-8) + mean[..., 2:3]
        return jnp.concatenate([denorm_xy, denorm_z], axis=-1)
    return denorm_xy


def _agent_world_xy(obsv, env_state, env_index, *, normalize_env, base_obs_dim):
    if normalize_env:
        return env_state.org_obs[env_index, :2]
    return obsv[env_index, :base_obs_dim][:2]


def save_agent_trajectory_xy(agent_xy, config, output_dir, exp_name):
    """Save a single agent's training trajectory for offline visualization."""
    agent_xy = np.asarray(agent_xy)
    flat_xy = agent_xy.reshape(-1, 2)
    num_updates, num_steps = agent_xy.shape[:2]
    update_idx = np.repeat(np.arange(num_updates), num_steps)
    step_in_update = np.tile(np.arange(num_steps), num_updates)
    env_step = update_idx * num_steps + step_in_update + 1
    global_step = env_step * config["NUM_ENVS"]

    path = os.path.join(output_dir, f"{exp_name}_agent_trajectory_xy.npz")
    np.savez(
        path,
        agent_xy=flat_xy,
        update_idx=update_idx,
        step_in_update=step_in_update,
        env_step=env_step,
        global_step=global_step,
        ref_env_index=config["AGENT_TRAJECTORY_REF_ENV_INDEX"],
        num_envs=config["NUM_ENVS"],
        num_steps_per_update=config["NUM_STEPS"],
    )
    return path, flat_xy, global_step


def plot_agent_trajectory_xy(xy, steps, save_path, *, title="Agent Trajectory"):
    """Scatter plot of agent (x, y) positions colored by training progress."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = np.asarray(xy).reshape(-1, 2)
    steps = np.asarray(steps).reshape(-1)
    if len(xy) == 0:
        raise ValueError("Need at least one trajectory point to plot.")

    cmap = plt.get_cmap("Blues")
    step_min = float(steps.min())
    step_max = float(steps.max())
    if step_max > step_min:
        color_values = (steps - step_min) / (step_max - step_min)
    else:
        color_values = np.zeros_like(steps, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color_values,
        cmap=cmap,
        s=4,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("off")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("normalized training steps")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, save_path


def _collapse_obs_norm_stats(stats_state, obs_dim):
    """Reduce batched ``(NUM_ENVS, obs_dim)`` stats to global 1D for eval."""
    if stats_state is None:
        return None
    mean = stats_state.mean
    var = stats_state.var
    if mean.ndim > 1 and mean.shape[-1] == obs_dim:
        # mean = jnp.mean(mean, axis=0)
        # var = jnp.mean(var, axis=0)
        mean = mean[0]
        var = var[0]
    return stats_state.replace(mean=mean, var=var)


def evaluate_multiple_goals(
    env,
    brax_env,
    network,
    params,
    goals,
    num_envs_per_goal,
    max_steps=1000,
    warmup_env_state=None,
    normalize_obs=False,
    condition_on_goal=True,
    use_distance_in_competence=False,
    config=None,
):
    """Evaluate success rate for each goal over multiple random starts.

    Uses the full wrapped env stack (VecEnv + LogWrapper + ...). VecEnv.reset
    already vmaps over envs, so each goal is evaluated with a batched reset of
    ``num_envs_per_goal`` keys rather than vmapping over scalar keys.

    ``goals`` must be raw world coordinates. When ``normalize_obs`` is True,
    observations and conditioned goals are normalized using collapsed warmup
    stats so eval can use a different batch size than training.
    """
    env_params = None
    obs_dim = brax_env.observation_size
    eval_stats = (
        _collapse_obs_norm_stats(warmup_env_state, obs_dim)
        if normalize_obs and warmup_env_state is not None
        else warmup_env_state
    )
    norm_mean = eval_stats.mean if eval_stats is not None else None
    norm_var = eval_stats.var if eval_stats is not None else None

    def eval_one_goal(rng, specific_goal):
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs_per_goal)
        if eval_stats is not None:
            obsv, env_state = env.reset_with_stats(
                reset_rngs, eval_stats, env_params
            )
        else:
            obsv, env_state = env.reset(reset_rngs, env_params)

        brax_state = _inner_brax_state(env_state)
        env_state = _replace_inner_brax_state(env_state, brax_state)

        if normalize_obs:
            obsv = (brax_state.obs - norm_mean) / jnp.sqrt(norm_var + 1e-8)
        else:
            obsv = brax_state.obs

        if condition_on_goal:
            if normalize_obs:
                policy_goal = _normalize_xy(specific_goal, norm_mean, norm_var)
            else:
                policy_goal = specific_goal
            goal_batch = jnp.broadcast_to(
                policy_goal, (num_envs_per_goal, policy_goal.shape[-1])
            )
            raw_goal_batch = jnp.broadcast_to(
                specific_goal, (num_envs_per_goal, policy_goal.shape[-1])
            )

        def step_fn(carry, _):
            obsv, env_state, rng, ever_done = carry
            rng, step_rng, action_rng = jax.random.split(rng, 3)
            step_rngs = jax.random.split(step_rng, num_envs_per_goal)
            if condition_on_goal:
                policy_obs = jnp.concatenate([obsv, goal_batch], axis=-1)
            else:
                policy_obs = obsv
            pi, _ = network.apply(params, policy_obs)
            action = pi.sample(seed=action_rng)
            obsv, env_state, reward, done, info = env.step(
                step_rngs, env_state, action, env_params
            )
            current_pos = env_state.org_obs[..., : raw_goal_batch.shape[-1]]
            dist = jnp.linalg.norm(current_pos - raw_goal_batch, axis=-1)
            active = 1.0 - ever_done
            # Ignore post-auto-reset steps after the first episode done.
            if use_distance_in_competence:
                success = jnp.where(active > 0, dist, jnp.inf)
            else:
                success = (dist <= 1.0).astype(dist.dtype) * active
            ever_done = jnp.maximum(ever_done, done.astype(ever_done.dtype))
            return (obsv, env_state, rng, ever_done), success

        init_ever_done = jnp.zeros((num_envs_per_goal,), dtype=obsv.dtype)
        _, successes = jax.lax.scan(
            step_fn,
            (obsv, env_state, rng, init_ever_done),
            None,
            length=max_steps,
        )
        if use_distance_in_competence:
            return successes.min(axis=0).mean()
        return successes.max(axis=0).mean()

    vmap_goals = jax.vmap(eval_one_goal, in_axes=(0, 0))
    goal_rngs = jax.random.split(jax.random.PRNGKey(42), goals.shape[0])
    return vmap_goals(goal_rngs, goals)


def evaluate_student_env_goal(
    env,
    brax_env,
    network,
    params,
    num_envs,
    rng,
    max_steps=1000,
    warmup_env_state=None,
    normalize_obs=False,
    condition_on_goal=True,
):
    """Evaluate student on env goals (no teacher sampling).

    Conditions the policy on each env's own maze goal from
    ``org_obs[..., -goal_dim:]`` (2D for ant, 3D for humanoid).
    Returns ``(success_rate, episodic_return)`` averaged over ``num_envs``.
    """
    env_params = None
    obs_dim = brax_env.observation_size
    eval_stats = (
        _collapse_obs_norm_stats(warmup_env_state, obs_dim)
        if normalize_obs and warmup_env_state is not None
        else warmup_env_state
    )
    norm_mean = eval_stats.mean if eval_stats is not None else None
    norm_var = eval_stats.var if eval_stats is not None else None

    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    if eval_stats is not None:
        obsv, env_state = env.reset_with_stats(reset_rngs, eval_stats, env_params)
    else:
        obsv, env_state = env.reset(reset_rngs, env_params)

    brax_state = _inner_brax_state(env_state)
    if normalize_obs:
        org_obs = env_state.org_obs
        obsv = (brax_state.obs - norm_mean) / jnp.sqrt(norm_var + 1e-8)
    else:
        org_obs = brax_state.obs
        obsv = brax_state.obs

    # Teacher/conditioning goal equals the environment goal (no teacher sampling).
    env_goal_dim = int(getattr(brax_env, "goal_indices", jnp.array([0, 1])).shape[0])
    env_goals = org_obs[..., -env_goal_dim:]
    # jax.debug.print("env_goals is {x}", x=env_goals)
    if condition_on_goal:
        if normalize_obs:
            goal_batch = _normalize_xy(env_goals, norm_mean, norm_var)
        else:
            goal_batch = env_goals
    # jax.debug.print("goal_batch is {x}", x=goal_batch)

    def step_fn(carry, _):
        obsv, env_state, rng, ep_return, ep_success, ever_done = carry
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        step_rngs = jax.random.split(step_rng, num_envs)
        if condition_on_goal:
            policy_obs = jnp.concatenate([obsv, goal_batch], axis=-1)
        else:
            policy_obs = obsv
        pi, _ = network.apply(params, policy_obs)
        action = pi.sample(seed=action_rng)
        obsv, env_state, reward, done, info = env.step(
            step_rngs, env_state, action, env_params
        )
        active = 1.0 - ever_done
        step_success = _success_metric(env_state)
        ep_success = jnp.maximum(ep_success, step_success * active)
        ep_return = ep_return + reward * active
        ever_done = jnp.maximum(ever_done, done.astype(ever_done.dtype))
        return (obsv, env_state, rng, ep_return, ep_success, ever_done), None

    init_return = jnp.zeros((num_envs,), dtype=obsv.dtype)
    init_success = jnp.zeros((num_envs,), dtype=obsv.dtype)
    init_ever_done = jnp.zeros((num_envs,), dtype=obsv.dtype)
    (_, _, _, ep_return, ep_success, _), _ = jax.lax.scan(
        step_fn,
        (obsv, env_state, rng, init_return, init_success, init_ever_done),
        None,
        length=max_steps,
    )
    return ep_success.mean(), ep_return.mean()


def parse_config_from_cli() -> TrainConfig:
    return tyro.cli(TrainConfig)


def _compute_teacher_softmax_snapshot_indices(
    num_updates: int, num_snapshots: int
) -> np.ndarray:
    """Evenly spaced update indices for in-training teacher softmax visuals."""
    if num_snapshots <= 0 or num_updates <= 0:
        return np.array([], dtype=np.int64)
    count = min(num_snapshots, num_updates)
    if count == 1:
        return np.array([num_updates - 1], dtype=np.int64)
    return np.unique(
        np.linspace(0, num_updates - 1, count, dtype=np.int64)
    )


def _verify_teacher_softmax_snapshot_indices(
    indices: np.ndarray, num_updates: int, num_snapshots: int
) -> None:
    """Validate precomputed snapshot indices for common scheduling invariants."""
    expected_count = (
        0 if num_snapshots <= 0 or num_updates <= 0 else min(num_snapshots, num_updates)
    )
    assert len(indices) == expected_count
    if len(indices) == 0:
        return
    assert indices.min() >= 0
    assert indices.max() <= num_updates - 1
    assert np.all(np.diff(indices) > 0)


def plot_teacher_goal_grid_heatmap(
    goal_grid_xy,
    values,
    num_points,
    *,
    colorbar_label="Value",
    cmap="jet",
    vmin=None,
    vmax=None,
    start_xy=None,
    title=None,
    save_path=None,
):
    """Heatmap of per-goal scalar values over the teacher goal grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))

    goal_grid_xy = np.asarray(goal_grid_xy)
    values = np.asarray(values).reshape(-1)
    gx = goal_grid_xy[:, 0].reshape(num_points, num_points)
    gy = goal_grid_xy[:, 1].reshape(num_points, num_points)
    vgrid = values.reshape(num_points, num_points)

    mesh_kwargs = {"shading": "nearest", "cmap": cmap}
    if vmin is not None or vmax is not None:
        mesh_kwargs["vmin"] = vmin
        mesh_kwargs["vmax"] = vmax
    mesh = ax.pcolormesh(gx, gy, vgrid, **mesh_kwargs)
    fig.colorbar(mesh, ax=ax, label=colorbar_label)

    if start_xy is not None:
        start_xy = np.asarray(start_xy).reshape(-1)
        ax.scatter(
            start_xy[0],
            start_xy[1],
            s=200,
            marker="*",
            c="tab:red",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
            label="Agent start",
        )
        ax.legend(loc="best")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_teacher_softmax(
    goal_grid_xy,
    probs,
    num_points,
    *,
    start_xy=None,
    title=None,
    save_path=None,
    cmap="jet",
):
    """Heatmap of the teacher's categorical distribution over its goal grid."""
    return plot_teacher_goal_grid_heatmap(
        goal_grid_xy,
        probs,
        num_points,
        colorbar_label="P(goal)",
        cmap=cmap,
        start_xy=start_xy,
        title=title,
        save_path=save_path,
    )


def plot_teacher_empowerment_reward_heatmap(
    goal_grid_xy,
    values,
    num_points,
    *,
    start_xy=None,
    title=None,
    save_path=None,
    cmap="jet",
):
    """Heatmap of cached empowerment reward per teacher goal.

    Values are min-max scaled to [-1, 1] so the color scale is fixed.
    """
    values = np.asarray(values).reshape(-1).astype(np.float32)
    v_range = float(np.max(values) - np.min(values)) if values.size else 0.0
    if v_range > 0.0:
        values = 2.0 * (values - np.min(values)) / v_range - 1.0
    else:
        values = np.zeros_like(values)
    return plot_teacher_goal_grid_heatmap(
        goal_grid_xy,
        values,
        num_points,
        colorbar_label="Normalized empowerment reward",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        start_xy=start_xy,
        title=title,
        save_path=save_path,
    )


def plot_teacher_softmax_bar(probs, *, title=None, save_path=None):
    """Bar chart of P(goal_i) over discrete teacher goal indices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = np.asarray(probs).reshape(-1)
    num_goals = probs.shape[0]
    goal_indices = np.arange(num_goals)
    fig_width = max(8.0, num_goals * 0.01)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    colors = plt.cm.viridis(probs / max(probs.max(), 1e-8))
    ax.bar(goal_indices, probs, width=0.85, color=colors, edgecolor="none")
    y_max = max(float(probs.max()) * 1.05, 1e-6)
    if y_max > 0.95:
        y_max = 1.0
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Goal index")
    ax.set_ylabel("P(goal)")

    num_ticks = min(10, num_goals)
    if num_goals > 1:
        tick_positions = np.linspace(0, num_goals - 1, num_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(t)) for t in tick_positions])

    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_competence_vector_bar(competence, *, title=None, save_path=None):
    """Bar chart of student competence per discrete goal index."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    competence = np.asarray(competence).reshape(-1)
    num_goals = competence.shape[0]
    goal_indices = np.arange(num_goals)
    fig_width = max(8.0, num_goals * 0.01)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    colors = plt.cm.viridis(competence / max(competence.max(), 1e-8))
    ax.bar(goal_indices, competence, width=0.85, color=colors, edgecolor="none")
    y_max = max(float(competence.max()) * 1.05, 1e-6)
    if y_max > 0.95:
        y_max = 1.0
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Goal index")
    ax.set_ylabel("Competence")

    num_ticks = min(10, num_goals)
    if num_goals > 1:
        tick_positions = np.linspace(0, num_goals - 1, num_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(t)) for t in tick_positions])

    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def _scatter_goal_learning_progress(cache, goal_idx, lp_values, done):
    updated = jnp.where(
        done,
        lp_values,
        cache[goal_idx],
    )
    return cache.at[goal_idx].set(updated)


def _scatter_goal_empowerment_reward(cache, goal_idx, emp_values, done):
    updated = jnp.where(
        done,
        emp_values,
        cache[goal_idx],
    )
    return cache.at[goal_idx].set(updated)


def _global_gradient_norm(grads):
    flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
    return jnp.linalg.norm(flat_grads)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class TeacherTrainState(TrainState):
    """TrainState that also tracks an exponential moving average of params."""

    ema_params: Any


class TeacherActorCritic(nn.Module):
    num_actions: int
    obs_dim: int
    competence_dim: int
    student_action_dim: int
    activation: str = "tanh"
    hidden_dim: int = 128
    use_encoders: bool = False

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        if self.use_encoders:
            obs = x[..., :self.obs_dim]
            competence = x[..., self.obs_dim:self.obs_dim+self.competence_dim]
            student_action = x[..., self.obs_dim+self.competence_dim:self.obs_dim+self.competence_dim+self.student_action_dim]
            obs = nn.Dense(
                            32,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0),
                        )(obs)
            competence = nn.Dense(
                            32,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0),
                        )(competence)
            student_action = nn.Dense(
                        32,
                        kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0),
                    )(student_action)
            x = jnp.concatenate([obs, competence, student_action], axis=-1)
        else:
            x = x
        actor_mean = act(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        )
        actor_mean = act(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor_mean)
        )
        logits = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=logits)
        critic = act(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        )
        critic = act(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(critic)
        )
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(value, axis=-1)


class EmpowermentEncoder(nn.Module):
    repr_dim: int = 64
    hidden_dim: int = 256
    activation: str = "relu"
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = nn.relu if self.activation == "relu" else nn.tanh
        for _ in range(self.num_layers):
            h = act(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
                )
            x = h
        return nn.Dense(
            self.repr_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(h)


class EmpowermentRepr(NamedTuple):
    action_cond_repr: jnp.ndarray
    context_repr: jnp.ndarray
    future_state_repr: jnp.ndarray
    action_cond_future_state_repr: jnp.ndarray
    context_future_state_repr: jnp.ndarray


class EmpowermentModel(nn.Module):
    """Empowerment representation encoders.

    Encoder 1 (action_cond):
        concat(initial_state, action, goal, competence_vector)
    Encoder 2 (context):
        concat(initial_state, goal, competence_vector)
    Encoder 3 (future_state): future_state only. When
        ``use_separate_future_state_encoders`` is enabled, two independent
        future-state encoders are used for the action-conditioned and context
        energies.

    Each encoder outputs a vector of size ``repr_dim`` (default 64).
    """

    repr_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"
    use_separate_future_state_encoders: bool = False

    def setup(self):
        encoder_kwargs = {
            "repr_dim": self.repr_dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "num_layers": self.num_layers,
        }
        self.action_cond_encoder = EmpowermentEncoder(**encoder_kwargs)
        self.context_encoder = EmpowermentEncoder(**encoder_kwargs)
        self.future_state_encoder = EmpowermentEncoder(**encoder_kwargs)
        if self.use_separate_future_state_encoders:
            self.action_cond_future_state_encoder = EmpowermentEncoder(**encoder_kwargs)
            self.context_future_state_encoder = EmpowermentEncoder(**encoder_kwargs)

    def __call__(
        self,
        initial_state: jnp.ndarray,
        action: jnp.ndarray,
        goal: jnp.ndarray,
        future_state: jnp.ndarray,
        competence_vector: jnp.ndarray,
    ) -> EmpowermentRepr:
        action_cond_in = jnp.concatenate(
            [initial_state, action, goal, competence_vector], axis=-1
        )
        context_in = jnp.concatenate(
            [initial_state, goal, competence_vector], axis=-1
        )
        if self.use_separate_future_state_encoders:
            action_cond_future_state_repr = self.action_cond_future_state_encoder(
                future_state
            )
            context_future_state_repr = self.context_future_state_encoder(
                future_state
            )
        else:
            future_state_repr = self.future_state_encoder(future_state)
            action_cond_future_state_repr = future_state_repr
            context_future_state_repr = future_state_repr
        return EmpowermentRepr(
            action_cond_repr=self.action_cond_encoder(action_cond_in),
            context_repr=self.context_encoder(context_in),
            future_state_repr=action_cond_future_state_repr,
            action_cond_future_state_repr=action_cond_future_state_repr,
            context_future_state_repr=context_future_state_repr,
        )


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    task_reward: jnp.ndarray
    goal_reward: jnp.ndarray
    teacher_reward: jnp.ndarray
    teacher_empowerment_reward: jnp.ndarray
    teacher_success_reward: jnp.ndarray
    teacher_learning_progress_reward: jnp.ndarray
    ppo_updates_per_episode: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    initial_state: jnp.ndarray
    initial_action: jnp.ndarray
    initial_competence: jnp.ndarray
    competence_vector: jnp.ndarray
    goal: jnp.ndarray
    current_state: jnp.ndarray
    current_action: jnp.ndarray
    agent_xy: jnp.ndarray


class TrainRenderBuffer(NamedTuple):
    """Accumulates pipeline states for one training env across updates."""

    frames: Any
    length: jnp.ndarray
    completed_frames: Any
    completed_length: jnp.ndarray
    has_completed: jnp.ndarray
    ref_env_index: jnp.ndarray


def init_train_render_buffer(
    env_state, ref_env_index: jnp.ndarray, max_len: int
) -> TrainRenderBuffer:
    """Allocate a fixed-length buffer from a template pipeline state."""
    brax_state = _inner_brax_state(env_state)
    ref_ps = jax.tree_util.tree_map(
        lambda x: x[ref_env_index], brax_state.pipeline_state
    )
    empty_frames = jax.tree_util.tree_map(
        lambda x: jnp.zeros((max_len,) + x.shape, dtype=x.dtype), ref_ps
    )
    return TrainRenderBuffer(
        frames=empty_frames,
        length=jnp.array(0, dtype=jnp.int32),
        completed_frames=empty_frames,
        completed_length=jnp.array(0, dtype=jnp.int32),
        has_completed=jnp.array(False),
        ref_env_index=jnp.asarray(ref_env_index, dtype=jnp.int32),
    )


def update_train_render_buffer(
    buf: TrainRenderBuffer,
    ref_pipeline_state,
    ref_done: jnp.ndarray,
    max_len: int,
    rng,
    num_envs: int,
) -> tuple[TrainRenderBuffer, Any]:
    """Append one frame; on episode done, snapshot and start the next episode."""

    def _write(frames, length, frame):
        write_idx = jnp.minimum(length, max_len - 1)

        def _set_leaf(buf_leaf, frame_leaf):
            return buf_leaf.at[write_idx].set(frame_leaf)

        new_frames = jax.tree_util.tree_map(_set_leaf, frames, frame)
        new_length = jnp.minimum(length + 1, max_len)
        return new_frames, new_length

    def _on_continue(_):
        frames, length = _write(buf.frames, buf.length, ref_pipeline_state)
        new_buf = TrainRenderBuffer(
            frames=frames,
            length=length,
            completed_frames=buf.completed_frames,
            completed_length=buf.completed_length,
            has_completed=buf.has_completed,
            ref_env_index=buf.ref_env_index,
        )
        return new_buf, rng

    def _on_done(_):
        completed_frames = buf.frames
        completed_length = buf.length
        rng_next, sample_rng = jax.random.split(rng)
        new_ref_env_index = jax.random.randint(
            sample_rng, (), 0, num_envs, dtype=jnp.int32
        )
        new_buf = TrainRenderBuffer(
            frames=jax.tree_util.tree_map(jnp.zeros_like, buf.frames),
            length=jnp.array(0, dtype=jnp.int32),
            completed_frames=completed_frames,
            completed_length=completed_length,
            has_completed=jnp.array(True),
            ref_env_index=new_ref_env_index,
        )
        return new_buf, rng_next

    return jax.lax.cond(ref_done, _on_done, _on_continue, operand=None)


class TeacherEpisodeCarry(NamedTuple):
    teacher_obs: jnp.ndarray
    goal_idx: jnp.ndarray
    raw_goal: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray


class TeacherRolloutTransition(NamedTuple):
    teacher_obs: jnp.ndarray
    goal_idx: jnp.ndarray
    raw_goal: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray


class TeacherRolloutBuffer(NamedTuple):
    teacher_obs: jnp.ndarray
    goal_idx: jnp.ndarray
    raw_goal: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    write_idx: jnp.ndarray
    count: jnp.ndarray


class TeacherFlatBatch(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray


class AgentEpisodeCarry(NamedTuple):
    initial_state: jnp.ndarray
    initial_action: jnp.ndarray
    initial_competence: jnp.ndarray


class AgentEpisodeBuffer(NamedTuple):
    initial_state: jnp.ndarray
    initial_action: jnp.ndarray
    initial_competence: jnp.ndarray
    competence_vector: jnp.ndarray
    goal: jnp.ndarray
    current_state: jnp.ndarray
    current_action: jnp.ndarray
    done: jnp.ndarray
    future_state: jnp.ndarray


class AgentEpisodeChunk(NamedTuple):
    initial_state: jnp.ndarray
    initial_action: jnp.ndarray
    initial_competence: jnp.ndarray
    competence_vector: jnp.ndarray
    goal: jnp.ndarray
    current_state: jnp.ndarray
    current_action: jnp.ndarray
    done: jnp.ndarray


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (
            jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1) + 1e-6
        )
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(
            jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1)
        )
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(
            jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits)
            - jax.nn.logsumexp(logits, axis=1)
            - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def sample_future_states(rng, obs, dones, gamma):
    """Sample one discounted future state per timestep and environment.

    Parameters
    ----------
    rng : jax.random.PRNGKey
    obs : jnp.ndarray, shape (T, N, ...)
        Buffered observations for T timesteps across N environments.
    dones : jnp.ndarray, shape (T, N)
        Terminal flags aligned with ``obs``.
    gamma : float
        Discount factor for geometric sampling over future timesteps.
    """
    max_steps = obs.shape[0]
    num_envs = obs.shape[1]
    all_indices = jnp.arange(max_steps, dtype=jnp.int32)
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, max_steps)

    dones = dones.astype(bool).at[-1].set(jnp.ones((num_envs,), dtype=bool))
    gamma = jnp.asarray(gamma, dtype=jnp.float32)

    def _sample_one_step(_, inputs):
        i, rng_i = inputs
        valid_done = dones & (all_indices[:, None] >= i)
        first_done_after_i = jnp.argmax(valid_done, axis=0)
        diff = all_indices - i
        mask = (all_indices[:, None] >= i) & (
            all_indices[:, None] <= first_done_after_i[None, :]
        )
        diff = jnp.maximum(diff, 0)
        probs = jnp.power(gamma, diff.astype(jnp.float32))[:, None]
        probs = jnp.where(mask, probs, 0.0)
        probs_sum = jnp.sum(probs, axis=0, keepdims=True)
        probs = probs / jnp.maximum(probs_sum, 1e-12)

        sampled_t = jax.random.categorical(
            rng_i, jnp.log(jnp.clip(probs.T, a_min=1e-20, a_max=1.0)), axis=-1
        )
        future_obs_i = obs[sampled_t, env_indices]
        return None, future_obs_i

    _, future_obs = jax.lax.scan(
        _sample_one_step,
        None,
        (all_indices, rngs),
    )
    return future_obs


def init_agent_episode_carry(num_envs, base_obs_dim, action_dim, num_competence, dtype):
    return AgentEpisodeCarry(
        initial_state=jnp.zeros((num_envs, base_obs_dim), dtype=dtype),
        initial_action=jnp.zeros((num_envs, action_dim), dtype=dtype),
        initial_competence=jnp.zeros((num_envs, num_competence), dtype=dtype),
    )


def init_agent_episode_buffer(
    buffer_size, num_envs, base_obs_dim, action_dim, num_competence, goal_dim, dtype
):
    return AgentEpisodeBuffer(
        initial_state=jnp.zeros((buffer_size, num_envs, base_obs_dim), dtype=dtype),
        initial_action=jnp.zeros((buffer_size, num_envs, action_dim), dtype=dtype),
        initial_competence=jnp.zeros(
            (buffer_size, num_envs, num_competence), dtype=dtype
        ),
        competence_vector=jnp.zeros(
            (buffer_size, num_envs, num_competence), dtype=dtype
        ),
        goal=jnp.zeros((buffer_size, num_envs, goal_dim), dtype=dtype),
        current_state=jnp.zeros((buffer_size, num_envs, base_obs_dim), dtype=dtype),
        current_action=jnp.zeros((buffer_size, num_envs, action_dim), dtype=dtype),
        done=jnp.zeros((buffer_size, num_envs), dtype=bool),
        future_state=jnp.zeros((buffer_size, num_envs, base_obs_dim), dtype=dtype),
    )


def write_agent_episode_chunk(buffer, ptr, chunk):
    def _write(field, chunk_field):
        return jax.lax.dynamic_update_slice_in_dim(
            field, chunk_field, ptr, axis=0
        )

    return AgentEpisodeBuffer(
        initial_state=_write(buffer.initial_state, chunk.initial_state),
        initial_action=_write(buffer.initial_action, chunk.initial_action),
        initial_competence=_write(buffer.initial_competence, chunk.initial_competence),
        competence_vector=_write(buffer.competence_vector, chunk.competence_vector),
        goal=_write(buffer.goal, chunk.goal),
        current_state=_write(buffer.current_state, chunk.current_state),
        current_action=_write(buffer.current_action, chunk.current_action),
        done=_write(buffer.done, chunk.done),
        future_state=buffer.future_state,
    )


def fill_agent_episode_future_states(buffer, rng, gamma):
    future_state = sample_future_states(
        rng, buffer.current_state, buffer.done, gamma
    )
    return buffer._replace(future_state=future_state)


def compute_teacher_empowerment_reward(
    apply_fn,
    params,
    energy_name,
    initial_state,
    action,
    goal,
    future_state,
    competence_vector,
):
    def _forward(p):
        return apply_fn(
            p,
            initial_state,
            action,
            goal,
            future_state,
            competence_vector,
        )

    repr = jax.lax.stop_gradient(_forward(params))
    e13 = energy_fn(
        energy_name, repr.action_cond_repr, repr.action_cond_future_state_repr
    )
    e23 = energy_fn(
        energy_name, repr.context_repr, repr.context_future_state_repr
    )
    return jax.lax.stop_gradient(e13 - e23)


def compute_teacher_empowerment_reward_batch(
    apply_fn,
    params,
    energy_name,
    initial_state,
    action,
    goal,
    future_state,
    competence_vector,
):
    return jax.vmap(
        lambda init_s, a, g, fut, comp: compute_teacher_empowerment_reward(
            apply_fn, params, energy_name, init_s, a, g, fut, comp
        )
    )(initial_state, action, goal, future_state, competence_vector)


def compute_teacher_episode_empowerment_sum(
    rng,
    apply_fn,
    params,
    energy_name,
    buffer,
    episode_len,
    gamma,
):
    """Episode-summed empowerment reward for the teacher, one value per env.

    R_n = sum_{t < L_n} emp(s_0, a_t, g, C_0, s_f^(t)), where s_f^(t) is a
    truncated-geometric future of row t within the current episode and rows
    t >= L_n (left over from earlier episodes) are masked out.
    """
    buffer_rows = buffer.current_state.shape[0]
    rows = jnp.arange(buffer_rows)[:, None]
    effective_done = jnp.where(
        rows == (episode_len[None, :] - 1),
        True,
        jnp.where(rows >= episode_len[None, :], True, buffer.done),
    )
    future = sample_future_states(rng, buffer.current_state, effective_done, gamma)

    def bcast(x):
        return jnp.broadcast_to(x[None], (buffer_rows,) + x.shape)

    emp = jax.vmap(
        compute_teacher_empowerment_reward_batch,
        in_axes=(None, None, None, 0, 0, 0, 0, 0),
    )(
        apply_fn,
        params,
        energy_name,
        bcast(buffer.initial_state[0]),
        buffer.current_action,
        bcast(buffer.goal[0]),
        future,
        bcast(buffer.initial_competence[0]),
    )
    valid = rows < episode_len[None, :]
    return jax.lax.stop_gradient(jnp.where(valid, emp, 0.0).sum(axis=0))


def evaluate_teacher_empowerment_reward_grid(
    apply_fn,
    params,
    energy_name,
    initial_state,
    action,
    future_state,
    competence_vector,
    goal_grid,
):
    """Evaluate empowerment reward for every goal on the teacher grid."""
    num_goals = goal_grid.shape[0]
    init_s = jnp.broadcast_to(initial_state, (num_goals,) + initial_state.shape)
    init_a = jnp.broadcast_to(action, (num_goals,) + action.shape)
    fut = jnp.broadcast_to(future_state, (num_goals,) + future_state.shape)
    comp = jnp.broadcast_to(competence_vector, (num_goals,) + competence_vector.shape)
    return jax.vmap(
        lambda init_s_i, init_a_i, goal_i, fut_i, comp_i: (
            compute_teacher_empowerment_reward(
                apply_fn, params, energy_name, init_s_i, init_a_i, goal_i, fut_i, comp_i
            )
        )
    )(init_s, init_a, goal_grid, fut, comp)


def evaluate_teacher_empowerment_reward_grid_over_futures(
    apply_fn,
    params,
    energy_name,
    initial_state,
    actions,
    future_states,
    competence_vector,
    goal_grid,
):
    """Mean over sampled (a_t, s_f^(t)) pairs of the empowerment reward per goal.

    ``actions[i]`` and ``future_states[i]`` must come from the same row t.
    The mean is proportional to the episode-summed teacher reward.
    """
    num_goals = goal_grid.shape[0]
    num_futures = future_states.shape[0]

    def bcast(x, n):
        return jnp.broadcast_to(x, (n,) + x.shape)

    # Context encoder ignores the action and future inputs.
    context_repr = jax.lax.stop_gradient(
        apply_fn(
            params,
            bcast(initial_state, num_goals),
            bcast(actions[0], num_goals),
            goal_grid,
            bcast(future_states[0], num_goals),
            bcast(competence_vector, num_goals),
        )
    ).context_repr
    # Future-state encoders ignore every other input.
    future_repr = jax.lax.stop_gradient(
        apply_fn(
            params,
            bcast(initial_state, num_futures),
            actions,
            bcast(goal_grid[0], num_futures),
            future_states,
            bcast(competence_vector, num_futures),
        )
    )
    num_pairs = num_goals * num_futures
    action_cond_repr = jax.lax.stop_gradient(
        apply_fn(
            params,
            bcast(initial_state, num_pairs),
            jnp.tile(actions, (num_goals, 1)),
            jnp.repeat(goal_grid, num_futures, axis=0),
            jnp.tile(future_states, (num_goals, 1)),
            bcast(competence_vector, num_pairs),
        )
    ).action_cond_repr.reshape(num_goals, num_futures, -1)

    e13 = energy_fn(
        energy_name,
        action_cond_repr,
        future_repr.action_cond_future_state_repr[None, :, :],
    )
    e23 = energy_fn(
        energy_name,
        context_repr[:, None, :],
        future_repr.context_future_state_repr[None, :, :],
    )
    return (e13 - e23).mean(axis=1)


def sample_empowerment_grid_inputs(
    rng,
    buffer,
    episode_len,
    ref_env_index,
    num_futures,
    gamma,
):
    """Sample (a_t, s_f^(t)) pairs from the reference env's current episode.

    t is uniform over the episode's valid rows (matching the unweighted sum
    over t), and s_f^(t) is a truncated-geometric future of row t.
    """
    ref_env_index = jnp.asarray(ref_env_index, dtype=jnp.int32)
    episode_len = jnp.maximum(episode_len, jnp.array(1, dtype=jnp.int32))
    rows = jnp.arange(buffer.current_state.shape[0])
    t_rng, future_rng = jax.random.split(rng)
    t_indices = jax.random.randint(t_rng, (num_futures,), 0, episode_len)
    offsets = rows[None, :] - t_indices[:, None]
    valid = (offsets >= 0) & (rows[None, :] < episode_len)
    logits = jnp.where(
        valid,
        jnp.maximum(offsets, 0).astype(jnp.float32) * jnp.log(gamma),
        -jnp.inf,
    )
    future_indices = jax.random.categorical(future_rng, logits, axis=-1)
    return (
        buffer.initial_state[0, ref_env_index],
        buffer.current_action[t_indices, ref_env_index],
        buffer.current_state[future_indices, ref_env_index],
        buffer.initial_competence[0, ref_env_index],
    )


def plot_teacher_empowerment_grid(
    goal_grid_xy,
    values,
    num_points,
    *,
    start_xy=None,
    title=None,
    save_path=None,
):
    """Plot empowerment reward averaged over future states on the goal grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    goal_grid_xy = np.asarray(goal_grid_xy)
    values = np.asarray(values).reshape(-1)
    gx = goal_grid_xy[:, 0].reshape(num_points, num_points)
    gy = goal_grid_xy[:, 1].reshape(num_points, num_points)
    value_grid = values.reshape(num_points, num_points)
    value_grid = 2 * ((value_grid - np.min(value_grid)) / (np.max(value_grid) - np.min(value_grid))) - 1
    abs_max = float(np.max(np.abs(value_grid))) if value_grid.size else 0.0
    limit = 1.0

    fig, ax = plt.subplots(figsize=(6.5, 6))
    mesh = ax.pcolormesh(
        gx,
        gy,
        value_grid,
        shading="nearest",
        cmap="jet",
        vmin=-limit,
        vmax=limit,
    )
    fig.colorbar(mesh, ax=ax, label="Empowerment reward")
    if start_xy is not None:
        start_xy = np.asarray(start_xy).reshape(-1)
        ax.scatter(
            start_xy[0],
            start_xy[1],
            s=200,
            marker="*",
            c="tab:red",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
            label="Agent start",
        )
        ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def write_episode_step(buffer, ptr, step, is_episode_start):
    num_envs = step.current_state.shape[0]
    env_idx = jnp.arange(num_envs)
    cleared_done = jnp.where(
        is_episode_start[None, :],
        jnp.zeros_like(buffer.done),
        buffer.done,
    )
    buffer = buffer._replace(done=cleared_done)
    return AgentEpisodeBuffer(
        initial_state=buffer.initial_state.at[ptr, env_idx].set(step.initial_state),
        initial_action=buffer.initial_action.at[ptr, env_idx].set(step.initial_action),
        initial_competence=buffer.initial_competence.at[ptr, env_idx].set(
            step.initial_competence
        ),
        competence_vector=buffer.competence_vector.at[ptr, env_idx].set(
            step.competence_vector
        ),
        goal=buffer.goal.at[ptr, env_idx].set(step.goal),
        current_state=buffer.current_state.at[ptr, env_idx].set(step.current_state),
        current_action=buffer.current_action.at[ptr, env_idx].set(
            step.current_action
        ),
        done=buffer.done.at[ptr, env_idx].set(step.done),
        future_state=buffer.future_state,
    )


def flatten_cl_buffer(buffer):
    """Flatten CL buffer from (T, N, ...) to (T * N, ...) per field."""

    def _flatten(field):
        return field.reshape((field.shape[0] * field.shape[1],) + field.shape[2:])

    return AgentEpisodeBuffer(
        initial_state=_flatten(buffer.initial_state),
        initial_action=_flatten(buffer.initial_action),
        initial_competence=_flatten(buffer.initial_competence),
        competence_vector=_flatten(buffer.competence_vector),
        goal=_flatten(buffer.goal),
        current_state=_flatten(buffer.current_state),
        current_action=_flatten(buffer.current_action),
        done=_flatten(buffer.done),
        future_state=_flatten(buffer.future_state),
    )


def init_teacher_episode_carry(num_envs, teacher_obs_dim, goal_dim, dtype):
    return TeacherEpisodeCarry(
        teacher_obs=jnp.zeros((num_envs, teacher_obs_dim), dtype=dtype),
        goal_idx=jnp.zeros((num_envs,), dtype=jnp.int32),
        raw_goal=jnp.zeros((num_envs, goal_dim), dtype=dtype),
        log_prob=jnp.zeros((num_envs,), dtype=dtype),
        value=jnp.zeros((num_envs,), dtype=dtype),
    )


def teacher_carry_from_act(raw_goal, goal_idx, log_prob, value, teacher_obs):
    return TeacherEpisodeCarry(
        teacher_obs=teacher_obs,
        goal_idx=goal_idx,
        raw_goal=raw_goal,
        log_prob=log_prob,
        value=value,
    )


def init_teacher_rollout_buffer(buffer_size, num_envs, teacher_obs_dim, goal_dim, dtype):
    return TeacherRolloutBuffer(
        teacher_obs=jnp.zeros((buffer_size, num_envs, teacher_obs_dim), dtype=dtype),
        goal_idx=jnp.zeros((buffer_size, num_envs), dtype=jnp.int32),
        raw_goal=jnp.zeros((buffer_size, num_envs, goal_dim), dtype=dtype),
        log_prob=jnp.zeros((buffer_size, num_envs), dtype=dtype),
        value=jnp.zeros((buffer_size, num_envs), dtype=dtype),
        reward=jnp.zeros((buffer_size, num_envs), dtype=dtype),
        write_idx=jnp.array(0, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
    )


def push_teacher_transition(buffer, transition):
    idx = buffer.write_idx % buffer.teacher_obs.shape[0]
    buffer_size = buffer.teacher_obs.shape[0]
    return TeacherRolloutBuffer(
        teacher_obs=buffer.teacher_obs.at[idx].set(transition.teacher_obs),
        goal_idx=buffer.goal_idx.at[idx].set(transition.goal_idx),
        raw_goal=buffer.raw_goal.at[idx].set(transition.raw_goal),
        log_prob=buffer.log_prob.at[idx].set(transition.log_prob),
        value=buffer.value.at[idx].set(transition.value),
        reward=buffer.reward.at[idx].set(transition.reward),
        write_idx=buffer.write_idx + 1,
        count=jnp.minimum(buffer.count + 1, buffer_size),
    )


def _where_done(done, fresh, old):
    if old.ndim == 1:
        return jnp.where(done, fresh, old)
    return jnp.where(done[:, None], fresh, old)


def push_teacher_rollout_on_done(buffer, carry, teacher_reward, done):
    transition = TeacherRolloutTransition(
        teacher_obs=carry.teacher_obs,
        goal_idx=carry.goal_idx,
        raw_goal=carry.raw_goal,
        log_prob=carry.log_prob,
        value=carry.value,
        reward=teacher_reward,
    )
    return jax.lax.cond(
        jnp.any(done),
        lambda b: push_teacher_transition(b, transition),
        lambda b: b,
        buffer,
    )


def teacher_buffer_mean_reward(buffer):
    buffer_size = buffer.teacher_obs.shape[0]
    count = buffer.count
    indices = jnp.arange(buffer_size)
    mask = indices < count
    masked_rewards = jnp.where(mask[:, None], buffer.reward, 0.0)
    total = masked_rewards.sum()
    num_valid = count * buffer.reward.shape[1]
    return total / jnp.maximum(num_valid.astype(buffer.reward.dtype), 1.0)


def flatten_teacher_rollout_buffer(buffer, buffer_size):
    """Flatten valid ring-buffer slots into a terminal teacher PPO batch."""
    num_envs = buffer.teacher_obs.shape[1]
    start = (buffer.write_idx - buffer_size) % buffer_size
    slot_indices = (start + jnp.arange(buffer_size)) % buffer_size

    def _gather_and_flatten(field):
        gathered = field[slot_indices]
        return gathered.reshape((buffer_size * num_envs,) + gathered.shape[2:])

    return TeacherFlatBatch(
        obs=_gather_and_flatten(buffer.teacher_obs),
        action=_gather_and_flatten(buffer.goal_idx),
        value=_gather_and_flatten(buffer.value),
        reward=_gather_and_flatten(buffer.reward),
        log_prob=_gather_and_flatten(buffer.log_prob),
    )


def reset_teacher_rollout_buffer(buffer):
    return TeacherRolloutBuffer(
        teacher_obs=jnp.zeros_like(buffer.teacher_obs),
        goal_idx=jnp.zeros_like(buffer.goal_idx),
        raw_goal=jnp.zeros_like(buffer.raw_goal),
        log_prob=jnp.zeros_like(buffer.log_prob),
        value=jnp.zeros_like(buffer.value),
        reward=jnp.zeros_like(buffer.reward),
        write_idx=jnp.array(0, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
    )


def save_checkpoint(train_state, checkpoint_dir):
    """Save a Flax TrainState."""
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        checkpoint_dir,
        train_state,
        force=True,  # Overwrite if the checkpoint already exists
    )
def load_checkpoint(train_state, checkpoint_dir):
    """Load a Flax TrainState.

    Args:
        train_state: An initialized TrainState with the same structure as the
            saved checkpoint.
        checkpoint_dir: Path to the checkpoint directory.

    Returns:
        A TrainState containing the restored parameters and optimizer state.
    """
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer.restore(
        checkpoint_dir,
        train_state,
    )


def load_agent_positions(save_dir: str, index: int):
    """Load agent positions saved as `{index}_agent_positions.npz`.

    Returns a dict with keys `agent_xy` (numpy array) and `update_idx`.
    """
    import numpy as _np

    fname = os.path.join(save_dir, f"{int(index)}_agent_positions.npz")
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    with _np.load(fname) as data:
        agent_xy = data["agent_xy"].copy()
        update_idx = int(data["update_idx"]) if "update_idx" in data else int(index)
    return {"agent_xy": agent_xy, "update_idx": update_idx}


def extract_obs_norm_stats(env_state, expected_obs_dim):
    """Find observation normalization stats in nested wrapped env state.

    Returns ``(mean, var, count)``. ``mean`` / ``var`` are flattened to shape
    ``(expected_obs_dim,)`` when present; ``count`` may be ``None``.
    """
    current = env_state
    while True:
        if hasattr(current, "mean") and hasattr(current, "var"):
            mean = current.mean
            if jnp.ndim(mean) > 0 and mean.shape[-1] == expected_obs_dim:
                count = getattr(current, "count", None)
                if mean.ndim > 1:
                    flat_mean = mean.reshape((-1, expected_obs_dim))
                    flat_var = current.var.reshape((-1, expected_obs_dim))
                    return flat_mean[0], flat_var[0], count
                return mean, current.var, count
        if not hasattr(current, "env_state"):
            break
        current = current.env_state
    return None, None, None


def make_teacher_goal_set(config):
    env_name = config["ENV_NAME"]
    teacher_num_goal_points = int(config["TEACHER_NUM_GOAL_POINTS"])

    if "ant" in env_name:
        from envs.ant_maze import all_possible_goals, get_maze_xy_bounds
        from envs.ant_maze import (
            U_MAZE,
            BIG_MAZE,
            BIG_MAZE_ALL_GOALS,
            U_MAZE_ALL_STATES,
            HARDEST_MAZE,
            HARDEST_MAZE_ALL_GOALS,
        )

        if "u_maze" in env_name:
            maze_layout = U_MAZE
            all_goals_layout = U_MAZE_ALL_STATES
            custom_goal = jnp.array([12.0, 8.0])
        elif "big_maze" in env_name:
            maze_layout = BIG_MAZE
            all_goals_layout = BIG_MAZE_ALL_GOALS
            custom_goal = jnp.array([12.0, 8.0])
        elif "hardest_maze" in env_name:
            maze_layout = HARDEST_MAZE
            all_goals_layout = HARDEST_MAZE_ALL_GOALS
            custom_goal = jnp.array([28.0, 40.0])
        else:
            raise ValueError(f"Unknown maze layout: {env_name}")
        min_x, max_x, min_y, max_y = get_maze_xy_bounds(maze_layout)
        xs = jnp.linspace(min_x, max_x, teacher_num_goal_points)
        ys = jnp.linspace(min_y, max_y, teacher_num_goal_points)
        gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
        goal_grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
        replace_idx = jnp.argmin(jnp.sum((goal_grid - custom_goal) ** 2, axis=-1))
        goal_grid = goal_grid.at[replace_idx].set(custom_goal)
        num_teacher_goals = teacher_num_goal_points * teacher_num_goal_points
        all_goals = all_possible_goals(all_goals_layout)
        num_competence = int(all_goals.shape[0])
    elif "humanoid" in env_name:
        from envs.humanoid_maze import (
            TARGET_Z_COORD,
            all_possible_goals,
            get_maze_xy_bounds,
            U_MAZE,
            BIG_MAZE,
            BIG_MAZE_ALL_GOALS,
            U_MAZE_ALL_STATES,
        )

        if "u_maze" in env_name:
            maze_layout = U_MAZE
            all_goals_layout = U_MAZE_ALL_STATES
        elif "big_maze" in env_name:
            maze_layout = BIG_MAZE
            all_goals_layout = BIG_MAZE_ALL_GOALS
        else:
            raise ValueError(f"Unknown maze layout: {env_name}")
        min_x, max_x, min_y, max_y = get_maze_xy_bounds(
            maze_layout, size_scaling=2.0
        )
        xs = jnp.linspace(min_x, max_x, teacher_num_goal_points)
        ys = jnp.linspace(min_y, max_y, teacher_num_goal_points)
        gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
        zs = jnp.full(gx.size, TARGET_Z_COORD)
        goal_grid = jnp.stack([gx.ravel(), gy.ravel(), zs], axis=-1)
        custom_goal = jnp.array([6.0, 4.0, TARGET_Z_COORD])
        replace_idx = jnp.argmin(jnp.sum((goal_grid - custom_goal) ** 2, axis=-1))
        goal_grid = goal_grid.at[replace_idx].set(custom_goal)
        num_teacher_goals = teacher_num_goal_points * teacher_num_goal_points
        all_goals = all_possible_goals(all_goals_layout, size_scaling=2.0)
        num_competence = int(all_goals.shape[0])
    else:
        raise ValueError(f"Unknown env for teacher goal set: {env_name}")

    return (
        goal_grid,
        teacher_num_goal_points,
        num_teacher_goals,
        all_goals,
        num_competence,
    )


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config.setdefault("GAMMA_CL", 0.99)
    config.setdefault("CL_BUFFER_SIZE", config["EPISODE_LENGTH"])
    # assert config["CL_BUFFER_SIZE"] > 0, "CL_BUFFER_SIZE must be positive"
    assert (
        config["CL_BUFFER_SIZE"] >= config["NUM_STEPS"]
    ), "CL_BUFFER_SIZE must be >= NUM_STEPS"
    teacher_batch_size = (
        config["TEACHER_ROLLOUT_BUFFER_SIZE"] * config["NUM_ENVS"]
    )
    config["TEACHER_MINIBATCH_SIZE"] = (
        teacher_batch_size // config["TEACHER_NUM_MINIBATCHES"]
    )
    print(teacher_batch_size)
    print(config["TEACHER_MINIBATCH_SIZE"])
    print(config["TEACHER_NUM_MINIBATCHES"])
    assert (
        teacher_batch_size
        == config["TEACHER_MINIBATCH_SIZE"] * config["TEACHER_NUM_MINIBATCHES"]
    ), (
        "teacher batch size must equal "
        "TEACHER_MINIBATCH_SIZE * TEACHER_NUM_MINIBATCHES"
    )
    full_empowerment_batch_size = config["CL_BUFFER_SIZE"] * config["NUM_ENVS"]
    empowerment_subsample_size = int(config.get("EMPOWERMENT_SUBSAMPLE_SIZE", 0))
    if empowerment_subsample_size > 0:
        effective_empowerment_batch_size = empowerment_subsample_size
    else:
        effective_empowerment_batch_size = full_empowerment_batch_size
    assert (
        effective_empowerment_batch_size <= full_empowerment_batch_size
    ), "EMPOWERMENT_SUBSAMPLE_SIZE must be <= CL_BUFFER_SIZE * NUM_ENVS"
    config["EMPOWERMENT_MINIBATCH_SIZE"] = (
        effective_empowerment_batch_size // config["EMPOWERMENT_NUM_MINIBATCHES"]
    )
    assert (
        effective_empowerment_batch_size
        == config["EMPOWERMENT_MINIBATCH_SIZE"]
        * config["EMPOWERMENT_NUM_MINIBATCHES"]
    ), (
        "effective empowerment batch size must equal "
        "EMPOWERMENT_MINIBATCH_SIZE * EMPOWERMENT_NUM_MINIBATCHES"
    )
    # minibatch_size = config["EMPOWERMENT_MINIBATCH_SIZE"]
    # print(f"empowerment minibatch size: {minibatch_size}")
    # quit()
    env_kwargs = config.get("ENV_KWARGS", {})
    custom_env = make_custom_env(
        env_name=config["ENV_NAME"],
        backend=config.get("ENV_BACKEND"),
        env_kwargs=env_kwargs,
    )
    custom_env_2 = make_custom_env(
        env_name=config["ENV_NAME"],
        backend=config.get("ENV_BACKEND"),
        env_kwargs=env_kwargs,
    )
    add_goal_reward = config.get("ADD_GOAL_REWARD", False)
    condition_on_goal = config.get("CONDITION_ON_GOAL", False)
    goal_reach_epsilon = config.get("GOAL_REACH_EPSILON", 0.5)
    if custom_env is not None:
        base_env = custom_env
        base_env_2 = custom_env_2
    else:
        base_env = brax_envs.get_environment(
            env_name=config["ENV_NAME"],
            backend=config.get("ENV_BACKEND", "positional"),
        )
        base_env_2 = brax_envs.get_environment(
            env_name=config["ENV_NAME"],
            backend=config.get("ENV_BACKEND", "positional"),
        )
    env = BraxGymnaxWrapper(
        env=base_env,
        episode_length=config.get("EPISODE_LENGTH", 1000),
        action_repeat=config.get("ACTION_REPEAT", 1),
    )
    env_2 = BraxGymnaxWrapper(
        env=base_env_2,
        episode_length=config.get("EPISODE_LENGTH", 1000),
        action_repeat=config.get("ACTION_REPEAT", 1),
    )
    env_params = None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    env_2 = LogWrapper(env_2)
    env_2 = ClipAction(env_2)
    env_2 = VecEnv(env_2)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env_2 = NormalizeVecObservation(env_2)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    teacher_num_minibatches = int(config["TEACHER_NUM_MINIBATCHES"])
    teacher_update_epochs = int(config["TEACHER_UPDATE_EPOCHS"])
    teacher_gamma = config["TEACHER_GAMMA"]
    teacher_gae_lambda = config["TEACHER_GAE_LAMBDA"]
    teacher_clip_eps = config["TEACHER_CLIP_EPS"]
    teacher_vf_coef = config["TEACHER_VF_COEF"]
    teacher_ent_coef = config["TEACHER_ENT_COEF"]
    teacher_rollout_buffer_size = int(config["TEACHER_ROLLOUT_BUFFER_SIZE"])
    gamma_cl = config["GAMMA_CL"]
    cl_buffer_size = int(config["CL_BUFFER_SIZE"])
    empowerment_num_minibatches = int(config["EMPOWERMENT_NUM_MINIBATCHES"])
    empowerment_update_epochs = int(config["EMPOWERMENT_UPDATE_EPOCHS"])
    empowerment_energy_fn = config["EMPOWERMENT_ENERGY_FN"]
    empowerment_contrastive_loss = config["EMPOWERMENT_CONTRASTIVE_LOSS"]
    should_save_agent_trajectory_xy = bool(
        config.get("SAVE_AGENT_TRAJECTORY_XY", False)
    )
    agent_trajectory_ref_env_index = int(
        config.get("AGENT_TRAJECTORY_REF_ENV_INDEX", 0)
    )
    action_dim = int(env.action_space(env_params).shape[0])
    approx_episode_cycles = (
        config["TOTAL_TIMESTEPS"]
        // config["NUM_ENVS"]
        // config.get("EPISODE_LENGTH", 1000)
    )
    approx_total_episode_completions = (
    config["TOTAL_TIMESTEPS"] // config.get("EPISODE_LENGTH", 1000)
    )
    teacher_num_updates = max(
        1, approx_total_episode_completions // teacher_rollout_buffer_size
    )

    def teacher_linear_schedule(count):
        frac = (
            1.0
            - (
                count
                // (teacher_num_minibatches * teacher_update_epochs)
            )
            / teacher_num_updates
        )
        frac = jnp.maximum(frac, 0.000000001)
        return config["TEACHER_LR"] * frac

    empowerment_num_updates = max(1, approx_episode_cycles)

    def empowerment_linear_schedule(count):
        frac = (
            1.0
            - (
                count
                // (empowerment_num_minibatches * empowerment_update_epochs)
            )
            / empowerment_num_updates
        )
        return config["EMPOWERMENT_LR"] * frac

    network = ActorCritic(
        env.action_space(env_params).shape[0], activation=config["ACTIVATION"], hidden_dim=config["HIDDEN_DIM"]
    )
    base_obs_dim = int(env.observation_space(env_params).shape[0])
    (
        goal_grid,
        teacher_num_goal_points,
        num_teacher_goals,
        all_goals,
        num_competence,
    ) = make_teacher_goal_set(config)
    goal_dim = int(goal_grid.shape[-1])
    condition_teacher_on_competence = config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
    use_average_competence_reward = config.get("USE_AVERAGE_COMPETENCE_REWARD", False)
    use_learning_progress_reward = config.get("USE_LEARNING_PROGRESS_REWARD", False)
    use_teacher_empowerment_reward = config.get("USE_TEACHER_EMPOWERMENT_REWARD", False)
    # teacher_empowerment_reward_coef = config.get("TEACHER_EMPOWERMENT_REWARD_COEF", 1.0)
    update_competence = (
        condition_teacher_on_competence or use_average_competence_reward
    )
    teacher_obs_dim = (
        base_obs_dim
        + (num_competence if condition_teacher_on_competence else 0)
        + action_dim
    )
    teacher_network = TeacherActorCritic(
        num_actions=teacher_num_goal_points * teacher_num_goal_points,
        obs_dim=base_obs_dim, 
        competence_dim=num_competence if condition_teacher_on_competence else 0,
        student_action_dim=action_dim,
        activation=config["TEACHER_ACTIVATION"],
        hidden_dim=config["TEACHER_HIDDEN_DIM"],
        use_encoders=config["TEACHER_USE_ENCODERS"]
    )
    empowerment_network = EmpowermentModel(
        repr_dim=int(config["EMPOWERMENT_REPR_DIM"]),
        hidden_dim=int(config["EMPOWERMENT_HIDDEN_DIM"]),
        activation="relu",
        num_layers=config["EMPOWERMENT_NUM_LAYERS"],
        use_separate_future_state_encoders=config.get(
            "USE_SEPARATE_FUTURE_STATE_ENCODERS", False
        ),
    )

    def _extract_obs_norm_stats(env_state, expected_obs_dim):
        """Find observation normalization stats in nested wrapped env state."""
        mean, var, _count = extract_obs_norm_stats(env_state, expected_obs_dim)
        return mean, var

    def _normalize_eval_obs(obs, obs_mean, obs_var):
        if obs_mean is None or obs_var is None:
            return obs
        if obs.ndim > 1 and obs.shape[0] == 1:
            obs = obs[0]
        return (obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)

    def _build_teacher_input(obs, competence_vector, action):
        if obs.ndim == 1:
            obs = obs[None, :]
        if action.ndim == 1:
            action = action[None, :]
        # import pdb; pdb.set_trace()
        action_batch = jnp.broadcast_to(action, (obs.shape[0], action.shape[-1]))
        inputs = [obs]
        if condition_teacher_on_competence:
            comp_batch = jnp.broadcast_to(
                competence_vector, (obs.shape[0], competence_vector.shape[0])
            )
            inputs.append(comp_batch)
        inputs.append(action_batch)
        return jnp.concatenate(inputs, axis=-1)

    def _teacher_act(teacher_params, obs, competence_vector, action, rng):
        teacher_obs = _build_teacher_input(obs, competence_vector, action)
        pi, value = teacher_network.apply(teacher_params, teacher_obs)
        goal_idx = pi.sample(seed=rng)
        raw_goal = goal_grid[goal_idx].astype(jnp.float32)
        log_prob = pi.log_prob(goal_idx)
        return raw_goal, goal_idx, log_prob, value, teacher_obs

    teacher_net_obs_dim = base_obs_dim
    teacher_net_competence_dim = (
        num_competence if condition_teacher_on_competence else 0
    )
    teacher_net_action_dim = action_dim

    def _teacher_input_grad_norms(teacher_params, teacher_obs):
        """Frobenius norms / mean-|grad| of d(logits)/d(input) per input block.

        Differentiates teacher Categorical logits (not the discrete sample) w.r.t.
        the concatenated teacher input ``[obs | competence | action]``, then
        slices by ``teacher_net_*_dim``. Missing blocks return NaN.
        """
        x = jnp.asarray(teacher_obs).reshape(-1)

        def _logits(inp):
            pi, _ = teacher_network.apply(teacher_params, inp[None, ...])
            return jnp.asarray(pi.logits).reshape(-1)

        # (num_actions, teacher_obs_dim)
        jacobian = jax.jacrev(_logits)(x)

        def _block_stats(start, dim):
            if dim <= 0:
                nan = jnp.asarray(jnp.nan, dtype=jacobian.dtype)
                return nan, nan
            block = jacobian[:, start : start + dim]
            return jnp.linalg.norm(block), jnp.mean(jnp.abs(block))

        offset = 0
        norm_state, mean_abs_state = _block_stats(offset, teacher_net_obs_dim)
        offset += teacher_net_obs_dim
        norm_competence, mean_abs_competence = _block_stats(
            offset, teacher_net_competence_dim
        )
        offset += teacher_net_competence_dim
        norm_action, mean_abs_action = _block_stats(
            offset, teacher_net_action_dim
        )
        return (
            norm_state,
            norm_competence,
            norm_action,
            mean_abs_state,
            mean_abs_competence,
            mean_abs_action,
        )

    def _log_teacher_input_grads_to_wandb(step, grad_stats):
        """Host-side wandb logging for teacher input-gradient norms."""
        if config.get("WANDB_MODE", "disabled") != "online":
            return
        (
            norm_state,
            norm_competence,
            norm_action,
            mean_abs_state,
            mean_abs_competence,
            mean_abs_action,
        ) = grad_stats
        payload = {}
        if np.isfinite(float(norm_state)):
            payload["teacher/grad_norm_vs_state"] = float(norm_state)
            payload["teacher/grad_mean_abs_vs_state"] = float(mean_abs_state)
        if np.isfinite(float(norm_competence)):
            payload["teacher/grad_norm_vs_competence"] = float(norm_competence)
            payload["teacher/grad_mean_abs_vs_competence"] = float(
                mean_abs_competence
            )
        if np.isfinite(float(norm_action)):
            payload["teacher/grad_norm_vs_action"] = float(norm_action)
            payload["teacher/grad_mean_abs_vs_action"] = float(mean_abs_action)
        if payload:
            wandb.log(payload, step=int(step))

    def _sample_teacher_goals(
        teacher_params, obs, competence_vector, action, rng
    ):
        raw_goal, _, _, _, _ = _teacher_act(
            teacher_params, obs, competence_vector, action, rng
        )
        return raw_goal

    def _policy_goal_from_raw(raw_goal, obs_mean, obs_var):
        if config["NORMALIZE_ENV"]:
            return _normalize_xy(raw_goal, obs_mean, obs_var)
        return raw_goal

    render_sim_steps = int(config.get("EVAL_RENDER_STEPS", 500))
    render_max_frames = int(config.get("EVAL_RENDER_MAX_FRAMES", 1000))
    render_action_repeat = int(config.get("ACTION_REPEAT", 1))
    action_low = jnp.asarray(env.action_space(env_params).low)
    action_high = jnp.asarray(env.action_space(env_params).high)
    _render_obs_dim = int(env.observation_space(env_params).shape[0])

    def _obs_norm_stats_for_render(final_env_state):
        obs_mean, obs_var = _extract_obs_norm_stats(final_env_state, _render_obs_dim)
        if obs_mean is None:
            obs_mean = jnp.zeros(_render_obs_dim)
            obs_var = jnp.ones(_render_obs_dim)
        return obs_mean, obs_var

    def _eval_render_action(params, obs, obs_mean, obs_var, policy_goal=None):
        norm_obs = _normalize_eval_obs(obs, obs_mean, obs_var)
        if condition_on_goal:
            policy_obs = jnp.concatenate([norm_obs, policy_goal], axis=-1)
        else:
            policy_obs = norm_obs
        pi, _ = network.apply(params, policy_obs)
        return jnp.clip(pi.mean(), action_low, action_high)

    def _sample_render_action(params, obs, obs_mean, obs_var, policy_goal, rng):
        norm_obs = _normalize_eval_obs(obs, obs_mean, obs_var)
        if condition_on_goal:
            policy_obs = jnp.concatenate([norm_obs, policy_goal], axis=-1)
        else:
            policy_obs = norm_obs
        pi, _ = network.apply(params, policy_obs)
        return jnp.clip(pi.sample(seed=rng), action_low, action_high)

    def _bootstrap_teacher_action(student_params, base_obs, rng):
        policy_obs = base_obs
        if condition_on_goal:
            policy_obs = jnp.concatenate(
                [base_obs, jnp.zeros((goal_dim,), dtype=base_obs.dtype)], axis=-1
            )
        pi, _ = network.apply(student_params, policy_obs)
        return pi.sample(seed=rng)

    def _render_rollout_impl(
        params, teacher_params, competence_vector, rng, obs_mean, obs_var
    ):
        rng, reset_rng = jax.random.split(rng)
        state = base_env.reset(reset_rng)
        raw_goal = jnp.zeros((goal_dim,), dtype=jnp.float32)
        policy_goal = jnp.zeros((goal_dim,), dtype=jnp.float32)
        if condition_on_goal:
            rng, action_rng, teacher_rng = jax.random.split(rng, 3)
            norm_obs = _normalize_eval_obs(state.obs, obs_mean, obs_var)
            bootstrap_action = _sample_render_action(
                params, state.obs, obs_mean, obs_var, policy_goal, action_rng
            )
            raw_goal = _sample_teacher_goals(
                teacher_params,
                norm_obs,
                competence_vector,
                bootstrap_action,
                teacher_rng,
            )
            if raw_goal.ndim > 1:
                raw_goal = raw_goal[0]
            policy_goal = _policy_goal_from_raw(raw_goal, obs_mean, obs_var)

        def step_fn(carry, _):
            state, raw_goal, policy_goal, rng = carry
            rng, action_rng, bootstrap_rng, goal_rng = jax.random.split(rng, 4)
            if condition_on_goal:
                action = _sample_render_action(
                    params, state.obs, obs_mean, obs_var, policy_goal, action_rng
                )
            else:
                action = _sample_render_action(
                    params, state.obs, obs_mean, obs_var, policy_goal, action_rng
                )

            def repeat_step(s, __):
                return base_env.step(s, action), None

            state, _ = jax.lax.scan(
                repeat_step, state, None, length=render_action_repeat
            )
            if condition_on_goal:
                norm_obs = _normalize_eval_obs(state.obs, obs_mean, obs_var)
                bootstrap_goal = jnp.zeros_like(policy_goal)
                bootstrap_action = _sample_render_action(
                    params,
                    state.obs,
                    obs_mean,
                    obs_var,
                    bootstrap_goal,
                    bootstrap_rng,
                )
                new_raw_goal = _sample_teacher_goals(
                    teacher_params,
                    norm_obs,
                    competence_vector,
                    bootstrap_action,
                    goal_rng,
                )
                if new_raw_goal.ndim > 1:
                    new_raw_goal = new_raw_goal[0]
                new_policy_goal = _policy_goal_from_raw(
                    new_raw_goal, obs_mean, obs_var
                )
                raw_goal = jnp.where(state.done, new_raw_goal, raw_goal)
                policy_goal = jnp.where(state.done, new_policy_goal, policy_goal)
            return (state, raw_goal, policy_goal, rng), state.pipeline_state

        _, pipeline_states = jax.lax.scan(
            step_fn,
            (state, raw_goal, policy_goal, rng),
            None,
            length=render_sim_steps,
        )
        return pipeline_states

    _run_render_rollout = jax.jit(_render_rollout_impl)

    def _subsample_pipeline_states(pipeline_states, max_frames):
        n = jax.tree_util.tree_leaves(pipeline_states)[0].shape[0]
        if n <= max_frames:
            return pipeline_states
        idx = jnp.linspace(0, n - 1, max_frames).astype(jnp.int32)
        return jax.tree_util.tree_map(lambda x: x[idx], pipeline_states)

    def _pipeline_states_to_list(pipeline_states):
        n = int(jax.tree_util.tree_leaves(pipeline_states)[0].shape[0])
        return [
            jax.tree_util.tree_map(lambda x: x[i], pipeline_states)
            for i in range(n)
        ]

    def log_pipeline_html_to_wandb(pipeline_states, step, log_key="render"):
        """Subsample pipeline states, render HTML, and log to wandb."""
        if config.get("WANDB_MODE", "disabled") != "online":
            return
        try:
            n = int(jax.tree_util.tree_leaves(pipeline_states)[0].shape[0])
            if n <= 0:
                return
            pipeline_states = _subsample_pipeline_states(
                pipeline_states, render_max_frames
            )
            rollout = _pipeline_states_to_list(jax.device_get(pipeline_states))
            rendered_html = html.render(
                base_env.sys.tree_replace({"opt.timestep": base_env.dt}),
                rollout,
                height=int(config.get("EVAL_RENDER_HEIGHT", 480)),
            )
            exp_dir = config["EXP_DIR"]
            exp_name = f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
            html_path = os.path.join(
                exp_dir, f"{exp_name}_{log_key.replace('/', '_')}_{int(step)}.html"
            )
            with open(html_path, "w", encoding="utf-8") as file:
                file.write(rendered_html)
            if config.get("EVAL_RENDER_LOG_WANDB_HTML", False):
                wandb.log({log_key: wandb.Html(rendered_html)})
            else:
                wandb.save(html_path, base_path=exp_dir, policy="now")
                wandb.log({f"{log_key}/html_path": html_path})
        except Exception as err:
            print(f"[log_pipeline_html_to_wandb] skipped video logging: {err}")
            traceback.print_exc()

    teacher_softmax_viz_log_wandb = bool(
        config.get("TEACHER_SOFTMAX_VIZ_LOG_WANDB", True)
    )
    teacher_softmax_viz_num_snapshots = int(
        config.get("TEACHER_SOFTMAX_VIZ_NUM_SNAPSHOTS", 0)
    )
    num_updates = int(config["NUM_UPDATES"])
    snapshot_indices = _compute_teacher_softmax_snapshot_indices(
        num_updates, teacher_softmax_viz_num_snapshots
    )
    _verify_teacher_softmax_snapshot_indices(
        snapshot_indices, num_updates, teacher_softmax_viz_num_snapshots
    )
    snapshot_indices_jnp = jnp.asarray(snapshot_indices, dtype=jnp.int32)
    if teacher_softmax_viz_num_snapshots > 0:
        print(
            f"[teacher_softmax_viz] scheduling {len(snapshot_indices)} snapshots "
            f"over {num_updates} updates; "
            f"first={snapshot_indices[:3].tolist()} "
            f"last={snapshot_indices[-3:].tolist()}"
        )

    def log_teacher_softmax_viz(
        probs,
        ref_obs,
        env_state,
        step,
        competence_vector,
        goal_empowerment_reward_cache,
    ):
        """Plot teacher softmax, competence, and empowerment-reward visuals; log to wandb."""
        try:
            import matplotlib.pyplot as plt

            probs = np.asarray(jax.device_get(probs)).reshape(-1)
            competence = np.asarray(jax.device_get(competence_vector)).reshape(-1)
            goal_grid_xy = np.asarray(jax.device_get(goal_grid))
            ref_obs = np.asarray(jax.device_get(ref_obs)).reshape(-1)
            start_xy = ref_obs[:2].copy()

            obs_mean, obs_var = _extract_obs_norm_stats(env_state, base_obs_dim)
            if config.get("NORMALIZE_ENV", False) and obs_mean is not None:
                obs_mean = np.asarray(jax.device_get(obs_mean))
                obs_var = np.asarray(jax.device_get(obs_var))
                start_xy = np.asarray(
                    jax.device_get(_denorm_xy(start_xy, obs_mean, obs_var))
                ).reshape(-1)

            exp_dir = config["EXP_DIR"]
            exp_name = f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
            viz_dir = os.path.join(exp_dir, "teacher_softmax_visuals")
            os.makedirs(viz_dir, exist_ok=True)
            save_path = os.path.join(
                viz_dir, f"{exp_name}_teacher_softmax_{step}.png"
            )
            heatmap_cmap = config.get("TEACHER_HEATMAP_CMAP", "jet")
            fig, _ = plot_teacher_softmax(
                goal_grid_xy,
                probs,
                teacher_num_goal_points,
                start_xy=start_xy,
                title=f'Teacher softmax @ step {step} ({config["ENV_NAME"]})',
                save_path=save_path,
                cmap=heatmap_cmap,
            )
            bar_save_path = os.path.join(
                viz_dir, f"{exp_name}_teacher_softmax_bar_{step}.png"
            )
            bar_fig, _ = plot_teacher_softmax_bar(
                probs,
                title=f'Teacher softmax bar @ step {step} ({config["ENV_NAME"]})',
                save_path=bar_save_path,
            )
            competence_save_path = os.path.join(
                viz_dir, f"{exp_name}_competence_vector_bar_{step}.png"
            )
            competence_fig, _ = plot_competence_vector_bar(
                competence,
                title=f'Competence vector @ step {step} ({config["ENV_NAME"]})',
                save_path=competence_save_path,
            )
            emp_fig = None
            if use_teacher_empowerment_reward:
                emp_values = np.asarray(
                    jax.device_get(goal_empowerment_reward_cache)
                ).reshape(-1)
                emp_save_path = os.path.join(
                    viz_dir, f"{exp_name}_teacher_empowerment_reward_{step}.png"
                )
                emp_fig, _ = plot_teacher_empowerment_reward_heatmap(
                    goal_grid_xy,
                    emp_values,
                    teacher_num_goal_points,
                    start_xy=start_xy,
                    title=(
                        f'Empowerment reward @ step {step} '
                        f'({config["ENV_NAME"]})'
                    ),
                    save_path=emp_save_path,
                    cmap=heatmap_cmap,
                )
            if (
                teacher_softmax_viz_log_wandb
                and config.get("WANDB_MODE", "disabled") == "online"
            ):
                log_payload = {
                    "teacher/goal_softmax": wandb.Image(fig),
                    "teacher/goal_softmax_bar": wandb.Image(bar_fig),
                    "teacher/competence_vector_bar": wandb.Image(competence_fig),
                }
                if emp_fig is not None:
                    log_payload["teacher/empowerment_reward_heatmap"] = wandb.Image(
                        emp_fig
                    )
                wandb.log(log_payload, step=step)
            plt.close(fig)
            plt.close(bar_fig)
            plt.close(competence_fig)
            if emp_fig is not None:
                plt.close(emp_fig)
        except Exception as err:
            print(f"[log_teacher_softmax_viz] skipped teacher visual plots: {err}")
            traceback.print_exc()

    def log_teacher_softmax_snapshot(
        teacher_params,
        student_params,
        ref_obs,
        competence_vector,
        env_state,
        step,
        goal_empowerment_reward_cache,
        rng,
    ):
        """Compute teacher probs and log teacher visual snapshots."""
        ref_obs = jnp.asarray(ref_obs).reshape(-1)
        rng, action_rng = jax.random.split(rng)
        bootstrap_action = _bootstrap_teacher_action(
            student_params, ref_obs, action_rng
        )
        teacher_obs = _build_teacher_input(
            ref_obs, competence_vector, bootstrap_action
        )
        pi, _ = teacher_network.apply(teacher_params, teacher_obs)
        probs = pi.probs.reshape(-1)
        log_teacher_softmax_viz(
            probs,
            ref_obs,
            env_state,
            step,
            competence_vector,
            goal_empowerment_reward_cache,
        )
        if config.get("LOG_TEACHER_INPUT_GRADS", True):
            grad_stats = _teacher_input_grad_norms(teacher_params, teacher_obs)
            _log_teacher_input_grads_to_wandb(step, jax.device_get(grad_stats))

    def render_eval_episode(
        params, teacher_params, competence_vector, rng, final_env_state
    ):
        """Runs one eval rollout and logs rendered HTML to wandb."""
        if config.get("WANDB_MODE", "disabled") != "online":
            return
        try:
            obs_mean, obs_var = _obs_norm_stats_for_render(final_env_state)
            pipeline_states = _run_render_rollout(
                params, teacher_params, competence_vector, rng, obs_mean, obs_var
            )
            num_steps = int(config["TOTAL_TIMESTEPS"])
            log_pipeline_html_to_wandb(pipeline_states, num_steps, log_key="render")

            pipeline_states = _subsample_pipeline_states(
                pipeline_states, render_max_frames
            )
            rollout = _pipeline_states_to_list(jax.device_get(pipeline_states))
            exp_dir = config["EXP_DIR"]
            exp_name = f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
            log_multi_ant_maze = globals().get("_log_multi_ant_maze_top_gif")
            if callable(log_multi_ant_maze):
                log_multi_ant_maze(base_env, rollout, exp_dir, exp_name, num_steps)
        except Exception as err:
            print(f"[render_eval_episode] skipped video logging: {err}")
            traceback.print_exc()

    def train(rng):
        steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
        _wandb_timer = {"last_time": None}
        _best_competence_log_sum = {"value": -float("inf"), "step": -1}
        _teacher_ckpt_host = {"apply_fn": None, "tx": None}
        _student_ckpt_host = {"apply_fn": None, "tx": None}

        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        if condition_on_goal:
            init_x = jnp.concatenate([init_x, jnp.zeros((goal_dim, ))], axis=-1)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        _student_ckpt_host["apply_fn"] = train_state.apply_fn
        _student_ckpt_host["tx"] = train_state.tx
        rng, goal_rng, teacher_init_rng, empowerment_init_rng, _rng = jax.random.split(
            rng, 5
        )
        teacher_init_params = teacher_network.init(
            teacher_init_rng, jnp.zeros((teacher_obs_dim,))
        )
        if config["ANNEAL_LR"]:
            teacher_tx = optax.chain(
                optax.clip_by_global_norm(config["TEACHER_MAX_GRAD_NORM"]),
                optax.adam(learning_rate=teacher_linear_schedule, eps=1e-5),
            )
        else:
            teacher_tx = optax.chain(
                optax.clip_by_global_norm(config["TEACHER_MAX_GRAD_NORM"]),
                optax.adam(config["TEACHER_LR"], eps=1e-5),
            )
        teacher_train_state = TeacherTrainState.create(
            apply_fn=teacher_network.apply,
            params=teacher_init_params,
            tx=teacher_tx,
            ema_params=teacher_init_params,
        )
        _teacher_ckpt_host["apply_fn"] = teacher_train_state.apply_fn
        _teacher_ckpt_host["tx"] = teacher_train_state.tx
        empowerment_init_params = empowerment_network.init(
            empowerment_init_rng,
            jnp.zeros((base_obs_dim,)),
            jnp.zeros((action_dim,)),
            jnp.zeros((goal_dim,)),
            jnp.zeros((base_obs_dim,)),
            jnp.zeros((num_competence,)),
        ) # empowerment model is as follow: 
        if config["ANNEAL_LR"]:
            empowerment_tx = optax.chain(
                optax.clip_by_global_norm(config["EMPOWERMENT_MAX_GRAD_NORM"]),
                optax.adam(learning_rate=empowerment_linear_schedule, eps=1e-5),
            )
        else:
            empowerment_tx = optax.chain(
                optax.clip_by_global_norm(config["EMPOWERMENT_MAX_GRAD_NORM"]),
                optax.adam(config["EMPOWERMENT_LR"], eps=1e-5),
            )
        empowerment_train_state = TrainState.create(
            apply_fn=empowerment_network.apply,
            params=empowerment_init_params,
            tx=empowerment_tx,
        )

        def sample_teacher_goals(obs, competence_vector, action, rng):
            return _sample_teacher_goals(
                teacher_train_state.params, obs, competence_vector, action, rng
            )

        def teacher_act_and_carry(obs, competence_vector, action, rng):
            raw_goal, goal_idx, teacher_log_prob, teacher_value, teacher_obs = (
                _teacher_act(
                    teacher_train_state.params,
                    obs,
                    competence_vector,
                    action,
                    rng,
                )
            )
            carry = teacher_carry_from_act(
                raw_goal, goal_idx, teacher_log_prob, teacher_value, teacher_obs
            )
            return raw_goal, carry

        def compute_competence_vector(student_params, stats_state):
            return evaluate_multiple_goals(
                env_2,
                custom_env_2,
                network,
                student_params,
                all_goals,
                config["NUM_EVAL_ENVS"],
                max_steps=config.get("EPISODE_LENGTH", 1000),
                warmup_env_state=stats_state,
                normalize_obs=config["NORMALIZE_ENV"],
                condition_on_goal=condition_on_goal,
                use_distance_in_competence=config["USE_DISTANCE_IN_COMPETENCE"],
                config=config,
            )

        def evaluate_teacher_goal_success_rates(student_params, stats_state, goals):
            return evaluate_multiple_goals(
                env_2,
                custom_env_2,
                network,
                student_params,
                goals,
                config["NUM_EVAL_ENVS"],
                max_steps=config.get("EPISODE_LENGTH", 1000),
                warmup_env_state=stats_state,
                normalize_obs=config["NORMALIZE_ENV"],
                condition_on_goal=condition_on_goal,
                use_distance_in_competence=config["USE_DISTANCE_IN_COMPETENCE"],
                config=config,
            )

        def evaluate_student_on_env_goal(student_params, stats_state, rng):
            return evaluate_student_env_goal(
                env_2,
                custom_env_2,
                network,
                student_params,
                config["EVAL_NUM_ENVS"],
                rng,
                max_steps=config.get("EPISODE_LENGTH", 1000),
                warmup_env_state=stats_state,
                normalize_obs=config["NORMALIZE_ENV"],
                condition_on_goal=condition_on_goal,
            )

        if config["NORMALIZE_ENV"]:
            # import pdb; pdb.set_trace()
            # env = NormalizeVecReward(env, config["GAMMA"])
            ### run random actions to have a starting estimate of the observation normalization stats
            def run_policy(network_params, rng):
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state = env.reset(reset_rng, env_params)
                if condition_on_goal:
                    obsv = jnp.concatenate([obsv, jnp.zeros((config["NUM_ENVS"], goal_dim))], axis=-1)
                def step_fn(carry, _):
                    obsv, env_state, rng = carry
                    rng, rng_sample, rng_step = jax.random.split(rng, 3)
                    pi, _ = network.apply(network_params, obsv)
                    action = pi.sample(seed=rng_sample)
                    rng_step = jax.random.split(rng_step, config["NUM_ENVS"])
                    obsv, env_state, reward, done, info  = env.step(rng_step, env_state, action, env_params)
                    if condition_on_goal:
                        obsv = jnp.concatenate([obsv, jnp.zeros((config["NUM_ENVS"], goal_dim))], axis=-1)
                    return (obsv, env_state, rng), None
                _, pipeline_states = jax.lax.scan(
                    step_fn,
                    (obsv, env_state, rng),
                    None,
                    length=config["OBS_NORM_WARMUP_STEPS"],
                )
                return env_state
            rng, warmup_rng = jax.random.split(rng)
            warmup_env_state = run_policy(network_params, warmup_rng)
            obs_mean = warmup_env_state.mean
            obs_var = warmup_env_state.var
            # jax.debug.print("obs_mean: {obs_mean}", obs_mean=obs_mean[0])
            # jax.debug.print("obs_var: {obs_var}", obs_var=obs_var[0])

        # INIT ENV
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        if config["NORMALIZE_ENV"]:
            obsv, env_state = env.reset_with_stats(
                reset_rng, warmup_env_state, env_params
            )
            # jax.debug.print(
            #     "post_reset_obs_mean: {obs_mean}", obs_mean=env_state.mean[0]
            # )
            # jax.debug.print(
            #     "post_reset_obs_var: {obs_var}", obs_var=env_state.var[0]
            # )
        else:
            obsv, env_state = env.reset(reset_rng, env_params)
        # jax.debug.print("obsv: {obsv}", obsv=obsv[..., :2])

        episode_initial_base_obs = obsv[..., :base_obs_dim]
        # NOTE: replace this with the competence of the randomly initilized policy
        competence_vector = compute_competence_vector(train_state.params, env_state)
        bootstrap_policy_obs = obsv
        if condition_on_goal:
            bootstrap_policy_obs = jnp.concatenate(
                [bootstrap_policy_obs, jnp.zeros((config["NUM_ENVS"], goal_dim))],
                axis=-1,
            )
        bootstrap_pi, _ = network.apply(train_state.params, bootstrap_policy_obs)
        bootstrap_rng, teacher_rng = jax.random.split(goal_rng)
        bootstrap_action = bootstrap_pi.sample(seed=bootstrap_rng)
        raw_goals, teacher_episode_carry = teacher_act_and_carry(
            obsv[..., :base_obs_dim], competence_vector, bootstrap_action, teacher_rng
        )
        teacher_rollout_buffer = init_teacher_rollout_buffer(
            teacher_rollout_buffer_size,
            config["NUM_ENVS"],
            teacher_obs_dim,
            goal_dim,
            obsv.dtype,
        )
        agent_episode_carry = init_agent_episode_carry(
            config["NUM_ENVS"],
            base_obs_dim,
            action_dim,
            num_competence,
            obsv.dtype,
        )
        cl_buffer = init_agent_episode_buffer(
            cl_buffer_size,
            config["NUM_ENVS"],
            base_obs_dim,
            action_dim,
            num_competence,
            goal_dim,
            obsv.dtype,
        )
        episode_buf_ptr = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
        goals = raw_goals
        if condition_on_goal:
            if config["NORMALIZE_ENV"]:
                goals = _normalize_xy(raw_goals, env_state.mean, env_state.var)
            obsv = jnp.concatenate([obsv, goals], axis=-1)

        if use_learning_progress_reward:
            episode_goal_success_start = evaluate_teacher_goal_success_rates(
                train_state.params, env_state, raw_goals
            )
        else:
            episode_goal_success_start = jnp.zeros(
                (config["NUM_ENVS"],), dtype=obsv.dtype
            )
        episode_step_count = jnp.zeros((config["NUM_ENVS"],), dtype=obsv.dtype)
        goal_learning_progress_cache = jnp.zeros(
            (num_teacher_goals,), dtype=obsv.dtype
        )
        goal_empowerment_reward_cache = jnp.zeros(
            (num_teacher_goals,), dtype=obsv.dtype
        )
        teacher_goal_count_grid = jnp.zeros(
            (num_teacher_goals,), dtype=jnp.int32
        )
        train_render_freq = int(config.get("TRAIN_RENDER_FREQ", 0))
        enable_train_render = train_render_freq > 0
        train_render_max_len = int(config["EPISODE_LENGTH"])
        if enable_train_render:
            rng, train_render_rng = jax.random.split(rng)
            ref_env_index = jax.random.randint(
                train_render_rng, (), 0, config["NUM_ENVS"], dtype=jnp.int32
            )
            train_render_buf = init_train_render_buffer(
                env_state, ref_env_index, train_render_max_len
            )
            # Seed the post-reset frame so the first episode includes t=0.
            ref_ps0 = jax.tree_util.tree_map(
                lambda x: x[ref_env_index],
                _inner_brax_state(env_state).pipeline_state,
            )
            train_render_buf, rng = update_train_render_buffer(
                train_render_buf,
                ref_ps0,
                jnp.array(False),
                train_render_max_len,
                rng,
                config["NUM_ENVS"],
            )
        def plot_teacher_goal_selection_counts(
            goal_grid_xy,
            counts,
            num_points,
            *,
            start_xy=None,
            title=None,
            save_path=None,
            cmap="jet",
        ):
            """Heatmap of normalized teacher goal selection frequency in [0, 1]."""
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 6))
            goal_grid_xy = np.asarray(goal_grid_xy)
            counts = np.asarray(counts).reshape(-1)
            gx = goal_grid_xy[:, 0].reshape(num_points, num_points)
            gy = goal_grid_xy[:, 1].reshape(num_points, num_points)
            cgrid = counts.reshape(num_points, num_points).astype(np.float32)
            max_count = float(np.max(cgrid)) if cgrid.size else 0.0
            normalized_grid = np.zeros_like(cgrid, dtype=np.float32)
            if max_count > 0.0:
                normalized_grid = cgrid / max_count
            mesh = ax.pcolormesh(
                gx,
                gy,
                normalized_grid,
                shading="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )
            fig.colorbar(mesh, ax=ax, label="Normalized goal selection frequency")

            if start_xy is not None:
                start_xy = np.asarray(start_xy).reshape(-1)
                ax.scatter(
                    start_xy[0],
                    start_xy[1],
                    s=200,
                    marker="*",
                    c="tab:red",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=3,
                    label="Agent start",
                )
                ax.legend(loc="best")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="datalim")
            if title is not None:
                ax.set_title(title)
            if save_path is not None:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            return fig, ax

        def log_teacher_goal_selection_count_grid(goal_count_grid, step):
            """Log a count heatmap showing teacher goal selections over time."""
            try:
                if config.get("WANDB_MODE", "disabled") != "online":
                    return
                import matplotlib.pyplot as plt

                counts = np.asarray(jax.device_get(goal_count_grid)).reshape(-1)
                goal_grid_xy = np.asarray(jax.device_get(goal_grid))
                exp_dir = config["EXP_DIR"]
                exp_name = f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
                save_path = os.path.join(
                    exp_dir, "teacher_goal_visuals", f"{exp_name}_teacher_goal_counts_{int(step)}.png"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(
                    save_path.replace(".png", ".npz"),
                    counts=counts,
                    goal_grid_xy=goal_grid_xy,
                    num_points=int(teacher_num_goal_points),
                    step=int(step),
                )
                fig, _ = plot_teacher_goal_selection_counts(
                    goal_grid_xy,
                    counts,
                    teacher_num_goal_points,
                    title=(
                        f"Teacher goal selections @ step {int(step)} "
                        f"({config['ENV_NAME']})"
                    ),
                    save_path=save_path,
                    cmap=config.get("TEACHER_HEATMAP_CMAP", "jet"),
                )
                wandb.log(
                    {"teacher/goal_selection_counts": wandb.Image(fig)},
                )
                plt.close(fig)
            except Exception as err:
                print(
                    f"[log_teacher_goal_selection_count_grid] skipped count-grid visual: {err}"
                )
                traceback.print_exc()

        def log_teacher_empowerment_grid(values, start_xy, step):
            """Log a future-state-averaged empowerment heatmap to WandB."""
            try:
                if config.get("WANDB_MODE", "disabled") != "online":
                    return
                import matplotlib.pyplot as plt

                values = np.asarray(jax.device_get(values)).reshape(-1)
                start_xy = np.asarray(jax.device_get(start_xy)).reshape(-1)
                goal_grid_xy = np.asarray(jax.device_get(goal_grid))
                exp_dir = config["EXP_DIR"]
                exp_name = f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
                save_path = os.path.join(
                    exp_dir, "teacher_empowerment_visuals", f"{exp_name}_teacher_empowerment_grid_{int(step)}.png"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig, _ = plot_teacher_empowerment_grid(
                    goal_grid_xy,
                    values,
                    teacher_num_goal_points,
                    start_xy=start_xy,
                    title=(
                        f"Teacher empowerment reward @ step {int(step)} "
                        f"({config['ENV_NAME']})"
                    ),
                    save_path=save_path,
                )
                wandb.log(
                    {"teacher/empowerment_reward_grid": wandb.Image(fig)},
                )
                plt.close(fig)
            except Exception as err:
                print(
                    f"[log_teacher_empowerment_grid] skipped empowerment visual: {err}"
                )
                traceback.print_exc()

        def log_competence_vector_bar_viz(competence, step):
            """Log competence-vector bar chart to disk and wandb."""
            try:
                import matplotlib.pyplot as plt

                competence = np.asarray(jax.device_get(competence)).reshape(-1)
                exp_dir = config["EXP_DIR"]
                exp_name = f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
                viz_dir = os.path.join(exp_dir, "teacher_softmax_visuals")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(
                    viz_dir,
                    f"{exp_name}_competence_vector_bar_{int(step)}.png",
                )
                fig, _ = plot_competence_vector_bar(
                    competence,
                    title=(
                        f"Competence vector @ step {int(step)} "
                        f"({config['ENV_NAME']})"
                    ),
                    save_path=save_path,
                )
                if (
                    config.get("TEACHER_COMPETENCE_VIZ_LOG_WANDB", True)
                    and config.get("WANDB_MODE", "disabled") == "online"
                ):
                    wandb.log(
                        {"teacher/competence_vector_bar": wandb.Image(fig)},
                        step=int(step),
                    )
                plt.close(fig)
            except Exception as err:
                print(
                    f"[log_competence_vector_bar_viz] skipped competence bar: {err}"
                )
                traceback.print_exc()

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                *body, rng = runner_state
                if enable_train_render:
                    train_render_buf = body[-1]
                    body = body[:-1]
                (
                    train_state,
                    teacher_train_state,
                    empowerment_train_state,
                    env_state,
                    last_obs,
                    goals,
                    raw_goals,
                    competence_vector,
                    task_reward_sum,
                    task_reward_sq_sum,
                    goal_reward_sum,
                    goal_reward_sq_sum,
                    reward_count,
                    episode_success,
                    episode_goal_success_start,
                    episode_step_count,
                    episode_initial_base_obs,
                    goal_learning_progress_cache,
                    goal_empowerment_reward_cache,
                    teacher_episode_carry,
                    teacher_rollout_buffer,
                    agent_episode_carry,
                    cl_buffer,
                    episode_buf_ptr,
                    teacher_goal_count_grid,
                ) = body

                # SELECT ACTION
                rng, goal_rng, _rng, teacher_emp_rng = jax.random.split(rng, 4)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                current_state = last_obs[..., :base_obs_dim]

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # NOTE: this where i will add a goal-reaching reward
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                episode_initial_base_obs = jnp.where(
                    done[:, None],
                    obsv[..., :base_obs_dim],
                    episode_initial_base_obs,
                )
                step_success = _success_metric(env_state)
                episode_success = jnp.maximum(episode_success, step_success)
                # jax.debug.print("done: {done}", done=done)
                task_reward = reward
                goal_reward = jnp.zeros_like(task_reward) 
                ## Calulcate the goal reaching reward for following the teacher instructive goals
                if add_goal_reward:
                    if config["NORMALIZE_ENV"]:
                        agent_pos = env_state.org_obs[..., :goal_dim]
                    else:
                        agent_pos = obsv[..., :goal_dim]
                    dist = jnp.linalg.norm(agent_pos - raw_goals, axis=-1)
                    # jax.debug.print("dist_mean: {dist}", dist=dist.mean())
                    goal_reward = (dist <= goal_reach_epsilon).astype(task_reward.dtype)
                    # jax.debug.print("dist: {dist}", dist=dist)
                    # jax.debug.print("goals: {goals}", goals=goals)
                    # jax.debug.print("goal_reward_mean: {goal_reward}", goal_reward=goal_reward.mean())
                    # import pdb;pdb.set_trace()
                    reward = config["TASK_REWARD_COEF"] * task_reward + config["GOAL_REWARD_COEF"] * goal_reward
                    if config["INTERPOLATED_REWARD"]:
                        reward = (1-config["GOAL_REWARD_COEF"]) * task_reward + config["GOAL_REWARD_COEF"] * goal_reward
                ## Update the teacher's state input
                if update_competence:
                    competence_vector = jax.lax.cond(
                        jnp.any(done),
                        lambda _: compute_competence_vector(
                            train_state.params, env_state
                        ),
                        lambda _: competence_vector,
                        operand=None,
                    )
                    # import pdb;pdb.set_trace()
                    # jax.debug.print("competence_vector: {competence_vector}", competence_vector=competence_vector)
                if use_average_competence_reward:
                    # NOTE: review this
                    avg_competence = competence_vector.mean()
                    competence_part = jnp.where(done, avg_competence, 0.0)
                else:
                    competence_part = jnp.zeros_like(task_reward)
                success_part = jnp.where(done, episode_success, 0.0)

                # NOTE: is this jax jitting friendly?
                # import pdb;pdb.set_trace()
                episode_step_count = episode_step_count + 1.0
                is_episode_start = episode_step_count == 1.0
                # NOTE: what is the shpae of this vector?
                competence_per_env = jnp.broadcast_to(
                    competence_vector, (config["NUM_ENVS"], num_competence)
                )
                # an object to save the initial state, action, and competence vector for the episode and carry them over to the next step.
                # update where the episode step count is 1 which indicates a new starting state from the next episode.
                agent_episode_carry = AgentEpisodeCarry(
                    initial_state=jnp.where(
                        is_episode_start[:, None],
                        current_state,
                        agent_episode_carry.initial_state,
                    ),
                    initial_action=jnp.where(
                        is_episode_start[:, None],
                        action,
                        agent_episode_carry.initial_action,
                    ),
                    initial_competence=jnp.where(
                        is_episode_start[:, None],
                        competence_per_env,
                        agent_episode_carry.initial_competence,
                    ),
                )
                ppo_updates_per_episode = (
                    episode_step_count + config["NUM_STEPS"] - 1
                ) // config["NUM_STEPS"]
                ppo_updates_at_done = jnp.where(done, ppo_updates_per_episode, 0.0)

                # NOTE: this code train the teacher based on empowerment reward, so I need to remove this learning progress part.
                learning_progress_part = jnp.zeros_like(task_reward)
                if use_learning_progress_reward:

                    def _compute_and_cache_learning_progress(_):
                        end_rates = evaluate_teacher_goal_success_rates(
                            train_state.params,
                            env_state,
                            teacher_episode_carry.raw_goal,
                        )
                        learning_progress = (
                            end_rates - episode_goal_success_start
                        )
                        if config["ABSOLUTE_LEARNING_PROGRESS"]:
                            learning_progress = jnp.abs(learning_progress)
                        new_cache = _scatter_goal_learning_progress(
                            goal_learning_progress_cache,
                            teacher_episode_carry.goal_idx,
                            learning_progress,
                            done,
                        )
                        return (
                            jnp.where(done, learning_progress, 0.0),
                            new_cache,
                        )

                    learning_progress_part, goal_learning_progress_cache = (
                        jax.lax.cond(
                            jnp.any(done),
                            _compute_and_cache_learning_progress,
                            lambda _: (
                                learning_progress_part,
                                goal_learning_progress_cache,
                            ),
                            operand=None,
                        )
                    )

                safe_episode_ptr = jnp.minimum(
                    episode_buf_ptr, cl_buffer_size - 1
                )
                # create an episode step object and add the contrastive learning buffer.
                episode_step = AgentEpisodeChunk(
                    initial_state=agent_episode_carry.initial_state,
                    initial_action=agent_episode_carry.initial_action,
                    initial_competence=agent_episode_carry.initial_competence,
                    competence_vector=competence_per_env,
                    goal=raw_goals,
                    current_state=current_state,
                    current_action=action,
                    done=done,
                )
                cl_buffer = write_episode_step(
                    cl_buffer, safe_episode_ptr, episode_step, is_episode_start
                )

                teacher_emp_part = jnp.zeros_like(task_reward)
                if use_teacher_empowerment_reward:
                    episode_len = safe_episode_ptr + 1
                    emp_reward = jax.lax.cond(
                        jnp.any(done),
                        lambda _: compute_teacher_episode_empowerment_sum(
                            teacher_emp_rng,
                            empowerment_network.apply,
                            empowerment_train_state.params,
                            empowerment_energy_fn,
                            cl_buffer,
                            episode_len,
                            gamma_cl,
                        ),
                        lambda _: jnp.zeros_like(task_reward),
                        operand=None,
                    )
                    teacher_emp_part = jnp.where(done, emp_reward, 0.0)
                    goal_empowerment_reward_cache = jax.lax.cond(
                        jnp.any(done),
                        lambda _: _scatter_goal_empowerment_reward(
                            goal_empowerment_reward_cache,
                            teacher_episode_carry.goal_idx,
                            emp_reward,
                            done,
                        ),
                        lambda _: goal_empowerment_reward_cache,
                        operand=None,
                    )

                # compute the final teacher's reward
                teacher_reward = ( config["TASK_REWARD_COEF"] * success_part
                    + config["EMP_REWARD_COEF"] * teacher_emp_part
                )
                # c = config["EMP_REWARD_COEF"]
                # print(f"EMP_REWARD_COEF: {c}")
                # update the buffer pointer
                episode_buf_ptr = jnp.where(
                    done,
                    0,
                    jnp.minimum(episode_buf_ptr + 1, cl_buffer_size - 1),
                )
                episode_success = jnp.where(done, 0.0, episode_success)
                # store done episodes in the teacher's buffer
                # NOTE: I do not understand is this vectorized?
                teacher_rollout_buffer = push_teacher_rollout_on_done(
                    teacher_rollout_buffer,
                    teacher_episode_carry,
                    teacher_reward,
                    done,
                )
                # jax.debug.print("competence_vector: {competence_vector}", competence_vector=competence_vector)
                # import pdb;pdb.set_trace()
                (
                    new_raw_goals,
                    goal_idx,
                    teacher_log_prob,
                    teacher_value,
                    teacher_obs,
                ) = _teacher_act(
                    teacher_train_state.params,
                    current_state,
                    competence_vector,
                    action,
                    goal_rng,
                )
                fresh_carry = teacher_carry_from_act(
                    new_raw_goals,
                    goal_idx,
                    teacher_log_prob,
                    teacher_value,
                    teacher_obs,
                )
                teacher_episode_carry = jax.tree.map(
                    lambda f, o: _where_done(done, f, o),
                    fresh_carry,
                    teacher_episode_carry,
                )
                teacher_goal_count_grid = teacher_goal_count_grid.at[goal_idx].add(
                    jnp.ones_like(goal_idx, dtype=jnp.int32)
                )
                raw_goals = jnp.where(done[:, None], new_raw_goals, raw_goals)
                if use_learning_progress_reward:

                    def _update_episode_start_rates(_):
                        new_start_rates = evaluate_teacher_goal_success_rates(
                            train_state.params, env_state, new_raw_goals
                        )
                        return jnp.where(
                            done, new_start_rates, episode_goal_success_start
                        )

                    episode_goal_success_start = jax.lax.cond(
                        jnp.any(done),
                        _update_episode_start_rates,
                        lambda _: episode_goal_success_start,
                        operand=None,
                    )
                # NOTE: check the shape of episode_step_count, it should be a vecotr of size number of environments.
                episode_step_count = jnp.where(done, 0.0, episode_step_count)
                # jax.debug.print("done: {done}", done=jnp.any(done))
                # jax.lax.cond(
                #     jnp.any(done),
                #     lambda:jax.debug.print("obsv xy: {obsv}", obsv=obsv[..., :2]),
                #     lambda: None,
                # )
                def normalize_goals(raw_goals, env_state):
                    if config["NORMALIZE_ENV"]:
                        return _normalize_xy(raw_goals, env_state.mean, env_state.var)
                    else:
                        return raw_goals
                goals = normalize_goals(raw_goals, env_state)
                if condition_on_goal:
                    obsv = jnp.concatenate([obsv, goals], axis=-1)
                # capture x,y from the inner brax state (original/un-normalized coordinates)
                current_xy = env_state.org_obs[..., :2]
                if enable_train_render:
                    ref_idx = train_render_buf.ref_env_index
                    ref_ps = jax.tree_util.tree_map(
                        lambda x: x[ref_idx],
                        _inner_brax_state(env_state).pipeline_state,
                    )
                    train_render_buf, rng = update_train_render_buffer(
                        train_render_buf,
                        ref_ps,
                        done[ref_idx],
                        train_render_max_len,
                        rng,
                        config["NUM_ENVS"],
                    )
                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    task_reward,
                    goal_reward,
                    teacher_reward,
                    teacher_emp_part,
                    success_part,
                    learning_progress_part,
                    ppo_updates_at_done,
                    log_prob,
                    last_obs,
                    info,
                    agent_episode_carry.initial_state,
                    agent_episode_carry.initial_action,
                    agent_episode_carry.initial_competence,
                    competence_per_env,
                    raw_goals,
                    current_state,
                    action,
                    current_xy,
                )
                runner_state = (
                    train_state,
                    teacher_train_state,
                    empowerment_train_state,
                    env_state,
                    obsv,
                    goals,
                    raw_goals,
                    competence_vector,
                    task_reward_sum,
                    task_reward_sq_sum,
                    goal_reward_sum,
                    goal_reward_sq_sum,
                    reward_count,
                    episode_success,
                    episode_goal_success_start,
                    episode_step_count,
                    episode_initial_base_obs,
                    goal_learning_progress_cache,
                    goal_empowerment_reward_cache,
                    teacher_episode_carry,
                    teacher_rollout_buffer,
                    agent_episode_carry,
                    cl_buffer,
                    episode_buf_ptr,
                    teacher_goal_count_grid,
                )
                if enable_train_render:
                    runner_state = runner_state + (train_render_buf, rng)
                else:
                    runner_state = runner_state + (rng,)
                if should_save_agent_trajectory_xy:
                    agent_xy = _agent_world_xy(
                        obsv,
                        env_state,
                        agent_trajectory_ref_env_index,
                        normalize_env=config["NORMALIZE_ENV"],
                        base_obs_dim=base_obs_dim,
                    )
                    return runner_state, (transition, agent_xy)
                return runner_state, transition

            if should_save_agent_trajectory_xy:
                runner_state, (traj_batch, agent_xy_chunk) = jax.lax.scan(
                    _env_step, runner_state, None, config["NUM_STEPS"]
                )
            else:
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["NUM_STEPS"]
                )

            *body, rng = runner_state
            if enable_train_render:
                train_render_buf = body[-1]
                body = body[:-1]
            (
                train_state,
                teacher_train_state,
                empowerment_train_state,
                env_state,
                last_obs,
                goals,
                raw_goals,
                competence_vector,
                task_reward_sum,
                task_reward_sq_sum,
                goal_reward_sum,
                goal_reward_sq_sum,
                reward_count,
                episode_success,
                episode_goal_success_start,
                episode_step_count,
                episode_initial_base_obs,
                goal_learning_progress_cache,
                goal_empowerment_reward_cache,
                teacher_episode_carry,
                teacher_rollout_buffer,
                agent_episode_carry,
                cl_buffer,
                episode_buf_ptr,
                teacher_goal_count_grid,
            ) = body

            empowerment_should_update = jnp.any(traj_batch.done)

            def _on_episode_done(operand):
                buf, rng_in = operand
                rng_out, future_rng = jax.random.split(rng_in)
                new_buf = fill_agent_episode_future_states(buf, future_rng, gamma_cl)
                return new_buf, rng_out

            def _on_episode_not_done(operand):
                buf, rng_in = operand
                return buf, rng_in

            cl_buffer, rng = jax.lax.cond(
                empowerment_should_update,
                _on_episode_done,
                _on_episode_not_done,
                (cl_buffer, rng),
            )

            # CALCULATE ADVANTAGE
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # jax.debug.print("done: {done}", done=done)
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (value_loss, actor_loss, entropy)), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    gradient_norm = _global_gradient_norm(grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (
                        total_loss,
                        value_loss,
                        actor_loss,
                        entropy,
                        gradient_norm,
                    )

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            total_loss = loss_info[0].mean()
            value_loss = loss_info[1].mean()
            actor_loss = loss_info[2].mean()
            entropy = loss_info[3].mean()
            student_gradient_norm = loss_info[4].mean()

            def _run_empowerment_update(operands):
                e_state, buf, rng_in = operands
                flat_batch = flatten_cl_buffer(buf)
                rng_in, subsample_rng = jax.random.split(rng_in)
                if effective_empowerment_batch_size < full_empowerment_batch_size:
                    subsample_indices = jax.random.permutation(
                        subsample_rng, full_empowerment_batch_size
                    )[:effective_empowerment_batch_size]
                    flat_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, subsample_indices, axis=0), flat_batch
                    )

                def _e_update_epoch(update_state, unused):
                    def _e_update_minibatch(e_state, batch):
                        def _e_loss_fn(params, batch):
                            repr = empowerment_network.apply(
                                params,
                                batch.initial_state,
                                batch.current_action,
                                batch.goal,
                                batch.future_state,
                                batch.initial_competence,
                            )
                            logits_13 = energy_fn(
                                empowerment_energy_fn,
                                repr.action_cond_repr[:, None, :],
                                repr.action_cond_future_state_repr[None, :, :],
                            )
                            logits_23 = energy_fn(
                                empowerment_energy_fn,
                                repr.context_repr[:, None, :],
                                repr.context_future_state_repr[None, :, :],
                            )
                            loss_13 = contrastive_loss_fn(
                                empowerment_contrastive_loss, logits_13
                            )
                            loss_23 = contrastive_loss_fn(
                                empowerment_contrastive_loss, logits_23
                            )
                            return loss_13 + loss_23, (loss_13, loss_23)

                        grad_fn = jax.value_and_grad(_e_loss_fn, has_aux=True)
                        (total_loss, (loss_13, loss_23)), grads = grad_fn(
                            e_state.params, batch
                        )
                        e_state = e_state.apply_gradients(grads=grads)
                        gradient_norm = _global_gradient_norm(grads)
                        return e_state, jnp.stack(
                            [total_loss, loss_13, loss_23, gradient_norm], axis=0
                        )

                    e_state, flat_batch, rng_b = update_state
                    rng_b, _rng_b = jax.random.split(rng_b)
                    permutation = jax.random.permutation(
                        _rng_b, effective_empowerment_batch_size
                    )
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), flat_batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x,
                            [empowerment_num_minibatches, -1] + list(x.shape[1:]),
                        ),
                        shuffled_batch,
                    )
                    e_state, step_losses = jax.lax.scan(
                        _e_update_minibatch, e_state, minibatches
                    )
                    return (e_state, flat_batch, rng_b), step_losses.mean(axis=0)

                rng_in, epoch_rng = jax.random.split(rng_in)
                update_state = (e_state, flat_batch, epoch_rng)
                update_state, e_loss_info = jax.lax.scan(
                    _e_update_epoch, update_state, None, empowerment_update_epochs
                )
                e_state = update_state[0]
                metrics = (
                    e_loss_info[:, 0].mean(),
                    e_loss_info[:, 1].mean(),
                    e_loss_info[:, 2].mean(),
                    e_loss_info[:, 3].mean(),
                    jnp.array(1.0, dtype=jnp.float32),
                )
                return e_state, buf, rng_in, metrics

            def _skip_empowerment_update(operands):
                e_state, buf, rng_in = operands
                z = jnp.array(0.0, dtype=jnp.float32)
                return e_state, buf, rng_in, (z, z, z, z, z)

            (
                empowerment_train_state,
                cl_buffer,
                rng,
                empowerment_metrics,
            ) = jax.lax.cond(
                empowerment_should_update,
                _run_empowerment_update,
                _skip_empowerment_update,
                (empowerment_train_state, cl_buffer, rng),
            )
            empowerment_total_loss = empowerment_metrics[0]
            empowerment_loss_enc1_enc3 = empowerment_metrics[1]
            empowerment_loss_enc2_enc3 = empowerment_metrics[2]
            empowerment_gradient_norm = empowerment_metrics[3]
            empowerment_did_update = empowerment_metrics[4]
            empowerment_current_lr = (
                empowerment_linear_schedule(empowerment_train_state.step)
                if config["ANNEAL_LR"]
                else config["EMPOWERMENT_LR"]
            )

            if config.get("TEACHER_EMPOWERMENT_GRID_VIZ_LOG_WANDB", True):
                empowerment_grid_freq = int(
                    config.get("TEACHER_EMPOWERMENT_GRID_VIZ_FREQ", 25)
                )
                if empowerment_grid_freq > 0:
                    should_log_empowerment_grid = (
                        (update_idx) % empowerment_grid_freq == 0
                    )

                    def _run_empowerment_grid_viz(_):
                        ref_env_index = int(
                            config.get("TEACHER_EMPOWERMENT_GRID_REF_ENV_INDEX", 0)
                        )
                        num_futures = int(
                            config.get("TEACHER_EMPOWERMENT_GRID_NUM_FUTURES", 64)
                        )
                        viz_rng, next_rng = jax.random.split(rng)
                        episode_len = jnp.minimum(
                            episode_buf_ptr[ref_env_index] + 1,
                            cl_buffer_size,
                        )
                        (
                            initial_state,
                            grid_actions,
                            future_states,
                            initial_competence,
                        ) = sample_empowerment_grid_inputs(
                            viz_rng,
                            cl_buffer,
                            episode_len,
                            ref_env_index,
                            num_futures,
                            gamma_cl,
                        )
                        values = (
                            evaluate_teacher_empowerment_reward_grid_over_futures(
                                empowerment_network.apply,
                                empowerment_train_state.params,
                                empowerment_energy_fn,
                                initial_state,
                                grid_actions,
                                future_states,
                                initial_competence,
                                goal_grid,
                            )
                        )
                        if config.get("NORMALIZE_ENV", False):
                            start_xy = (
                                initial_state[:2]
                                * jnp.sqrt(env_state.var[ref_env_index, :2] + 1e-8)
                                + env_state.mean[ref_env_index, :2]
                            )
                        else:
                            start_xy = initial_state[:2]
                        jax.debug.callback(
                            log_teacher_empowerment_grid,
                            values,
                            start_xy,
                            (update_idx) * config["NUM_STEPS"] * config["NUM_ENVS"],
                        )
                        return next_rng

                    def _skip_empowerment_grid_viz(_):
                        return rng

                    rng = jax.lax.cond(
                        should_log_empowerment_grid,
                        _run_empowerment_grid_viz,
                        _skip_empowerment_grid_viz,
                        operand=None,
                    )

            teacher_should_update = (
                teacher_rollout_buffer.count >= teacher_rollout_buffer_size
            ) & (
                use_average_competence_reward
                | use_learning_progress_reward
                | use_teacher_empowerment_reward
            )

            def _teacher_calculate_gae(traj, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done = jnp.array(1.0, dtype=transition.value.dtype)
                    delta = (
                        transition.reward
                        + teacher_gamma * next_value * (1 - done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + teacher_gamma * teacher_gae_lambda * (1 - done) * gae
                    )
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.array(0.0, dtype=last_val.dtype), last_val),
                    traj,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj.value

            def _run_teacher_update(operands):
                t_train_state, t_buffer, t_rng = operands
                flat_batch = flatten_teacher_rollout_buffer(
                    t_buffer, teacher_rollout_buffer_size
                )
                last_val = jnp.array(0.0, dtype=flat_batch.value.dtype)
                advantages, targets = _teacher_calculate_gae(flat_batch, last_val)

                def _t_update_epoch(update_state, unused):
                    def _t_update_minibatch(t_state, batch_info):
                        traj_b, gae_b, tgt_b = batch_info

                        def _t_loss_fn(params, traj_b, gae, targets):
                            pi, value = teacher_network.apply(params, traj_b.obs)
                            log_prob = pi.log_prob(traj_b.action)
                            value_pred_clipped = traj_b.value + (
                                value - traj_b.value
                            ).clip(-teacher_clip_eps, teacher_clip_eps)
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(
                                value_pred_clipped - targets
                            )
                            value_loss = 0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()
                            ratio = jnp.exp(log_prob - traj_b.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - teacher_clip_eps,
                                    1.0 + teacher_clip_eps,
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                            entropy = pi.entropy().mean()
                            total_loss = (
                                loss_actor
                                + teacher_vf_coef * value_loss
                                - teacher_ent_coef * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_t_loss_fn, has_aux=True)
                        (total_loss, (value_loss, actor_loss, entropy)), grads = grad_fn(
                            t_state.params, traj_b, gae_b, tgt_b
                        )
                        gradient_norm = _global_gradient_norm(grads)
                        t_state = t_state.apply_gradients(grads=grads)
                        return t_state, (
                            total_loss,
                            value_loss,
                            actor_loss,
                            entropy,
                            gradient_norm,
                        )

                    t_state, traj_b, advantages_b, targets_b, rng_b = update_state
                    rng_b, _rng_b = jax.random.split(rng_b)
                    permutation = jax.random.permutation(_rng_b, teacher_batch_size)
                    batch = (traj_b, advantages_b, targets_b)
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x,
                            [teacher_num_minibatches, -1] + list(x.shape[1:]),
                        ),
                        shuffled_batch,
                    )
                    t_state, total_loss = jax.lax.scan(
                        _t_update_minibatch, t_state, minibatches
                    )
                    update_state = (
                        t_state,
                        traj_b,
                        advantages_b,
                        targets_b,
                        rng_b,
                    )
                    return update_state, total_loss

                t_rng, epoch_rng = jax.random.split(t_rng)
                update_state = (
                    t_train_state,
                    flat_batch,
                    advantages,
                    targets,
                    epoch_rng,
                )
                update_state, t_loss_info = jax.lax.scan(
                    _t_update_epoch, update_state, None, teacher_update_epochs
                )
                t_train_state = update_state[0]
                new_buffer = reset_teacher_rollout_buffer(t_buffer)
                metrics = (
                    t_loss_info[0].mean(),
                    t_loss_info[1].mean(),
                    t_loss_info[2].mean(),
                    t_loss_info[3].mean(),
                    t_loss_info[4].mean(),
                    jnp.array(1.0, dtype=jnp.float32),
                    jnp.array(float(teacher_batch_size), dtype=jnp.float32),
                )
                return t_train_state, new_buffer, t_rng, metrics

            def _skip_teacher_update(operands):
                t_train_state, t_buffer, t_rng = operands
                z = jnp.array(0.0, dtype=jnp.float32)
                return t_train_state, t_buffer, t_rng, (z, z, z, z, z, z, z)

            (
                teacher_train_state,
                teacher_rollout_buffer,
                rng,
                teacher_metrics,
            ) = jax.lax.cond(
                teacher_should_update,
                _run_teacher_update,
                _skip_teacher_update,
                (teacher_train_state, teacher_rollout_buffer, rng),
            )
            teacher_total_loss = teacher_metrics[0]
            teacher_value_loss = teacher_metrics[1]
            teacher_actor_loss = teacher_metrics[2]
            teacher_entropy = teacher_metrics[3]
            teacher_gradient_norm = teacher_metrics[4]
            teacher_did_update = teacher_metrics[5]
            teacher_update_batch_size = teacher_metrics[6]
            teacher_ema_coeff = float(config.get("TEACHER_EMA_COEFF", 0.99))

            def _update_teacher_ema(operands):
                t_state = operands
                new_ema = jax.tree_util.tree_map(
                    lambda e, p: teacher_ema_coeff * e + (1.0 - teacher_ema_coeff) * p,
                    t_state.ema_params,
                    t_state.params,
                )
                return t_state.replace(ema_params=new_ema)

            def _keep_teacher_ema(operands):
                return operands

            teacher_train_state = jax.lax.cond(
                teacher_did_update > 0.0,
                _update_teacher_ema,
                _keep_teacher_ema,
                teacher_train_state,
            )
            teacher_current_lr = (
                teacher_linear_schedule(
                    update_idx
                    * teacher_num_minibatches
                    * teacher_update_epochs
                )
                if config["ANNEAL_LR"]
                else config["TEACHER_LR"]
            )

            task_reward_mean = traj_batch.task_reward.mean()
            task_reward_std = traj_batch.task_reward.std()
            goal_reward_mean = traj_batch.goal_reward.mean()
            goal_reward_std = traj_batch.goal_reward.std()
            batch_count = jnp.asarray(
                traj_batch.task_reward.size, dtype=traj_batch.task_reward.dtype
            )
            task_reward_sum = task_reward_sum + traj_batch.task_reward.sum()
            task_reward_sq_sum = task_reward_sq_sum + jnp.square(
                traj_batch.task_reward
            ).sum()
            goal_reward_sum = goal_reward_sum + traj_batch.goal_reward.sum()
            goal_reward_sq_sum = goal_reward_sq_sum + jnp.square(
                traj_batch.goal_reward
            ).sum()
            reward_count = reward_count + batch_count
            safe_count = jnp.maximum(reward_count, 1.0)
            task_reward_running_mean = task_reward_sum / safe_count
            task_reward_running_var = jnp.maximum(
                task_reward_sq_sum / safe_count - jnp.square(task_reward_running_mean),
                0.0,
            )
            task_reward_running_std = jnp.sqrt(task_reward_running_var)
            goal_reward_running_mean = goal_reward_sum / safe_count
            goal_reward_running_var = jnp.maximum(
                goal_reward_sq_sum / safe_count - jnp.square(goal_reward_running_mean),
                0.0,
            )
            goal_reward_running_std = jnp.sqrt(goal_reward_running_var)
            current_lr = (
                linear_schedule(
                    update_idx
                    * config["NUM_MINIBATCHES"]
                    * config["UPDATE_EPOCHS"]
                )
                if config["ANNEAL_LR"]
                else config["LR"]
            )
            if config["NORMALIZE_ENV"]:
                obs_norm_mean = env_state.mean.mean()
                obs_norm_var = env_state.var.mean()
            else:
                obs_norm_mean = jnp.array(0.0, dtype=task_reward_mean.dtype)
                obs_norm_var = jnp.array(0.0, dtype=task_reward_mean.dtype)
            competence_mean = competence_vector.mean()
            competence_log_sum = jnp.log(jnp.sum(competence_vector) + 1e-8)
            done_mask = traj_batch.done
            teacher_reward_at_done = jnp.where(
                done_mask, traj_batch.teacher_reward, jnp.nan
            )
            teacher_average_competence_reward = jnp.nanmean(teacher_reward_at_done)
            teacher_average_competence_reward_all_steps = (
                traj_batch.teacher_reward.mean()
            )
            teacher_success_reward_at_done = jnp.where(
                done_mask, traj_batch.teacher_success_reward, jnp.nan
            )
            teacher_average_success_reward = jnp.nanmean(teacher_success_reward_at_done)
            teacher_learning_progress_at_done = jnp.where(
                done_mask, traj_batch.teacher_learning_progress_reward, jnp.nan
            )
            teacher_average_learning_progress = jnp.nanmean(
                teacher_learning_progress_at_done
            )
            teacher_empowerment_at_done = jnp.where(
                done_mask, traj_batch.teacher_empowerment_reward, jnp.nan
            )
            teacher_average_empowerment_reward = jnp.nanmean(teacher_empowerment_at_done)
            teacher_average_empowerment_reward_all_steps = (
                traj_batch.teacher_empowerment_reward.mean()
            )
            ppo_updates_at_done = jnp.where(
                done_mask, traj_batch.ppo_updates_per_episode, jnp.nan
            )
            average_ppo_updates_per_episode = jnp.nanmean(ppo_updates_at_done)
            teacher_buffer_count = teacher_rollout_buffer.count
            teacher_buffer_mean_reward_val = teacher_buffer_mean_reward(
                teacher_rollout_buffer
            )
            empowerment_episode_done_f = empowerment_should_update.astype(jnp.float32)

            # jax.lax.cond(
            #     jnp.any(done_mask),
            #     lambda: jax.debug.print(
            #         "learning_progress_mean={lp}, ppo_updates_per_episode_mean={u}",
            #         lp=teacher_average_learning_progress,
            #         u=average_ppo_updates_per_episode,
            #     ),
            #     lambda: None,
            # )

            if config.get("DEBUG"):
                # jax.lax.cond(
                #     cl_buffer_window_full,
                #     lambda: jax.debug.print(
                #         "cl_buffer_window_full, future_state_mean={m}",
                #         m=cl_buffer.future_state.mean(),
                #     ),
                #     lambda: None,
                # )

                def debug_callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(debug_callback, metric)

            if config.get("WANDB_MODE", "disabled") == "online":

                def wandb_callback(args):
                    (
                        info,
                        total_loss,
                        value_loss,
                        actor_loss,
                        entropy,
                        student_gradient_norm,
                        task_reward_mean,
                        task_reward_std,
                        goal_reward_mean,
                        goal_reward_std,
                        task_reward_running_mean,
                        task_reward_running_std,
                        goal_reward_running_mean,
                        goal_reward_running_std,
                        current_lr,
                        obs_norm_mean,
                        obs_norm_var,
                        competence_mean,
                        competence_log_sum,
                        teacher_average_competence_reward,
                        teacher_average_competence_reward_all_steps,
                        teacher_average_success_reward,
                        teacher_average_learning_progress,
                        teacher_average_empowerment_reward,
                        teacher_average_empowerment_reward_all_steps,
                        average_ppo_updates_per_episode,
                        teacher_buffer_count,
                        teacher_buffer_mean_reward_val,
                        teacher_total_loss,
                        teacher_value_loss,
                        teacher_actor_loss,
                        teacher_entropy,
                        teacher_did_update,
                        teacher_current_lr,
                        teacher_update_batch_size,
                        teacher_gradient_norm,
                        empowerment_episode_done_f,
                        empowerment_total_loss,
                        empowerment_loss_enc1_enc3,
                        empowerment_loss_enc2_enc3,
                        empowerment_gradient_norm,
                        empowerment_did_update,
                        empowerment_current_lr,
                    ) = args
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]

                    now = time.perf_counter()
                    log_metrics = {}
                    if len(return_values) > 0:
                        log_metrics["episodic_return"] = float(return_values.mean())
                    if _wandb_timer["last_time"] is not None:
                        elapsed = now - _wandb_timer["last_time"]
                        if elapsed > 0:
                            log_metrics["sps"] = float(steps_per_update / elapsed)
                    _wandb_timer["last_time"] = now

                    step = int(info["timestep"].max() * config["NUM_ENVS"])
                    log_metrics.update(
                        {
                            "total_loss": float(total_loss),
                            "value_loss": float(value_loss),
                            "actor_loss": float(actor_loss),
                            "entropy": float(entropy),
                            "student/gradient_norm": float(student_gradient_norm),
                            "task_reward_mean": float(task_reward_mean),
                            "task_reward_std": float(task_reward_std),
                            "goal_reward_mean": float(goal_reward_mean),
                            "goal_reward_std": float(goal_reward_std),
                            "task_reward_running_mean": float(task_reward_running_mean),
                            "task_reward_running_std": float(task_reward_running_std),
                            "goal_reward_running_mean": float(goal_reward_running_mean),
                            "goal_reward_running_std": float(goal_reward_running_std),
                            "learning_rate": float(current_lr),
                            "student_competence_mean": float(competence_mean),
                            "student_competence_log_sum": float(competence_log_sum),
                            "teacher/average_competence_reward_all_steps": float(
                                teacher_average_competence_reward_all_steps
                            ),
                            "teacher/buffer_count": float(teacher_buffer_count),
                            "teacher/buffer_mean_reward": float(
                                teacher_buffer_mean_reward_val
                            ),
                            "teacher/did_update": float(teacher_did_update),
                            "empowerment/episode_done_trigger": float(
                                empowerment_episode_done_f
                            ),
                            "empowerment/did_update": float(empowerment_did_update),
                        }
                    )
                    if _best_competence_log_sum["step"] >= 0:
                        log_metrics["student_competence_log_sum_max"] = float(
                            _best_competence_log_sum["value"]
                        )
                    if float(empowerment_did_update) > 0:
                        log_metrics.update(
                            {
                                "empowerment/learning_rate": float(
                                    empowerment_current_lr
                                ),
                                "empowerment/total_loss": float(
                                    empowerment_total_loss
                                ),
                                "empowerment/loss_enc1_enc3": float(
                                    empowerment_loss_enc1_enc3
                                ),
                                "empowerment/loss_enc2_enc3": float(
                                    empowerment_loss_enc2_enc3
                                ),
                            }
                        )
                    if config.get("NORMALIZE_ENV", False):
                        log_metrics["obs_norm_mean"] = float(obs_norm_mean)
                        log_metrics["obs_norm_var"] = float(obs_norm_var)
                    avg_competence_at_done = float(teacher_average_competence_reward)
                    if math.isfinite(avg_competence_at_done):
                        log_metrics["teacher/average_competence_reward"] = (
                            avg_competence_at_done
                        )
                    avg_success_at_done = float(teacher_average_success_reward)
                    if math.isfinite(avg_success_at_done):
                        log_metrics["teacher/average_success_reward"] = (
                            avg_success_at_done
                        )
                    avg_learning_progress = float(teacher_average_learning_progress)
                    if math.isfinite(avg_learning_progress):
                        log_metrics["teacher/average_learning_progress_reward"] = (
                            avg_learning_progress
                        )
                    avg_empowerment_at_done = float(teacher_average_empowerment_reward)
                    if math.isfinite(avg_empowerment_at_done):
                        log_metrics["teacher/empowerment_reward_at_done"] = (
                            avg_empowerment_at_done
                        )
                    log_metrics["teacher/empowerment_reward_all_steps"] = float(
                        teacher_average_empowerment_reward_all_steps
                    )
                    avg_ppo_updates = float(average_ppo_updates_per_episode)
                    if math.isfinite(avg_ppo_updates):
                        log_metrics["teacher/average_ppo_updates_per_episode"] = (
                            avg_ppo_updates
                        )
                    if float(teacher_did_update) > 0:
                        log_metrics.update(
                            {
                                "teacher/total_loss": float(teacher_total_loss),
                                "teacher/value_loss": float(teacher_value_loss),
                                "teacher/actor_loss": float(teacher_actor_loss),
                                "teacher/entropy": float(teacher_entropy),
                                "teacher/learning_rate": float(teacher_current_lr),
                                "teacher/batch_size": float(
                                    teacher_update_batch_size
                                ),
                                "teacher/gradient_norm": float(teacher_gradient_norm),
                            }
                        )
                    if float(empowerment_did_update) > 0:
                        log_metrics["empowerment/gradient_norm"] = float(
                            empowerment_gradient_norm
                        )
                    wandb.log(log_metrics, step=step)

                jax.debug.callback(
                    wandb_callback,
                    (
                        metric,
                        total_loss,
                        value_loss,
                        actor_loss,
                        entropy,
                        student_gradient_norm,
                        task_reward_mean,
                        task_reward_std,
                        goal_reward_mean,
                        goal_reward_std,
                        task_reward_running_mean,
                        task_reward_running_std,
                        goal_reward_running_mean,
                        goal_reward_running_std,
                        current_lr,
                        obs_norm_mean,
                        obs_norm_var,
                        competence_mean,
                        competence_log_sum,
                        teacher_average_competence_reward,
                        teacher_average_competence_reward_all_steps,
                        teacher_average_success_reward,
                        teacher_average_learning_progress,
                        teacher_average_empowerment_reward,
                        teacher_average_empowerment_reward_all_steps,
                        average_ppo_updates_per_episode,
                        teacher_buffer_count,
                        teacher_buffer_mean_reward_val,
                        teacher_total_loss,
                        teacher_value_loss,
                        teacher_actor_loss,
                        teacher_entropy,
                        teacher_did_update,
                        teacher_current_lr,
                        teacher_update_batch_size,
                        teacher_gradient_norm,
                        empowerment_episode_done_f,
                        empowerment_total_loss,
                        empowerment_loss_enc1_enc3,
                        empowerment_loss_enc2_enc3,
                        empowerment_gradient_norm,
                        empowerment_did_update,
                        empowerment_current_lr,
                    ),
                )

            if config.get("SAVE_MODEL", False):

                def _maybe_save_max_log_sum_teacher(args):
                    (
                        score,
                        competence_mean_val,
                        global_step,
                        teacher_step,
                        teacher_params,
                        teacher_opt_state,
                        student_step,
                        student_params,
                        student_opt_state,
                        *obs_norm_args,
                    ) = args
                    score = float(score)
                    if not math.isfinite(score):
                        return
                    if score <= _best_competence_log_sum["value"]:
                        return
                    _best_competence_log_sum["value"] = score
                    _best_competence_log_sum["step"] = int(global_step)
                    checkpoint_root = os.path.join(
                        config["EXP_DIR"],
                        config.get("checkpoint_dir", "checkpoints"),
                    )
                    teacher_ckpt_dir = os.path.join(
                        checkpoint_root, "teacher_max_log_sum_competence"
                    )
                    student_ckpt_dir = os.path.join(
                        checkpoint_root, "student_max_log_sum_competence"
                    )
                    os.makedirs(teacher_ckpt_dir, exist_ok=True)
                    os.makedirs(student_ckpt_dir, exist_ok=True)
                    teacher_ckpt_state = TrainState(
                        step=int(teacher_step),
                        apply_fn=_teacher_ckpt_host["apply_fn"],
                        params=teacher_params,
                        tx=_teacher_ckpt_host["tx"],
                        opt_state=teacher_opt_state,
                    )
                    student_ckpt_state = TrainState(
                        step=int(student_step),
                        apply_fn=_student_ckpt_host["apply_fn"],
                        params=student_params,
                        tx=_student_ckpt_host["tx"],
                        opt_state=student_opt_state,
                    )
                    save_checkpoint(teacher_ckpt_state, teacher_ckpt_dir)
                    save_checkpoint(student_ckpt_state, student_ckpt_dir)
                    if config.get("NORMALIZE_ENV", False) and len(obs_norm_args) == 3:
                        obs_mean, obs_var, obs_count = obs_norm_args
                        obs_mean = np.asarray(obs_mean)
                        obs_var = np.asarray(obs_var)
                        if obs_mean.ndim > 1:
                            obs_mean = obs_mean.reshape((-1, obs_mean.shape[-1]))[0]
                            obs_var = obs_var.reshape((-1, obs_var.shape[-1]))[0]
                        stats_path = os.path.join(
                            checkpoint_root, "obs_norm_stats.npz"
                        )
                        np.savez(
                            stats_path,
                            mean=obs_mean,
                            var=obs_var,
                            count=np.asarray(obs_count),
                        )
                        print(
                            f"[checkpoint] saved obs norm stats to {stats_path}"
                        )
                    meta_path = os.path.join(
                        checkpoint_root, "teacher_max_log_sum_competence_meta.json"
                    )
                    with open(meta_path, "w") as f:
                        json.dump(
                            {
                                "value": score,
                                "step": int(global_step),
                                "competence_mean": float(competence_mean_val),
                                "teacher_checkpoint": teacher_ckpt_dir,
                                "student_checkpoint": student_ckpt_dir,
                            },
                            f,
                            indent=2,
                        )
                    print(
                        f"[checkpoint] new max log-sum competence={score:.6f} "
                        f"at step={int(global_step)}; saved teacher to "
                        f"{teacher_ckpt_dir} and student to {student_ckpt_dir}"
                    )
                    if config.get("WANDB_MODE", "disabled") == "online":
                        wandb.log(
                            {
                                "student_competence_log_sum_max": score,
                            },
                            step=int(global_step),
                        )

                save_step = (
                    (update_idx + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                max_log_sum_ckpt_args = (
                    competence_log_sum,
                    competence_mean,
                    save_step,
                    teacher_train_state.step,
                    teacher_train_state.params,
                    teacher_train_state.opt_state,
                    train_state.step,
                    train_state.params,
                    train_state.opt_state,
                )
                if config.get("NORMALIZE_ENV", False):
                    max_log_sum_ckpt_args = max_log_sum_ckpt_args + (
                        env_state.mean,
                        env_state.var,
                        env_state.count,
                    )
                jax.debug.callback(
                    _maybe_save_max_log_sum_teacher,
                    max_log_sum_ckpt_args,
                )

            if (
                teacher_softmax_viz_num_snapshots > 0
                and config.get("WANDB_MODE", "disabled") == "online"
            ):
                should_log_viz = jnp.isin(update_idx, snapshot_indices_jnp)

                def _run_teacher_viz(_):
                    ref_env_index = int(
                        config.get("TEACHER_SOFTMAX_VIZ_REF_ENV_INDEX", 0)
                    )
                    ref_base_obs = episode_initial_base_obs[ref_env_index]
                    _, action_rng = jax.random.split(rng)
                    bootstrap_action = _bootstrap_teacher_action(
                        train_state.params, ref_base_obs, action_rng
                    )
                    teacher_obs = _build_teacher_input(
                        ref_base_obs, competence_vector, bootstrap_action
                    )
                    pi, _ = teacher_network.apply(
                        teacher_train_state.params, teacher_obs
                    )
                    probs_snapshot = pi.probs.reshape(-1)

                    def _softmax_viz_callback(args):
                        step, probs, ref_obs, env_state, competence, emp_cache = args
                        log_teacher_softmax_viz(
                            probs,
                            ref_obs,
                            env_state,
                            int(step),
                            competence,
                            emp_cache,
                        )

                    step = (
                        (update_idx + 1)
                        * config["NUM_STEPS"]
                        * config["NUM_ENVS"]
                    )
                    jax.debug.callback(
                        _softmax_viz_callback,
                        (
                            step,
                            probs_snapshot,
                            ref_base_obs,
                            env_state,
                            competence_vector,
                            goal_empowerment_reward_cache,
                        ),
                    )
                    return jnp.array(0, dtype=jnp.int32)

                def _skip_teacher_viz(_):
                    return jnp.array(0, dtype=jnp.int32)

                jax.lax.cond(
                    should_log_viz,
                    _run_teacher_viz,
                    _skip_teacher_viz,
                    operand=None,
                )

            if (
                config.get("LOG_TEACHER_INPUT_GRADS", True)
                and config.get("WANDB_MODE", "disabled") == "online"
            ):
                input_grads_freq = int(
                    config.get("LOG_TEACHER_INPUT_GRADS_FREQ", 0)
                )
                if input_grads_freq > 0:
                    should_log_input_grads = (update_idx) % input_grads_freq == 0

                    def _run_teacher_input_grads(_):
                        ref_env_index = int(
                            config.get("TEACHER_SOFTMAX_VIZ_REF_ENV_INDEX", 0)
                        )
                        ref_base_obs = episode_initial_base_obs[ref_env_index]
                        _, action_rng = jax.random.split(rng)
                        bootstrap_action = _bootstrap_teacher_action(
                            train_state.params, ref_base_obs, action_rng
                        )
                        teacher_obs = _build_teacher_input(
                            ref_base_obs, competence_vector, bootstrap_action
                        )
                        grad_stats = _teacher_input_grad_norms(
                            teacher_train_state.params, teacher_obs
                        )
                        step = (
                            (update_idx + 1)
                            * config["NUM_STEPS"]
                            * config["NUM_ENVS"]
                        )

                        def _input_grad_callback(args):
                            step_val, stats = args
                            _log_teacher_input_grads_to_wandb(step_val, stats)

                        jax.debug.callback(
                            _input_grad_callback,
                            (step, grad_stats),
                        )
                        return jnp.array(0, dtype=jnp.int32)

                    def _skip_teacher_input_grads(_):
                        return jnp.array(0, dtype=jnp.int32)

                    jax.lax.cond(
                        should_log_input_grads,
                        _run_teacher_input_grads,
                        _skip_teacher_input_grads,
                        operand=None,
                    )

            if config.get("TEACHER_GOAL_COUNT_VIZ_LOG_WANDB", True):
                freq = int(config.get("TEACHER_GOAL_COUNT_VIZ_FREQ", 0))
                if freq > 0:
                    should_log_goal_count_viz = (update_idx) % freq == 0

                    def _run_goal_count_viz(_):
                        step = (update_idx) * config["NUM_STEPS"] * config["NUM_ENVS"]
                        jax.debug.callback(
                            log_teacher_goal_selection_count_grid,
                            teacher_goal_count_grid,
                            step,
                        )
                        return jnp.array(0, dtype=jnp.int32)

                    def _skip_goal_count_viz(_):
                        return jnp.array(0, dtype=jnp.int32)

                    jax.lax.cond(
                        should_log_goal_count_viz,
                        _run_goal_count_viz,
                        _skip_goal_count_viz,
                        operand=None,
                    )

            if config.get("TEACHER_COMPETENCE_VIZ_LOG_WANDB", True):
                competence_viz_freq = int(
                    config.get("TEACHER_COMPETENCE_VIZ_FREQ", 0)
                )
                if competence_viz_freq > 0:
                    should_log_competence_viz = (
                        (update_idx) % competence_viz_freq == 0
                    )

                    def _run_competence_viz(_):
                        step = (update_idx) * config["NUM_STEPS"] * config["NUM_ENVS"]
                        jax.debug.callback(
                            log_competence_vector_bar_viz,
                            competence_vector,
                            step,
                        )
                        return jnp.array(0, dtype=jnp.int32)

                    def _skip_competence_viz(_):
                        return jnp.array(0, dtype=jnp.int32)

                    jax.lax.cond(
                        should_log_competence_viz,
                        _run_competence_viz,
                        _skip_competence_viz,
                        operand=None,
                    )

            # Periodic student eval on environment goals (no teacher sampling).
            eval_freq = int(config.get("EVAL_FREQ", 0))
            if eval_freq > 0:
                should_eval = (update_idx) % eval_freq == 0
                rng, eval_rng = jax.random.split(rng)

                def _run_student_eval(_):
                    # jax.debug.print("running student eval")
                    success_rate, episodic_return = evaluate_student_on_env_goal(
                        train_state.params, env_state, eval_rng
                    )
                    step = (update_idx) * config["NUM_STEPS"] * config["NUM_ENVS"]

                    def _log_eval(args):
                        sr, er, st = args
                        if config.get("WANDB_MODE", "disabled") == "online":
                            wandb.log(
                                {
                                    "eval/success_rate": float(sr),
                                    "eval/episodic_return": float(er),
                                },
                                # step=int(st),
                            )
                        if config.get("DEBUG"):
                            print(
                                f"eval step={int(st)}, "
                                f"success_rate={float(sr):.4f}, "
                                f"episodic_return={float(er):.4f}"
                            )
                        return None

                    jax.debug.callback(
                        _log_eval, (success_rate, episodic_return, step)
                    )
                    return jnp.array(0, dtype=jnp.int32)

                def _skip_student_eval(_):
                    return jnp.array(0, dtype=jnp.int32)

                jax.lax.cond(
                    should_eval,
                    _run_student_eval,
                    _skip_student_eval,
                    operand=None,
                )

            # Agent positions saving (in-jit trigger calling host callback)
            freq_xy = int(config.get("AGENT_POSITIONS_LOG_FREQ", 0))
            if freq_xy > 0:
                should_save_xy = (update_idx) % freq_xy == 0

                def _run_save(_):
                    def _save_agent_xy_host(args):
                        agent_xy, uidx = args
                        try:
                            import numpy as _np

                            agent_xy_np = jax.device_get(agent_xy)
                            # default: save only reference env to limit size
                            ref = int(config.get("AGENT_POSITIONS_REF_ENV_INDEX", 0))
                            only_ref = bool(config.get("AGENT_POSITIONS_ONLY_REF_ENV", True))
                            if only_ref:
                                # agent_xy shape: (NUM_STEPS, NUM_ENVS, 2)
                                agent_xy_np = agent_xy_np[:, ref, :]
                            flat = agent_xy_np.reshape(-1, 2)
                            maxp = int(config.get("AGENT_POSITIONS_MAX_POINTS", 10000))
                            if flat.shape[0] > maxp:
                                idx = _np.linspace(0, flat.shape[0] - 1, maxp).astype(_np.int32)
                                flat = flat[idx]
                            exp_dir = config["EXP_DIR"]
                            save_dir = os.path.join(exp_dir, "agent_positions")
                            os.makedirs(save_dir, exist_ok=True)
                            fname = os.path.join(save_dir, f"{int(uidx)}_agent_positions.npz")
                            _np.savez(fname, agent_xy=flat, update_idx=int(uidx))
                        except Exception as err:
                            print(f"[save_agent_xy_host] skipped saving agent xy: {err}")
                            traceback.print_exc()
                        return 0

                    jax.debug.callback(_save_agent_xy_host, (traj_batch.agent_xy, update_idx))
                    return jnp.array(0, dtype=jnp.int32)

                def _skip_save(_):
                    return jnp.array(0, dtype=jnp.int32)

                jax.lax.cond(should_save_xy, _run_save, _skip_save, operand=None)

            if enable_train_render:
                should_render = (update_idx) % train_render_freq == 0
                should_log_train_render = jnp.logical_and(
                    should_render, train_render_buf.has_completed
                )
                # jax.debug.print("update_idx is {x}", x=update_idx)
                # jax.debug.print("should_render is {x}", x=should_render)
                # jax.debug.print("train_render_buf.has_completed is {x}", x=train_render_buf.has_completed)
                # jax.debug.print("should_log_train_render is {x}", x=should_log_train_render)

                def _run_train_render(_):
                    step = (update_idx) * config["NUM_STEPS"] * config["NUM_ENVS"]

                    def _log_train_render_host(args):
                        frames, length, st = args
                        try:
                            length = int(jax.device_get(length))
                            if length <= 0:
                                return 0
                            frames_np = jax.device_get(frames)
                            sliced = jax.tree_util.tree_map(
                                lambda x: jnp.asarray(x[:length]), frames_np
                            )
                            log_pipeline_html_to_wandb(
                                sliced, st, log_key="train/render"
                            )
                        except Exception as err:
                            print(
                                f"[log_train_render_host] skipped train render: {err}"
                            )
                            traceback.print_exc()
                        return 0

                    jax.debug.callback(
                        _log_train_render_host,
                        (
                            train_render_buf.completed_frames,
                            train_render_buf.completed_length,
                            step,
                        ),
                    )
                    return train_render_buf._replace(
                        has_completed=jnp.array(False)
                    )

                def _skip_train_render(_):
                    return train_render_buf

                train_render_buf = jax.lax.cond(
                    should_log_train_render,
                    _run_train_render,
                    _skip_train_render,
                    operand=None,
                )

            runner_state = (
                train_state,
                teacher_train_state,
                empowerment_train_state,
                env_state,
                last_obs,
                goals,
                raw_goals,
                competence_vector,
                task_reward_sum,
                task_reward_sq_sum,
                goal_reward_sum,
                goal_reward_sq_sum,
                reward_count,
                episode_success,
                episode_goal_success_start,
                episode_step_count,
                episode_initial_base_obs,
                goal_learning_progress_cache,
                goal_empowerment_reward_cache,
                teacher_episode_carry,
                teacher_rollout_buffer,
                agent_episode_carry,
                cl_buffer,
                episode_buf_ptr,
                teacher_goal_count_grid,
            )
            if enable_train_render:
                runner_state = runner_state + (train_render_buf, rng)
            else:
                runner_state = runner_state + (rng,)
            if should_save_agent_trajectory_xy:
                return runner_state, (metric, agent_xy_chunk)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        reward_dtype = obsv.dtype
        runner_state = (
            train_state,
            teacher_train_state,
            empowerment_train_state,
            env_state,
            obsv,
            goals,
            raw_goals,
            competence_vector,
            jnp.array(0.0, dtype=reward_dtype),
            jnp.array(0.0, dtype=reward_dtype),
            jnp.array(0.0, dtype=reward_dtype),
            jnp.array(0.0, dtype=reward_dtype),
            jnp.array(0.0, dtype=reward_dtype),
            jnp.zeros((config["NUM_ENVS"],), dtype=reward_dtype),
            episode_goal_success_start,
            episode_step_count,
            episode_initial_base_obs,
            goal_learning_progress_cache,
            goal_empowerment_reward_cache,
            teacher_episode_carry,
            teacher_rollout_buffer,
            agent_episode_carry,
            cl_buffer,
            episode_buf_ptr,
            teacher_goal_count_grid,
        )
        if enable_train_render:
            runner_state = runner_state + (train_render_buf, _rng)
        else:
            runner_state = runner_state + (_rng,)
        if should_save_agent_trajectory_xy:
            runner_state, (metric, agent_trajectory_xy) = jax.lax.scan(
                _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
            )
        else:
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
            )
        final_teacher_rollout_buffer = runner_state[20]
        final_cl_buffer = runner_state[22]
        final_episode_buf_ptr = runner_state[23]
        final_empowerment_train_state = runner_state[2]
        train_output = {
            "runner_state": runner_state,
            "metrics": metric,
            "teacher_params": runner_state[1].params,
            "teacher_train_state": runner_state[1],
            "empowerment_train_state": final_empowerment_train_state,
            "teacher_rollout_buffer": final_teacher_rollout_buffer,
            "cl_buffer": final_cl_buffer,
            "episode_buf_ptr": final_episode_buf_ptr,
        }
        if should_save_agent_trajectory_xy:
            train_output["agent_trajectory_xy"] = agent_trajectory_xy
        return train_output

    return train, render_eval_episode, log_teacher_softmax_snapshot


def main():
    config_obj = parse_config_from_cli()
    config = asdict(config_obj)
    gpu_names = sorted({d.device_kind for d in jax.devices("gpu")})
    config["GPU_NAME"] = gpu_names[0]

    scratch = os.environ.get("SCRATCH")
    random_name = RandomWord().word()
    random_id = np.random.randint(1000000000)
    while os.path.exists(
        os.path.join(scratch, "purejaxrl_emp_teachers", f"{random_name}_{random_id}")
    ):
        random_name = RandomWord().word()
        random_id = np.random.randint(1000000000)
    experiment_name = f"{random_name}_{random_id}"
    config["EXP_DIR"] = os.path.join(
        scratch, "purejaxrl_emp_teachers", experiment_name
    )
    config["AGENT_POSITIONS_SAVE_DIR"] = os.path.join(
        config["EXP_DIR"], "agent_positions"
    )

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["PPO", "BRAX", config["ENV_NAME"], f"jax_{jax.__version__}"],
        name=experiment_name,
        config=config,
        mode=config["WANDB_MODE"],
    )

    print(f"Experiment directory: {config['EXP_DIR']}")
    os.makedirs(config["EXP_DIR"], exist_ok=True)

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn, render_eval_episode, log_teacher_softmax_snapshot = make_train(config)
    config_path = os.path.join(config["EXP_DIR"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"[checkpoint] saved config to {config_path}")
    train_jit = jax.jit(train_fn)
    train_output = train_jit(rng)
    jax.block_until_ready(train_output["runner_state"][-1])
    rs = train_output["runner_state"]
    final_train_state = rs[0]
    final_teacher_train_state = rs[1]
    final_empowerment_train_state = rs[2]
    final_env_state = rs[3]

    exp_dir = config["EXP_DIR"]
    exp_name = f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
    if config.get("SAVE_AGENT_TRAJECTORY_XY", False):
        agent_positions_dir = os.path.join(exp_dir, "agent_positions")
        os.makedirs(agent_positions_dir, exist_ok=True)
        agent_xy = np.asarray(jax.device_get(train_output["agent_trajectory_xy"]))
        trajectory_path, flat_xy, global_step = save_agent_trajectory_xy(
            agent_xy, config, agent_positions_dir, exp_name
        )
        num_points = int(flat_xy.shape[0])
        print(
            f"[agent_trajectory] saved {trajectory_path} with {num_points} points"
        )
        trajectory_plot_path = os.path.join(
            agent_positions_dir, f"{exp_name}_agent_trajectory.png"
        )
        plot_agent_trajectory_xy(
            flat_xy,
            global_step,
            trajectory_plot_path,
            title=f'Agent trajectory ({config["ENV_NAME"]})',
        )
        print(f"[agent_trajectory] saved plot {trajectory_plot_path}")
        if config.get("WANDB_MODE", "disabled") == "online":
            wandb.save(trajectory_path, base_path=exp_dir, policy="now")
            wandb.save(trajectory_plot_path, base_path=exp_dir, policy="now")
            wandb.log(
                {
                    "agent/trajectory_plot": wandb.Image(trajectory_plot_path),
                },
                step=int(config["TOTAL_TIMESTEPS"]),
            )
    if config["SAVE_MODEL"]:
        checkpoint_root = os.path.join(exp_dir, config.get("checkpoint_dir", "checkpoints"))
        student_ckpt_dir = os.path.join(checkpoint_root, "student")
        teacher_ckpt_dir = os.path.join(checkpoint_root, "teacher")
        teacher_ema_ckpt_dir = os.path.join(checkpoint_root, "teacher_ema")
        empowerment_ckpt_dir = os.path.join(checkpoint_root, "empowerment")
        os.makedirs(student_ckpt_dir, exist_ok=True)
        os.makedirs(teacher_ckpt_dir, exist_ok=True)
        os.makedirs(teacher_ema_ckpt_dir, exist_ok=True)
        os.makedirs(empowerment_ckpt_dir, exist_ok=True)
        save_checkpoint(final_train_state, student_ckpt_dir)
        # Save plain TrainState (no ema_params) for backward-compatible loading.
        teacher_ckpt_state = TrainState(
            step=final_teacher_train_state.step,
            apply_fn=final_teacher_train_state.apply_fn,
            params=final_teacher_train_state.params,
            tx=final_teacher_train_state.tx,
            opt_state=final_teacher_train_state.opt_state,
        )
        teacher_ema_ckpt_state = TrainState(
            step=final_teacher_train_state.step,
            apply_fn=final_teacher_train_state.apply_fn,
            params=final_teacher_train_state.ema_params,
            tx=final_teacher_train_state.tx,
            opt_state=final_teacher_train_state.opt_state,
        )
        save_checkpoint(teacher_ckpt_state, teacher_ckpt_dir)
        save_checkpoint(teacher_ema_ckpt_state, teacher_ema_ckpt_dir)
        save_checkpoint(final_empowerment_train_state, empowerment_ckpt_dir)
        print(f"[checkpoint] saved student model to {student_ckpt_dir}")
        print(f"[checkpoint] saved teacher model to {teacher_ckpt_dir}")
        print(f"[checkpoint] saved teacher EMA model to {teacher_ema_ckpt_dir}")
        print(f"[checkpoint] saved empowerment model to {empowerment_ckpt_dir}")
        if config.get("NORMALIZE_ENV", False):
            # Observation dim is the trailing dim of the running mean in env_state.
            expected_obs_dim = None
            current = final_env_state
            while current is not None:
                if hasattr(current, "mean") and jnp.ndim(current.mean) > 0:
                    expected_obs_dim = int(current.mean.shape[-1])
                    break
                current = getattr(current, "env_state", None)
            if expected_obs_dim is not None:
                obs_mean, obs_var, obs_count = extract_obs_norm_stats(
                    final_env_state, expected_obs_dim
                )
                if obs_mean is not None and obs_var is not None:
                    stats_path = os.path.join(checkpoint_root, "obs_norm_stats.npz")
                    save_kwargs = {
                        "mean": np.asarray(jax.device_get(obs_mean)),
                        "var": np.asarray(jax.device_get(obs_var)),
                    }
                    if obs_count is not None:
                        save_kwargs["count"] = np.asarray(jax.device_get(obs_count))
                    np.savez(stats_path, **save_kwargs)
                    print(f"[checkpoint] saved obs norm stats to {stats_path}")
    # render_eval_episode(
    #     final_train_state.params,
    #     final_teacher_train_state.params,
    #     rs[7],
    #     rs[-1],
    #     final_env_state,
    # )


def _run_teacher_softmax_snapshot_index_checks() -> None:
    """Lightweight host-side checks for snapshot index scheduling."""
    for num_updates, num_snapshots in (
        (18310, 100),
        (18310, 1000),
        (100, 1000),
        (1_831_050, 10_000),
    ):
        indices = _compute_teacher_softmax_snapshot_indices(
            num_updates, num_snapshots
        )
        _verify_teacher_softmax_snapshot_indices(
            indices, num_updates, num_snapshots
        )


if __name__ == "__main__":
    _run_teacher_softmax_snapshot_index_checks()
    main()