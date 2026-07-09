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
    GAE_LAMBDA: float = 0.8
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.0
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
    DEBUG: bool = True
    SEED: int = 30
    WANDB_MODE: str = "online"
    ENTITY: str = ""
    PROJECT: str = "purejaxrl"
    EVAL_RENDER_STEPS: int = 1000
    EVAL_RENDER_MAX_FRAMES: int = 100
    EVAL_RENDER_HEIGHT: int = 360
    EVAL_RENDER_LOG_WANDB_HTML: bool = True
    COMMENT: str = ""
    ADD_GOAL_REWARD: bool = True
    CONDITION_ON_GOAL: bool = True
    GOAL_REACH_EPSILON: float = 0.5
    TEACHER_GOAL_X_MIN: float = 4.0
    TEACHER_GOAL_X_MAX: float = 12.0
    TEACHER_GOAL_Y_MIN: float = 4.0
    TEACHER_GOAL_Y_MAX: float = 12.0
    TEACHER_NUM_GOAL_POINTS: int = 30
    TEACHER_HIDDEN_DIM: int = 256
    SAVE_MODEL: bool = False
    checkpoint_dir: str = "checkpoints"
    GOAL_REWARD_COEF: float = 1.0
    INTERPOLATED_REWARD: bool = False
    NUM_EVAL_ENVS: int = 4
    CONDITION_TEACHER_ON_COMPETENCE: bool = True
    USE_AVERAGE_COMPETENCE_REWARD: bool = False
    USE_LEARNING_PROGRESS_REWARD: bool = True
    ABSOLUTE_LEARNING_PROGRESS: bool = False
    TEACHER_SOFTMAX_VIZ_NUM_SNAPSHOTS: int = 0  # in-training softmax visuals (evenly spaced); 0 disables
    TEACHER_SOFTMAX_VIZ_LOG_WANDB: bool = True
    TEACHER_SOFTMAX_VIZ_REF_ENV_INDEX: int = 0
    TEACHER_ROLLOUT_BUFFER_SIZE: int = 4
    TEACHER_NUM_MINIBATCHES: int = 8
    TEACHER_UPDATE_EPOCHS: int = 4
    TEACHER_LR: float = 3e-4
    TEACHER_GAMMA: float = 0.99
    TEACHER_GAE_LAMBDA: float = 0.8
    TEACHER_CLIP_EPS: float = 0.2
    TEACHER_ENT_COEF: float = 0.0
    TEACHER_VF_COEF: float = 0.5
    TEACHER_MAX_GRAD_NORM: float = 1.0


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
    return (goal - mean_xy) / jnp.sqrt(var_xy + 1e-8)


def _denorm_xy(xy, mean, var):
    mean_xy = mean[..., :2]
    var_xy = var[..., :2]
    return xy * jnp.sqrt(var_xy + 1e-8) + mean_xy


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

    def _reinit_with_goal(brax_state, goal):
        q = brax_state.pipeline_state.q.at[-2:].set(goal)
        pipeline_state = brax_env.pipeline_init(q, brax_state.pipeline_state.qd)
        obs = brax_env._get_obs(pipeline_state)
        return brax_state.replace(pipeline_state=pipeline_state, obs=obs)

    reinit_with_goal = jax.vmap(_reinit_with_goal, in_axes=(0, None))

    def eval_one_goal(rng, specific_goal):
        reset_rngs = jax.random.split(rng, num_envs_per_goal)
        if eval_stats is not None:
            obsv, env_state = env.reset_with_stats(
                reset_rngs, eval_stats, env_params
            )
        else:
            obsv, env_state = env.reset(reset_rngs, env_params)

        brax_state = _inner_brax_state(env_state)
        brax_state = reinit_with_goal(brax_state, specific_goal)
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

        def step_fn(carry, _):
            obsv, env_state, rng = carry
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
            success = _success_metric(env_state)
            return (obsv, env_state, rng), success

        _, successes = jax.lax.scan(
            step_fn, (obsv, env_state, rng), None, length=max_steps
        )

        return successes.max(axis=0).mean()

    vmap_goals = jax.vmap(eval_one_goal, in_axes=(0, 0))
    goal_rngs = jax.random.split(jax.random.PRNGKey(42), goals.shape[0])
    return vmap_goals(goal_rngs, goals)


def parse_config_from_cli() -> TrainConfig:
    return tyro.cli(TrainConfig)


def _should_log_softmax_snapshot(update_idx, num_updates, num_snapshots):
    """Return True when this update is an evenly spaced in-training snapshot."""
    return (
        (num_snapshots > 0)
        & (((update_idx + 1) * num_snapshots) // num_updates)
        > ((update_idx * num_snapshots) // num_updates)
    )


def _scatter_goal_learning_progress(cache, goal_idx, lp_values, done):
    updated = jnp.where(
        done,
        lp_values,
        cache[goal_idx],
    )
    return cache.at[goal_idx].set(updated)


def plot_teacher_goal_grid_heatmap(
    goal_grid_xy,
    values,
    num_points,
    *,
    colorbar_label="Value",
    cmap="viridis",
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
    goal_grid_xy, probs, num_points, *, start_xy=None, title=None, save_path=None
):
    """Heatmap of the teacher's categorical distribution over its goal grid."""
    return plot_teacher_goal_grid_heatmap(
        goal_grid_xy,
        probs,
        num_points,
        colorbar_label="P(goal)",
        cmap="viridis",
        start_xy=start_xy,
        title=title,
        save_path=save_path,
    )


def plot_teacher_learning_progress_heatmap(
    goal_grid_xy, values, num_points, *, start_xy=None, title=None, save_path=None
):
    """Heatmap of cached learning-progress reward per teacher goal."""
    values = np.asarray(values).reshape(-1)
    abs_max = float(np.max(np.abs(values))) if values.size else 0.0
    vmin = -abs_max if abs_max > 0.0 else None
    vmax = abs_max if abs_max > 0.0 else None
    return plot_teacher_goal_grid_heatmap(
        goal_grid_xy,
        values,
        num_points,
        colorbar_label="Learning progress reward",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
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
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class TeacherActorCritic(nn.Module):
    num_actions: int
    activation: str = "tanh"
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    task_reward: jnp.ndarray
    goal_reward: jnp.ndarray
    teacher_reward: jnp.ndarray
    teacher_success_reward: jnp.ndarray
    teacher_learning_progress_reward: jnp.ndarray
    ppo_updates_per_episode: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


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


def init_teacher_episode_carry(num_envs, teacher_obs_dim, dtype):
    return TeacherEpisodeCarry(
        teacher_obs=jnp.zeros((num_envs, teacher_obs_dim), dtype=dtype),
        goal_idx=jnp.zeros((num_envs,), dtype=jnp.int32),
        raw_goal=jnp.zeros((num_envs, 2), dtype=dtype),
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


def init_teacher_rollout_buffer(buffer_size, num_envs, teacher_obs_dim, dtype):
    return TeacherRolloutBuffer(
        teacher_obs=jnp.zeros((buffer_size, num_envs, teacher_obs_dim), dtype=dtype),
        goal_idx=jnp.zeros((buffer_size, num_envs), dtype=jnp.int32),
        raw_goal=jnp.zeros((buffer_size, num_envs, 2), dtype=dtype),
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

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    teacher_batch_size = (
        config["TEACHER_ROLLOUT_BUFFER_SIZE"] * config["NUM_ENVS"]
    )
    config["TEACHER_MINIBATCH_SIZE"] = (
        teacher_batch_size // config["TEACHER_NUM_MINIBATCHES"]
    )
    assert (
        teacher_batch_size
        == config["TEACHER_MINIBATCH_SIZE"] * config["TEACHER_NUM_MINIBATCHES"]
    ), (
        "teacher batch size must equal "
        "TEACHER_MINIBATCH_SIZE * TEACHER_NUM_MINIBATCHES"
    )
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
    approx_episode_cycles = (
        config["TOTAL_TIMESTEPS"]
        // config["NUM_ENVS"]
        // config.get("EPISODE_LENGTH", 1000)
    )
    teacher_num_updates = max(
        1, approx_episode_cycles // teacher_rollout_buffer_size
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
        return config["TEACHER_LR"] * frac

    network = ActorCritic(
        env.action_space(env_params).shape[0], activation=config["ACTIVATION"], hidden_dim=config["HIDDEN_DIM"]
    )
    base_obs_dim = int(env.observation_space(env_params).shape[0])
    teacher_num_goal_points = int(config["TEACHER_NUM_GOAL_POINTS"])
    xs = jnp.linspace(
        config["TEACHER_GOAL_X_MIN"],
        config["TEACHER_GOAL_X_MAX"],
        teacher_num_goal_points,
    )
    ys = jnp.linspace(
        config["TEACHER_GOAL_Y_MIN"],
        config["TEACHER_GOAL_Y_MAX"],
        teacher_num_goal_points,
    )
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
    goal_grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
    num_teacher_goals = teacher_num_goal_points * teacher_num_goal_points
    all_goals = all_possible_goals()
    num_competence = int(all_goals.shape[0])
    condition_teacher_on_competence = config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
    use_average_competence_reward = config.get("USE_AVERAGE_COMPETENCE_REWARD", False)
    use_learning_progress_reward = config.get("USE_LEARNING_PROGRESS_REWARD", False)
    update_competence = (
        condition_teacher_on_competence or use_average_competence_reward
    )
    teacher_obs_dim = (
        base_obs_dim + num_competence if condition_teacher_on_competence else base_obs_dim
    )
    teacher_network = TeacherActorCritic(
        num_actions=teacher_num_goal_points * teacher_num_goal_points,
        activation=config["ACTIVATION"],
        hidden_dim=config["TEACHER_HIDDEN_DIM"],
    )

    def _extract_obs_norm_stats(env_state, expected_obs_dim):
        """Find observation normalization stats in nested wrapped env state."""
        current = env_state
        while hasattr(current, "env_state"):
            if hasattr(current, "mean") and hasattr(current, "var"):
                mean = current.mean
                # Only keep stats whose trailing dim matches observation dim.
                if jnp.ndim(mean) > 0 and mean.shape[-1] == expected_obs_dim:
                    if mean.ndim > 1:
                        flat_mean = mean.reshape((-1, expected_obs_dim))
                        flat_var = current.var.reshape((-1, expected_obs_dim))
                        return flat_mean[0], flat_var[0]
                    return mean, current.var
            current = current.env_state
        return None, None

    def _normalize_eval_obs(obs, obs_mean, obs_var):
        if obs_mean is None or obs_var is None:
            return obs
        if obs.ndim > 1 and obs.shape[0] == 1:
            obs = obs[0]
        return (obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)

    def _build_teacher_input(obs, competence_vector):
        if obs.ndim == 1:
            obs = obs[None, :]
        if not condition_teacher_on_competence:
            return obs
        comp_batch = jnp.broadcast_to(
            competence_vector, (obs.shape[0], competence_vector.shape[0])
        )
        return jnp.concatenate([obs, comp_batch], axis=-1)

    def _teacher_act(teacher_params, obs, competence_vector, rng):
        teacher_obs = _build_teacher_input(obs, competence_vector)
        pi, value = teacher_network.apply(teacher_params, teacher_obs)
        goal_idx = pi.sample(seed=rng)
        raw_goal = goal_grid[goal_idx].astype(jnp.float32)
        log_prob = pi.log_prob(goal_idx)
        return raw_goal, goal_idx, log_prob, value, teacher_obs

    def _sample_teacher_goals(teacher_params, obs, competence_vector, rng):
        raw_goal, _, _, _, _ = _teacher_act(
            teacher_params, obs, competence_vector, rng
        )
        return raw_goal

    def _policy_goal_from_raw(raw_goal, obs_mean, obs_var):
        if config["NORMALIZE_ENV"]:
            return _normalize_xy(raw_goal, obs_mean, obs_var)
        return raw_goal

    render_sim_steps = int(config.get("EVAL_RENDER_STEPS", 500))
    render_max_frames = int(config.get("EVAL_RENDER_MAX_FRAMES", 120))
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

    def _render_rollout_impl(
        params, teacher_params, competence_vector, rng, obs_mean, obs_var
    ):
        rng, reset_rng = jax.random.split(rng)
        state = base_env.reset(reset_rng)
        raw_goal = jnp.zeros((2,), dtype=jnp.float32)
        policy_goal = jnp.zeros((2,), dtype=jnp.float32)
        if condition_on_goal:
            rng, goal_rng = jax.random.split(rng)
            norm_obs = _normalize_eval_obs(state.obs, obs_mean, obs_var)
            raw_goal = _sample_teacher_goals(
                teacher_params, norm_obs, competence_vector, goal_rng
            )
            if raw_goal.ndim > 1:
                raw_goal = raw_goal[0]
            policy_goal = _policy_goal_from_raw(raw_goal, obs_mean, obs_var)

        def step_fn(carry, _):
            state, raw_goal, policy_goal, rng = carry
            if condition_on_goal:
                action = _eval_render_action(
                    params, state.obs, obs_mean, obs_var, policy_goal
                )
            else:
                action = _eval_render_action(params, state.obs, obs_mean, obs_var)

            def repeat_step(s, __):
                return base_env.step(s, action), None

            state, _ = jax.lax.scan(
                repeat_step, state, None, length=render_action_repeat
            )
            if condition_on_goal:
                rng, goal_rng = jax.random.split(rng)
                norm_obs = _normalize_eval_obs(state.obs, obs_mean, obs_var)
                new_raw_goal = _sample_teacher_goals(
                    teacher_params, norm_obs, competence_vector, goal_rng
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

    teacher_softmax_viz_log_wandb = bool(
        config.get("TEACHER_SOFTMAX_VIZ_LOG_WANDB", True)
    )
    teacher_softmax_viz_num_snapshots = int(
        config.get("TEACHER_SOFTMAX_VIZ_NUM_SNAPSHOTS", 0)
    )
    num_updates = int(config["NUM_UPDATES"])

    def log_teacher_softmax_viz(
        probs,
        ref_obs,
        env_state,
        step,
        competence_vector,
        goal_learning_progress_cache,
    ):
        """Plot teacher softmax, competence, and learning-progress visuals; log to wandb."""
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

            exp_dir = wandb.run.dir if wandb.run is not None else os.getcwd()
            exp_name = (
                wandb.run.name
                if wandb.run is not None and wandb.run.name
                else f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
            )
            save_path = os.path.join(
                exp_dir, f"{exp_name}_teacher_softmax_{step}.png"
            )
            fig, _ = plot_teacher_softmax(
                goal_grid_xy,
                probs,
                teacher_num_goal_points,
                start_xy=start_xy,
                title=f'Teacher softmax @ step {step} ({config["ENV_NAME"]})',
                save_path=save_path,
            )
            bar_save_path = os.path.join(
                exp_dir, f"{exp_name}_teacher_softmax_bar_{step}.png"
            )
            bar_fig, _ = plot_teacher_softmax_bar(
                probs,
                title=f'Teacher softmax bar @ step {step} ({config["ENV_NAME"]})',
                save_path=bar_save_path,
            )
            competence_save_path = os.path.join(
                exp_dir, f"{exp_name}_competence_vector_bar_{step}.png"
            )
            competence_fig, _ = plot_competence_vector_bar(
                competence,
                title=f'Competence vector @ step {step} ({config["ENV_NAME"]})',
                save_path=competence_save_path,
            )
            lp_fig = None
            if use_learning_progress_reward:
                lp_values = np.asarray(
                    jax.device_get(goal_learning_progress_cache)
                ).reshape(-1)
                lp_save_path = os.path.join(
                    exp_dir, f"{exp_name}_teacher_learning_progress_{step}.png"
                )
                # jax.debug.print("lp_values: {lp_values}", lp_values=lp_values)
                lp_fig, _ = plot_teacher_learning_progress_heatmap(
                    goal_grid_xy,
                    lp_values,
                    teacher_num_goal_points,
                    start_xy=start_xy,
                    title=(
                        f'Learning progress reward @ step {step} '
                        f'({config["ENV_NAME"]})'
                    ),
                    save_path=lp_save_path,
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
                if lp_fig is not None:
                    log_payload["teacher/learning_progress_heatmap"] = wandb.Image(
                        lp_fig
                    )
                wandb.log(log_payload, step=step)
            plt.close(fig)
            plt.close(bar_fig)
            plt.close(competence_fig)
            if lp_fig is not None:
                plt.close(lp_fig)
        except Exception as err:
            print(f"[log_teacher_softmax_viz] skipped teacher visual plots: {err}")
            traceback.print_exc()

    def log_teacher_softmax_snapshot(
        teacher_params,
        ref_obs,
        competence_vector,
        env_state,
        step,
        goal_learning_progress_cache,
    ):
        """Compute teacher probs and log teacher visual snapshots."""
        ref_obs = jnp.asarray(ref_obs).reshape(-1)
        teacher_obs = _build_teacher_input(ref_obs, competence_vector)
        pi, _ = teacher_network.apply(teacher_params, teacher_obs)
        probs = pi.probs.reshape(-1)
        log_teacher_softmax_viz(
            probs,
            ref_obs,
            env_state,
            step,
            competence_vector,
            goal_learning_progress_cache,
        )

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
            pipeline_states = _subsample_pipeline_states(
                pipeline_states, render_max_frames
            )
            rollout = _pipeline_states_to_list(jax.device_get(pipeline_states))

            rendered_html = html.render(
                base_env.sys.tree_replace({"opt.timestep": base_env.dt}),
                rollout,
                height=int(config.get("EVAL_RENDER_HEIGHT", 480)),
            )
            exp_dir = wandb.run.dir if wandb.run is not None else os.getcwd()
            exp_name = (
                wandb.run.name
                if wandb.run is not None and wandb.run.name
                else f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
            )
            num_steps = int(config["TOTAL_TIMESTEPS"])
            html_path = os.path.join(exp_dir, f"{exp_name}_{num_steps}.html")
            with open(html_path, "w", encoding="utf-8") as file:
                file.write(rendered_html)
            if config.get("EVAL_RENDER_LOG_WANDB_HTML", False):
                wandb.log({"render": wandb.Html(rendered_html)}, step=num_steps)
            else:
                wandb.save(html_path, base_path=exp_dir, policy="now")
                wandb.log({"render/html_path": html_path}, step=num_steps)

            log_multi_ant_maze = globals().get("_log_multi_ant_maze_top_gif")
            if callable(log_multi_ant_maze):
                log_multi_ant_maze(base_env, rollout, exp_dir, exp_name, num_steps)
        except Exception as err:
            print(f"[render_eval_episode] skipped video logging: {err}")
            traceback.print_exc()

    def train(rng):
        steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
        _wandb_timer = {"last_time": None}

        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        goal_dim = 2
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
        rng, goal_rng, teacher_init_rng, _rng = jax.random.split(rng, 4)
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
        teacher_train_state = TrainState.create(
            apply_fn=teacher_network.apply,
            params=teacher_init_params,
            tx=teacher_tx,
        )

        def sample_teacher_goals(obs, competence_vector, rng):
            return _sample_teacher_goals(
                teacher_train_state.params, obs, competence_vector, rng
            )

        def teacher_act_and_carry(obs, competence_vector, rng):
            raw_goal, goal_idx, teacher_log_prob, teacher_value, teacher_obs = (
                _teacher_act(teacher_train_state.params, obs, competence_vector, rng)
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
                    step_fn, (obsv, env_state, rng), None, length=5000
                )
                return env_state
            warmup_env_state = run_policy(network_params, rng)
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
        else:
            obsv, env_state = env.reset(reset_rng, env_params)

        episode_initial_base_obs = obsv[..., :base_obs_dim]
        competence_vector = jnp.zeros((num_competence,), dtype=obsv.dtype)
        raw_goals, teacher_episode_carry = teacher_act_and_carry(
            obsv[..., :base_obs_dim], competence_vector, goal_rng
        )
        teacher_rollout_buffer = init_teacher_rollout_buffer(
            teacher_rollout_buffer_size,
            config["NUM_ENVS"],
            teacher_obs_dim,
            obsv.dtype,
        )
        goals = raw_goals
        if condition_on_goal:
            if config["NORMALIZE_ENV"]:
                mean_xy = env_state.mean[..., :2]
                var_xy = env_state.var[..., :2]
                goals = (raw_goals - mean_xy) / jnp.sqrt(var_xy + 1e-8)
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

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    teacher_train_state,
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
                    teacher_episode_carry,
                    teacher_rollout_buffer,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, goal_rng,  _rng = jax.random.split(rng, 3)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

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
                if add_goal_reward:
                    if config["NORMALIZE_ENV"]:
                        dist = jnp.linalg.norm(
                            env_state.org_obs[..., :2] - raw_goals, axis=-1
                        )
                    else:
                        dist = jnp.linalg.norm(obsv[..., :2] - raw_goals, axis=-1)
                    # jax.debug.print("dist_mean: {dist}", dist=dist.mean())
                    goal_reward = (dist <= goal_reach_epsilon).astype(task_reward.dtype)
                    # jax.debug.print("dist: {dist}", dist=dist)
                    # jax.debug.print("goals: {goals}", goals=goals)
                    # jax.debug.print("goal_reward_mean: {goal_reward}", goal_reward=goal_reward.mean())
                    reward = task_reward + config["GOAL_REWARD_COEF"] * goal_reward
                    if config["INTERPOLATED_REWARD"]:
                        reward = (1-config["GOAL_REWARD_COEF"]) * task_reward + config["GOAL_REWARD_COEF"] * goal_reward
                if update_competence:
                    competence_vector = jax.lax.cond(
                        jnp.any(done),
                        lambda _: compute_competence_vector(
                            train_state.params, env_state
                        ),
                        lambda _: competence_vector,
                        operand=None,
                    )
                    # jax.debug.print("competence_vector: {competence_vector}", competence_vector=competence_vector)
                if use_average_competence_reward:
                    avg_competence = competence_vector.mean()
                    competence_part = jnp.where(done, avg_competence, 0.0)
                else:
                    competence_part = jnp.zeros_like(task_reward)
                success_part = jnp.where(done, episode_success, 0.0)

                episode_step_count = episode_step_count + 1.0
                ppo_updates_per_episode = (
                    episode_step_count + config["NUM_STEPS"] - 1
                ) // config["NUM_STEPS"]
                ppo_updates_at_done = jnp.where(done, ppo_updates_per_episode, 0.0)

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

                teacher_reward = (
                    competence_part + success_part + learning_progress_part
                )
                episode_success = jnp.where(done, 0.0, episode_success)
                teacher_rollout_buffer = push_teacher_rollout_on_done(
                    teacher_rollout_buffer,
                    teacher_episode_carry,
                    teacher_reward,
                    done,
                )
                # jax.debug.print("competence_vector: {competence_vector}", competence_vector=competence_vector)
                (
                    new_raw_goals,
                    goal_idx,
                    teacher_log_prob,
                    teacher_value,
                    teacher_obs,
                ) = _teacher_act(
                    teacher_train_state.params,
                    obsv[..., :base_obs_dim],
                    competence_vector,
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
                episode_step_count = jnp.where(done, 0.0, episode_step_count)
                # jax.debug.print("done: {done}", done=jnp.any(done))
                # jax.lax.cond(
                #     jnp.any(done),
                #     lambda:jax.debug.print("obsv xy: {obsv}", obsv=obsv[..., :2]),
                #     lambda: None,
                # )
                def normalize_goals(raw_goals, env_state):
                    if config["NORMALIZE_ENV"]:
                        mean_xy = env_state.mean[..., :2]
                        var_xy = env_state.var[..., :2]
                        return (raw_goals - mean_xy) / jnp.sqrt(var_xy + 1e-8)
                    else:
                        return raw_goals
                goals = normalize_goals(raw_goals, env_state)
                if condition_on_goal:
                    obsv = jnp.concatenate([obsv, goals], axis=-1)
                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    task_reward,
                    goal_reward,
                    teacher_reward,
                    success_part,
                    learning_progress_part,
                    ppo_updates_at_done,
                    log_prob,
                    last_obs,
                    info,
                )
                runner_state = (
                    train_state,
                    teacher_train_state,
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
                    teacher_episode_carry,
                    teacher_rollout_buffer,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                teacher_train_state,
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
                teacher_episode_carry,
                teacher_rollout_buffer,
                rng,
            ) = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
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
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

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

            teacher_should_update = (
                teacher_rollout_buffer.count >= teacher_rollout_buffer_size
            ) & (use_average_competence_reward | use_learning_progress_reward)

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
                        total_loss, grads = grad_fn(
                            t_state.params, traj_b, gae_b, tgt_b
                        )
                        t_state = t_state.apply_gradients(grads=grads)
                        return t_state, total_loss

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
                    t_loss_info[1][0].mean(),
                    t_loss_info[1][1].mean(),
                    t_loss_info[1][2].mean(),
                    jnp.array(1.0, dtype=jnp.float32),
                    jnp.array(float(teacher_batch_size), dtype=jnp.float32),
                )
                return t_train_state, new_buffer, t_rng, metrics

            def _skip_teacher_update(operands):
                t_train_state, t_buffer, t_rng = operands
                z = jnp.array(0.0, dtype=jnp.float32)
                return t_train_state, t_buffer, t_rng, (z, z, z, z, z, z)

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
            teacher_did_update = teacher_metrics[4]
            teacher_update_batch_size = teacher_metrics[5]
            teacher_current_lr = (
                teacher_linear_schedule(
                    update_idx
                    * teacher_num_minibatches
                    * teacher_update_epochs
                )
                if config["ANNEAL_LR"]
                else config["TEACHER_LR"]
            )

            value_loss = loss_info[1][0].mean()
            actor_loss = loss_info[1][1].mean()
            entropy = loss_info[1][2].mean()
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
            ppo_updates_at_done = jnp.where(
                done_mask, traj_batch.ppo_updates_per_episode, jnp.nan
            )
            average_ppo_updates_per_episode = jnp.nanmean(ppo_updates_at_done)
            teacher_buffer_count = teacher_rollout_buffer.count
            teacher_buffer_mean_reward_val = teacher_buffer_mean_reward(
                teacher_rollout_buffer
            )

            jax.lax.cond(
                jnp.any(done_mask),
                lambda: jax.debug.print(
                    "learning_progress_mean={lp}, ppo_updates_per_episode_mean={u}",
                    lp=teacher_average_learning_progress,
                    u=average_ppo_updates_per_episode,
                ),
                lambda: None,
            )

            if config.get("DEBUG"):

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
                        teacher_average_competence_reward,
                        teacher_average_competence_reward_all_steps,
                        teacher_average_success_reward,
                        teacher_average_learning_progress,
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
                    ) = args
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    if len(return_values) == 0:
                        return

                    now = time.perf_counter()
                    log_metrics = {"episodic_return": float(return_values.mean())}
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
                            "teacher/average_competence_reward_all_steps": float(
                                teacher_average_competence_reward_all_steps
                            ),
                            "teacher/buffer_count": float(teacher_buffer_count),
                            "teacher/buffer_mean_reward": float(
                                teacher_buffer_mean_reward_val
                            ),
                            "teacher/did_update": float(teacher_did_update),
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
                            }
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
                        teacher_average_competence_reward,
                        teacher_average_competence_reward_all_steps,
                        teacher_average_success_reward,
                        teacher_average_learning_progress,
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
                    ),
                )

            if (
                teacher_softmax_viz_num_snapshots > 0
                and config.get("WANDB_MODE", "disabled") == "online"
            ):
                should_log_viz = _should_log_softmax_snapshot(
                    update_idx, num_updates, teacher_softmax_viz_num_snapshots
                )
                # jax.debug.print("should_log_viz: {should_log_viz}", should_log_viz=should_log_viz)

                def _run_teacher_viz(_):
                    ref_env_index = int(
                        config.get("TEACHER_SOFTMAX_VIZ_REF_ENV_INDEX", 0)
                    )
                    ref_base_obs = episode_initial_base_obs[ref_env_index]
                    teacher_obs = _build_teacher_input(
                        ref_base_obs, competence_vector
                    )
                    pi, _ = teacher_network.apply(
                        teacher_train_state.params, teacher_obs
                    )
                    probs_snapshot = pi.probs.reshape(-1)

                    def _softmax_viz_callback(args):
                        step, probs, ref_obs, env_state, competence, lp_cache = args
                        log_teacher_softmax_viz(
                            probs,
                            ref_obs,
                            env_state,
                            int(step),
                            competence,
                            lp_cache,
                        )

                    step = metric["timestep"].max() * config["NUM_ENVS"]
                    jax.debug.callback(
                        _softmax_viz_callback,
                        (
                            step,
                            probs_snapshot,
                            ref_base_obs,
                            env_state,
                            competence_vector,
                            goal_learning_progress_cache,
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

            runner_state = (
                train_state,
                teacher_train_state,
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
                teacher_episode_carry,
                teacher_rollout_buffer,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        reward_dtype = obsv.dtype
        runner_state = (
            train_state,
            teacher_train_state,
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
            teacher_episode_carry,
            teacher_rollout_buffer,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        final_teacher_rollout_buffer = runner_state[-2]
        return {
            "runner_state": runner_state,
            "metrics": metric,
            "teacher_params": teacher_train_state.params,
            "teacher_train_state": teacher_train_state,
            "teacher_rollout_buffer": final_teacher_rollout_buffer,
        }

    return train, render_eval_episode, log_teacher_softmax_snapshot


def main():
    config_obj = parse_config_from_cli()
    config = asdict(config_obj)
    gpu_names = sorted({d.device_kind for d in jax.devices("gpu")})
    config["GPU_NAME"] = gpu_names[0]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["PPO", "BRAX", config["ENV_NAME"], f"jax_{jax.__version__}"],
        name=f'purejaxrl_ppo_brax_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn, render_eval_episode, log_teacher_softmax_snapshot = make_train(config)
    train_jit = jax.jit(train_fn)
    train_output = train_jit(rng)
    jax.block_until_ready(train_output["runner_state"][-1])
    (
        final_train_state,
        _final_teacher_train_state,
        final_env_state,
        final_last_obs,
        _,
        _,
        final_competence,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        final_episode_initial_base_obs,
        final_goal_learning_progress_cache,
        _,
        _,
        final_rng,
    ) = train_output["runner_state"]
    teacher_params = train_output["teacher_params"]
    if config["SAVE_MODEL"]:
        scratch = os.environ.get("SCRATCH")   
        random_name = RandomWord().word()
        random_id = np.random.randint(1000000)
        while os.path.exists(os.path.join(scratch, "purejaxrl", "checkpoints", random_name + f"_{random_id}")):
            random_name = RandomWord().word()
            random_id = np.random.randint(1000000)
        model_save_path =os.path.join(scratch, "purejaxrl", "checkpoints", random_name + f"_{random_id}")
        os.makedirs(model_save_path, exist_ok=False)
        save_checkpoint(final_train_state, model_save_path)
        load_checkpoint(final_train_state, model_save_path)
    render_eval_episode(
        final_train_state.params,
        teacher_params,
        final_competence,
        final_rng,
        final_env_state,
    )
    ref_env_index = int(config.get("TEACHER_SOFTMAX_VIZ_REF_ENV_INDEX", 0))
    ref_base_obs = np.asarray(
        jax.device_get(final_episode_initial_base_obs[ref_env_index])
    )
    log_teacher_softmax_snapshot(
        teacher_params,
        ref_base_obs,
        final_competence,
        final_env_state,
        int(config["TOTAL_TIMESTEPS"]),
        final_goal_learning_progress_cache,
    )


if __name__ == "__main__":
    main()
