import os
import traceback
import json
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
from dataclasses import dataclass, asdict
from flax.training.train_state import TrainState
import distrax
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


@dataclass
class TrainConfig:
    LR: float = 3e-4
    NUM_ENVS: int = 1024
    NUM_STEPS: int = 64
    TOTAL_TIMESTEPS: int = int(5e8)
    UPDATE_EPOCHS: int = 8
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.001
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTIVATION: str = "tanh"
    ENV_NAME: str = "ant_u_maze_single_goal"
    ENV_BACKEND: str | None = None
    EPISODE_LENGTH: int = 500
    ACTION_REPEAT: int = 1
    HIDDEN_DIM: int = 256
    TEACHER_HIDDEN_DIM: int = 256
    # Pass as JSON from CLI, e.g. --env-kwargs '{"foo": 1}'.
    ENV_KWARGS: str = "{}"
    ANNEAL_LR: bool = False
    NORMALIZE_ENV: bool = False
    DEBUG: bool = True
    SEED: int = 30
    WANDB_MODE: str = "online"
    ENTITY: str = ""
    PROJECT: str = "purejaxrl"
    WANDB_LOG_EVERY_UPDATES: int = 1
    EVAL_RENDER_STEPS: int = 300
    EVAL_RENDER_MAX_FRAMES: int = 100
    EVAL_RENDER_HEIGHT: int = 360
    EVAL_RENDER_LOG_WANDB_HTML: bool = True
    GOAL_CONDITIONED: bool = True
    TASK_REWARD_WEIGHT: float = 1.0
    GOAL_REWARD_WEIGHT: float = 1.0
    ANNEAL_GOAL_REWARD_WEIGHT: bool = False
    # When True, the student uses two separate value heads: one for the task
    # (weighted base env) reward and one for the (unweighted) goal-reaching
    # reward. The actor advantage is combined as
    # ``adv_task + GOAL_REWARD_WEIGHT * adv_goal``
    # while the total critic loss is the unweighted sum of both value losses.
    USE_SEPARATE_STUDENT_VALUE_FUNCTIONS: bool = False
    STUDENT_GOAL_REWARD_TYPE: str="sparse" # this can either sparse or dense
    # Optional goal-penalty normalization: "none", "running_std" (per-env running
    # std) or "batch_minmax" (per-rollout min-max, applied before GAE).
    GOAL_PENALTY_NORM_TYPE: str = "none"
    GOAL_PENALTY_NORM_EPS: float = 1e-8
    # Legacy continuous-delta bounds (unused in this discrete-teacher script,
    # kept so existing CLI invocations do not break).
    TEACHER_DELTA_LOW: float = -8.0
    TEACHER_DELTA_HIGH: float = 8.0
    # Discrete teacher action space: a grid of (x, y) offsets built with
    # linspace + meshgrid. The teacher emits a categorical index into this grid
    # and the goal is reference_obs[:2] + offset_grid[index].
    TEACHER_OFFSET_X_LOW: float = 0.0
    TEACHER_OFFSET_X_HIGH: float = 8.0
    TEACHER_OFFSET_Y_LOW: float = 0.0
    TEACHER_OFFSET_Y_HIGH: float = 10.0
    TEACHER_NUM_OFFSET_POINTS: int = 30  # points per axis; grid = points^2 actions
    TEACHER_NUM_PROBING_STATES: int = 100
    TEACHER_PROBE_AGG: str = "concat"
    TEACHER_SAMPLE_EVERY_N_EPISODES: int = 1
    GOAL_REACHED_THRESHOLD: float = 0.5
    SUCCESS_RATE_ALPHA: float = 0.05
    # "competence_lp" (default): teacher reward = student success-rate on the SAME
    # proposed goal evaluated after vs. before that episode's student updates.
    # "success_rate"/"goal_return": legacy cross-episode difference rewards.
    TEACHER_REWARD_TYPE: str = "competence_lp"
    TEACHER_TASK_RETURN_WEIGHT: float = 1.0
    # Competence-LP eval settings (only used when TEACHER_REWARD_TYPE == "competence_lp").
    TEACHER_EVAL_HORIZON: int = 500
    TEACHER_EVAL_EPISODES: int = 1
    TEACHER_EVAL_NUM_ENVS: int = 4
    TEACHER_LP_ABSOLUTE: bool = False
    TEACHER_BATCH_SIZE: int = 4096
    TEACHER_LR: float = 3e-4
    TEACHER_GAMMA: float = 0.999
    TEACHER_GAE_LAMBDA: float = 0.95
    TEACHER_CLIP_EPS: float = 0.2
    TEACHER_ENT_COEF: float = 0.05
    TEACHER_VF_COEF: float = 0.5
    TEACHER_MAX_GRAD_NORM: float = 0.5
    TEACHER_UPDATE_EPOCHS: int = 4
    TEACHER_NUM_MINIBATCHES: int = 4
    # Teacher goal visualization: scatter the last N proposed goals together
    # with the agent's starting x/y position.
    TEACHER_GOAL_VIZ_BUFFER_SIZE: int = 5000
    TEACHER_GOAL_VIZ_LOG_EVERY_UPDATES: int = 500
    TEACHER_GOAL_VIZ_LOG_WANDB: bool = True


    ### Bounding the studnet variacne
    BOUND_STUDENT_VARIANCE: bool = False
    BOUND_TEACHER_VARIANCE: bool = False
    # Per-minibatch advantage normalization: (gae - mean) / (std + eps).
    NORMALIZE_STUDENT_ADVANTAGE: bool = True
    NORMALIZE_TEACHER_ADVANTAGE: bool = True
    USE_ACTOR_PROBING_STATES: bool = False
    USE_CRITIC_PROBING_STATES: bool = True
    LAYER_NORM: bool = False
    UPDATE_GOAL_ON_REACH: bool = True

    def to_dict(self) -> dict[str, Any]:
        config = asdict(self)
        try:
            parsed_env_kwargs = json.loads(config["ENV_KWARGS"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON for ENV_KWARGS: {config['ENV_KWARGS']}"
            ) from exc
        if not isinstance(parsed_env_kwargs, dict):
            raise ValueError("ENV_KWARGS must decode to a JSON object.")
        config["ENV_KWARGS"] = parsed_env_kwargs
        return config


def parse_config_from_cli() -> TrainConfig:
    return tyro.cli(TrainConfig)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    bound_variance: bool = False
    hidden_dim: int = 256
    # When True, a second critic trunk/head is added so the module returns a
    # stacked value of shape ``(..., 2)`` = (task_value, goal_value). When False
    # the module returns a single squeezed value, preserving legacy behavior.
    separate_value_functions: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        else:
            x = x
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
        if self.bound_variance:
            pi = distrax.MultivariateNormalDiag(actor_mean,  jax.nn.sigmoid(actor_logtstd))
        else: 
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        def _critic_head(name_prefix):
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"{name_prefix}_dense0",
            )(x)
            h = activation(h)
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"{name_prefix}_dense1",
            )(h)
            h = activation(h)
            v = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name=f"{name_prefix}_out",
            )(h)
            return jnp.squeeze(v, axis=-1)

        if self.separate_value_functions:
            task_value = _critic_head("critic_task")
            goal_value = _critic_head("critic_goal")
            value = jnp.stack([task_value, goal_value], axis=-1)
        else:
            value = _critic_head("critic")

        return pi, value


class TeacherGoalPolicy(nn.Module):
    """Categorical (softmax) PPO policy over a discrete grid of goal offsets.

    The actor outputs ``num_actions`` logits, one per ``(x, y)`` offset in a
    precomputed grid (see ``offset_grid`` in ``make_train``). The teacher emits a
    categorical index; the goal is ``reference_obs[..., :2] + offset_grid[index]``.
    PPO trains directly on the discrete action index.
    """

    num_actions: int
    student_obs_dim: int
    num_probing_states: int
    probe_agg: str = "mean"
    activation: str = "tanh"
    hidden_dim: int = 256
    use_actor_probing_states: bool = False
    use_critic_probing_states: bool = True
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, student_apply, student_params):
        # jax.debug.print("teacher input shape: {x}", x=x)
        activation = nn.relu if self.activation == "relu" else nn.tanh

        if self.layer_norm:
            x = nn.LayerNorm()(x)
        else:
            x = x

        # Learnable probing states ~ U(-1, 1), sized to the student's observation
        # so they can be fed through the student policy.
        probing_states = self.param(
            "probing_states",
            lambda key, shape: jax.random.uniform(key, shape, jnp.float32, -1.0, 1.0),
            (self.num_probing_states, self.student_obs_dim),
        )
        # jax.debug.print("probing_states shape: {probing_states}", probing_states=probing_states[0])
        # Evaluate the student's deterministic actions at the probing states and
        # aggregate them into a single resulting vector.
        probe_pi, _ = student_apply(student_params, probing_states)
        probe_actions = probe_pi.mean()  # (K, action_dim)
        if self.probe_agg == "concat":
            probe_vec = probe_actions.reshape(-1)  # (K * action_dim,)
        else:
            probe_vec = jnp.mean(probe_actions, axis=0)  # (action_dim,)
        probe_b = jnp.broadcast_to(
            probe_vec, x.shape[:-1] + (probe_vec.shape[-1],)
        )
        # Actor sees the probing vector frozen so probing states are only learned
        # from the critic signal; the critic sees the live probing vector.
        if self.use_actor_probing_states:
            actor_in = jnp.concatenate([x, probe_b], axis=-1)
        else:
            actor_in = jnp.concatenate([x, jax.lax.stop_gradient(probe_b)], axis=-1)
        if self.use_critic_probing_states:
            critic_in = jnp.concatenate([x, probe_b], axis=-1)
        else:
            critic_in = jnp.concatenate([x, jax.lax.stop_gradient(probe_b)], axis=-1)

        actor_h = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_in)
        actor_h = activation(actor_h)
        actor_h = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_h)
        actor_h = activation(actor_h)
        logits = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_h)
        pi = distrax.Categorical(logits=logits)
        critic_h = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_in)
        critic_h = activation(critic_h)
        critic_h = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_h)
        critic_h = activation(critic_h)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_h
        )
        return pi, jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class TeacherTransition(NamedTuple):
    """One teacher transition = one ended teacher proposal event."""

    done: jnp.ndarray
    action: jnp.ndarray  # raw (pre-squash) delta
    value: jnp.ndarray
    reward: jnp.ndarray  # learning progress on the ended proposal
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # teacher input = [reference_obs, avg_success_rate]


class PendingTeacher(NamedTuple):
    """Per-env components of the goal proposal currently active for an episode."""

    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    # Student success rate on this proposal measured when it was proposed
    # (the "before" term of the competence-LP teacher reward).
    competence_before: jnp.ndarray
    # The proposed goal in normalized obs space, kept so competence can be
    # re-evaluated on the SAME goal at episode end.
    goal: jnp.ndarray


class GoalPenaltyNormState(NamedTuple):
    """Per-environment running mean/variance for the goal penalty.

    Used by ``GOAL_PENALTY_NORM_TYPE == "running_std"`` to divide the goal
    penalty by its per-env running standard deviation. Variance is initialized
    to 1 (and count to a small value) so early normalization stays numerically
    stable, mirroring the standard ``RunningMeanStd`` used for reward scaling.
    """

    mean: jnp.ndarray  # [NUM_ENVS]
    var: jnp.ndarray  # [NUM_ENVS]
    count: jnp.ndarray  # [NUM_ENVS]


class TeacherGoalVizBuffer(NamedTuple):
    """Fixed-size ring buffer holding the most recent teacher-proposed goals.

    ``goals`` keeps the last ``buffer_size`` 2D goals; ``write_idx`` is the next
    slot to overwrite and ``count`` is ``min(total_written, buffer_size)``.
    """

    goals: jnp.ndarray  # [buffer_size, 2]
    write_idx: jnp.ndarray  # scalar int32
    count: jnp.ndarray  # scalar int32


def _append_teacher_goals(buf, new_goals, record):
    """Append ``new_goals`` (``[N, 2]``) into the ring buffer when ``record``.

    Writing happens in a JAX-pure way: ``N`` consecutive (wrapped) slots are
    overwritten starting at ``write_idx``. When ``record`` is false the buffer
    is returned unchanged so this is safe to call every env step.
    """
    n = new_goals.shape[0]
    buffer_size = buf.goals.shape[0]
    idx = (buf.write_idx + jnp.arange(n)) % buffer_size
    updated_goals = buf.goals.at[idx].set(new_goals)
    goals = jnp.where(record, updated_goals, buf.goals)
    write_idx = jnp.where(record, (buf.write_idx + n) % buffer_size, buf.write_idx)
    count = jnp.where(
        record, jnp.minimum(buf.count + n, buffer_size), buf.count
    )
    return TeacherGoalVizBuffer(goals=goals, write_idx=write_idx, count=count)


def _extract_teacher_goals(buf) -> np.ndarray:
    """Return the valid goals from a (host-side) buffer as ``[M, 2]`` numpy."""
    goals = np.asarray(buf.goals)
    count = int(buf.count)
    buffer_size = goals.shape[0]
    if count >= buffer_size:
        return goals
    return goals[:count]


def _denorm_xy(xy, obs_mean, obs_var):
    """Invert ``NormalizeVecObservation`` on the x/y dims for plotting."""
    if obs_mean is None or obs_var is None:
        return xy
    mean_xy = np.asarray(obs_mean)[:2]
    std_xy = np.sqrt(np.asarray(obs_var)[:2] + 1e-8)
    return np.asarray(xy) * std_xy + mean_xy


def plot_teacher_goals(start_xy, goals, *, ax=None, title=None, save_path=None):
    """Scatter the agent start position and the teacher-proposed goals.

    ``start_xy`` is the agent's starting ``(x, y)`` and ``goals`` is an
    ``[M, 2]`` array of proposed goals. Returns ``(fig, ax)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    goals = np.asarray(goals)
    if goals.size > 0:
        ax.scatter(
            goals[:, 0],
            goals[:, 1],
            s=12,
            c="tab:blue",
            alpha=0.9,
            edgecolors="none",
            label=f"Teacher goals (n={goals.shape[0]})",
        )

    start_xy = np.asarray(start_xy).reshape(-1)
    ax.scatter(
        start_xy[0],
        start_xy[1],
        s=200,
        marker="*",
        c="tab:green",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
        label="Agent start",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_teacher_softmax(
    goal_grid_xy, probs, num_points, *, start_xy=None, title=None, save_path=None
):
    """Heatmap of the teacher's categorical distribution over its goal grid.

    ``goal_grid_xy`` is an ``[num_actions, 2]`` array of (de-normalized) goal
    positions (one per grid offset) and ``probs`` is the matching
    ``[num_actions]`` softmax probability vector. Both are reshaped to a
    ``[num_points, num_points]`` grid and drawn with ``pcolormesh``. Returns
    ``(fig, ax)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))

    goal_grid_xy = np.asarray(goal_grid_xy)
    probs = np.asarray(probs).reshape(-1)
    gx = goal_grid_xy[:, 0].reshape(num_points, num_points)
    gy = goal_grid_xy[:, 1].reshape(num_points, num_points)
    pgrid = probs.reshape(num_points, num_points)

    mesh = ax.pcolormesh(gx, gy, pgrid, shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="P(goal)")

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


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    student_episode_length = int(config.get("EPISODE_LENGTH", 1000))
    if student_episode_length < 1:
        raise ValueError("EPISODE_LENGTH must be >= 1")
    total_env_steps_per_env = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    config["NUM_STUDENT_EPISODES_PER_ENV"] = (
        total_env_steps_per_env // student_episode_length
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env_kwargs = config.get("ENV_KWARGS", {})
    custom_env = make_custom_env(
        env_name=config["ENV_NAME"],
        backend=config.get("ENV_BACKEND"),
        env_kwargs=env_kwargs,
    )
    if custom_env is not None:
        base_env = custom_env
    else:
        base_env = brax_envs.get_environment(
            env_name=config["ENV_NAME"],
            backend=config.get("ENV_BACKEND", "positional"),
        )
    env = BraxGymnaxWrapper(
        env=base_env,
        episode_length=config.get("EPISODE_LENGTH", 1000),
        action_repeat=config.get("ACTION_REPEAT", 1),
    )
    env_params = None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    goal_conditioned = config.get("GOAL_CONDITIONED", False)
    task_reward_weight = float(config.get("TASK_REWARD_WEIGHT", 1.0))
    goal_reward_weight = config.get("GOAL_REWARD_WEIGHT", 0.0)
    anneal_goal_reward_weight = config.get("ANNEAL_GOAL_REWARD_WEIGHT", False)
    use_separate_student_value_functions = bool(
        config.get("USE_SEPARATE_STUDENT_VALUE_FUNCTIONS", False)
    )
    teacher_goal_viz_buffer_size = int(
        config.get("TEACHER_GOAL_VIZ_BUFFER_SIZE", 10000)
    )
    bound_student_variance = config.get("BOUND_STUDENT_VARIANCE", False)
    bound_teacher_variance = config.get("BOUND_TEACHER_VARIANCE", False)
    normalize_student_advantage = bool(
        config.get("NORMALIZE_STUDENT_ADVANTAGE", True)
    )
    normalize_teacher_advantage = bool(
        config.get("NORMALIZE_TEACHER_ADVANTAGE", True)
    )
    if teacher_goal_viz_buffer_size < 1:
        raise ValueError("TEACHER_GOAL_VIZ_BUFFER_SIZE must be >= 1")
    teacher_goal_viz_log_every_updates = int(
        config.get("TEACHER_GOAL_VIZ_LOG_EVERY_UPDATES", 100)
    )
    if teacher_goal_viz_log_every_updates < 1:
        raise ValueError("TEACHER_GOAL_VIZ_LOG_EVERY_UPDATES must be >= 1")
    teacher_goal_viz_log_wandb = bool(config.get("TEACHER_GOAL_VIZ_LOG_WANDB", True))
    student_goal_reward_type = config.get("STUDENT_GOAL_REWARD_TYPE", "sparse")
    if student_goal_reward_type not in ("sparse", "dense"):
        raise ValueError(
            "STUDENT_GOAL_REWARD_TYPE must be 'sparse' or 'dense'"
        )
    goal_penalty_norm_type = config.get("GOAL_PENALTY_NORM_TYPE", "none")
    if goal_penalty_norm_type not in ("none", "running_std", "batch_minmax"):
        raise ValueError(
            "GOAL_PENALTY_NORM_TYPE must be 'none', 'running_std' or "
            "'batch_minmax'"
        )
    goal_penalty_norm_eps = float(config.get("GOAL_PENALTY_NORM_EPS", 1e-8))
    teacher_delta_low = config.get("TEACHER_DELTA_LOW", -1.0)
    teacher_delta_high = config.get("TEACHER_DELTA_HIGH", 1.0)
    goal_reached_threshold = config.get("GOAL_REACHED_THRESHOLD", 0.1)
    success_rate_alpha = config.get("SUCCESS_RATE_ALPHA", 0.05)
    teacher_reward_type = config.get("TEACHER_REWARD_TYPE", "competence_lp")
    teacher_task_return_weight = float(config.get("TEACHER_TASK_RETURN_WEIGHT", 1.0))
    if teacher_reward_type not in ("success_rate", "goal_return", "competence_lp"):
        raise ValueError(
            "TEACHER_REWARD_TYPE must be 'success_rate', 'goal_return' or "
            "'competence_lp'"
        )
    teacher_eval_horizon = int(config.get("TEACHER_EVAL_HORIZON", 250))
    teacher_eval_episodes = int(config.get("TEACHER_EVAL_EPISODES", 1))
    teacher_eval_num_envs = int(config.get("TEACHER_EVAL_NUM_ENVS", 4))
    if teacher_eval_horizon < 1:
        raise ValueError("TEACHER_EVAL_HORIZON must be >= 1")
    if teacher_eval_episodes < 1:
        raise ValueError("TEACHER_EVAL_EPISODES must be >= 1")
    if teacher_eval_num_envs < 1:
        raise ValueError("TEACHER_EVAL_NUM_ENVS must be >= 1")
    teacher_lp_absolute = bool(config.get("TEACHER_LP_ABSOLUTE", False))
    base_obs_dim = int(env.observation_space(env_params).shape[0])
    goal_dim = 2
    policy_obs_dim = base_obs_dim + goal_dim if goal_conditioned else base_obs_dim
    teacher_num_probing_states = int(config.get("TEACHER_NUM_PROBING_STATES", 8))
    teacher_probe_agg = config.get("TEACHER_PROBE_AGG", "mean")

    # Discrete teacher action space: a grid of (x, y) offsets. The teacher emits
    # a categorical index into ``offset_grid`` and the goal is
    # ``reference_obs[:2] + offset_grid[index]``.
    teacher_offset_x_low = float(config.get("TEACHER_OFFSET_X_LOW", 0.0))
    teacher_offset_x_high = float(config.get("TEACHER_OFFSET_X_HIGH", 8.0))
    teacher_offset_y_low = float(config.get("TEACHER_OFFSET_Y_LOW", 0.0))
    teacher_offset_y_high = float(config.get("TEACHER_OFFSET_Y_HIGH", 10.0))
    teacher_num_offset_points = int(config.get("TEACHER_NUM_OFFSET_POINTS", 30))
    if teacher_num_offset_points < 1:
        raise ValueError("TEACHER_NUM_OFFSET_POINTS must be >= 1")
    _offset_xs = jnp.linspace(
        teacher_offset_x_low, teacher_offset_x_high, teacher_num_offset_points
    )
    _offset_ys = jnp.linspace(
        teacher_offset_y_low, teacher_offset_y_high, teacher_num_offset_points
    )
    _offset_gx, _offset_gy = jnp.meshgrid(_offset_xs, _offset_ys)
    offset_grid = jnp.stack(
        [_offset_gx.reshape(-1), _offset_gy.reshape(-1)], axis=-1
    )  # [num_actions, 2]
    num_teacher_actions = teacher_num_offset_points * teacher_num_offset_points

    # Teacher PPO hyper-parameters. The teacher collects one transition per
    # ended proposal event (done OR reached) and updates once
    # TEACHER_BATCH_SIZE transitions are buffered.
    teacher_batch_size = int(config.get("TEACHER_BATCH_SIZE", 4096))
    if teacher_batch_size < 1:
        raise ValueError("TEACHER_BATCH_SIZE must be >= 1")
    total_student_env_steps = (
        int(config["NUM_UPDATES"]) * int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])
    )
    config["TEACHER_NUM_UPDATES"] = max(total_student_env_steps // teacher_batch_size, 1)
    config["TEACHER_STUDENT_UPDATE_RATIO"] = (
        config["TEACHER_NUM_UPDATES"] / config["NUM_UPDATES"]
        if config["NUM_UPDATES"] > 0
        else 0.0
    )
    teacher_lr = config.get("TEACHER_LR", config["LR"])
    teacher_gamma = config.get("TEACHER_GAMMA", config["GAMMA"])
    teacher_gae_lambda = config.get("TEACHER_GAE_LAMBDA", config["GAE_LAMBDA"])
    teacher_clip_eps = config.get("TEACHER_CLIP_EPS", config["CLIP_EPS"])
    teacher_ent_coef = config.get("TEACHER_ENT_COEF", config["ENT_COEF"])
    teacher_vf_coef = config.get("TEACHER_VF_COEF", config["VF_COEF"])
    teacher_max_grad_norm = config.get("TEACHER_MAX_GRAD_NORM", config["MAX_GRAD_NORM"])
    teacher_update_epochs = int(config.get("TEACHER_UPDATE_EPOCHS", config["UPDATE_EPOCHS"]))
    teacher_num_minibatches = int(config.get("TEACHER_NUM_MINIBATCHES", 1))
    if teacher_batch_size % teacher_num_minibatches != 0:
        raise ValueError(
            "TEACHER_BATCH_SIZE must be divisible by TEACHER_NUM_MINIBATCHES"
        )
    config["TEACHER_MINIBATCH_SIZE"] = teacher_batch_size // teacher_num_minibatches

    hidden_dim = config.get("HIDDEN_DIM", 256)
    teacher_hidden_dim = config.get("TEACHER_HIDDEN_DIM", 256)
    layer_norm = config.get("LAYER_NORM", False)

    network = ActorCritic(
        env.action_space(env_params).shape[0], activation=config["ACTIVATION"], bound_variance=bound_student_variance, hidden_dim=hidden_dim,
        separate_value_functions=use_separate_student_value_functions,
        layer_norm=layer_norm,
    )

    # Trainable teacher policy: maps a reference observation to a categorical
    # distribution over the discrete grid of goal offsets.
    use_actor_probing_states = config.get("USE_ACTOR_PROBING_STATES", False)
    use_critic_probing_states = config.get("USE_CRITIC_PROBING_STATES", True)
    teacher_network = TeacherGoalPolicy(
        num_actions=num_teacher_actions,
        student_obs_dim=policy_obs_dim,
        num_probing_states=teacher_num_probing_states,
        probe_agg=teacher_probe_agg,
        activation=config["ACTIVATION"],
        hidden_dim=teacher_hidden_dim,
        use_actor_probing_states=use_actor_probing_states,
        use_critic_probing_states=use_critic_probing_states,
        layer_norm=layer_norm,
    )
    teacher_rng, student_dummy_rng = jax.random.split(
        jax.random.PRNGKey(config.get("SEED", 0))
    )
    dummy_student_params = network.init(student_dummy_rng, jnp.zeros((policy_obs_dim,)))
    teacher_params = teacher_network.init(
        teacher_rng, jnp.zeros((base_obs_dim + 1,)), network.apply, dummy_student_params
    )

    def _teacher_input(reference_obs, avg_success_rate):
        """Build the teacher observation ``[reference_obs, avg_success_rate]``."""
        avg_success_rate = jnp.asarray(avg_success_rate, dtype=reference_obs.dtype)
        if reference_obs.ndim == 1:
            return jnp.concatenate(
                [reference_obs, jnp.reshape(avg_success_rate, (1,))], axis=-1
            )
        return jnp.concatenate([reference_obs, avg_success_rate[..., None]], axis=-1)

    def _index_to_offset(action):
        """Map a categorical action index to its ``(x, y)`` offset on the grid."""
        return offset_grid[action]

    def _teacher_apply(teacher_params, teacher_input, student_params):
        return teacher_network.apply(
            teacher_params, teacher_input, network.apply, student_params
        )

    def _teacher_sample(
        teacher_params, reference_obs, avg_success_rate, student_params, rng
    ):
        """Sample a goal from the teacher policy.

        Returns ``(goal, teacher_input, action, log_prob, value)`` where
        ``action`` is the discrete grid index, so the proposal can be stored as a
        PPO transition.
        """
        teacher_input = _teacher_input(reference_obs, avg_success_rate)
        pi, value = _teacher_apply(teacher_params, teacher_input, student_params)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        goal = reference_obs[..., :2] + _index_to_offset(action)
        return goal, teacher_input, action, log_prob, value

    def _teacher_goal_det(teacher_params, reference_obs, avg_success_rate, student_params):
        """Deterministic goal (uses the most-likely action) for eval/rendering."""
        pi, _ = _teacher_apply(
            teacher_params, _teacher_input(reference_obs, avg_success_rate), student_params
        )
        return reference_obs[..., :2] + _index_to_offset(pi.mode())

    def _concat_goal(obs, goals):
        if not goal_conditioned:
            return obs
        return jnp.concatenate([obs, goals], axis=-1)

    def _goal_xy_delta(obs, goals):
        # Goal reaching is defined in the positional plane only.
        return obs[..., :2] - goals[..., :2]

    def _update_goal_penalty_norm(norm_state, x):
        """Per-env RunningMeanStd update with one new sample per env (batch=1)."""
        delta = x - norm_state.mean
        tot_count = norm_state.count + 1.0
        mean = norm_state.mean + delta / tot_count
        m_a = norm_state.var * norm_state.count
        m2 = m_a + jnp.square(delta) * norm_state.count / tot_count
        var = m2 / tot_count
        return GoalPenaltyNormState(mean=mean, var=var, count=tot_count)

    def _goal_penalty_running_std(norm_state):
        """Per-env running standard deviation (variance floored by eps)."""
        return jnp.sqrt(norm_state.var + goal_penalty_norm_eps)

    def _make_linear_lr_schedule(base_lr, steps_per_update, num_updates):
        """Build a linear LR decay schedule shared by the student and teacher.

        ``frac`` goes from 1.0 -> 0.0 across the agent's full set of PPO updates,
        so both agents anneal at the same rate relative to training progress.
        ``steps_per_update`` is the number of gradient steps per update
        (minibatches * update epochs) used to convert the optax step count into
        an update index.
        """
        steps_per_update = max(int(steps_per_update), 1)
        num_updates = max(int(num_updates), 1)

        def schedule(count):
            frac = 1.0 - (count // steps_per_update) / num_updates
            return base_lr * frac

        return schedule

    linear_schedule = _make_linear_lr_schedule(
        config["LR"],
        config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"],
        config["NUM_UPDATES"],
    )
    teacher_linear_schedule = _make_linear_lr_schedule(
        teacher_lr,
        teacher_num_minibatches * teacher_update_epochs,
        config["TEACHER_NUM_UPDATES"],
    )

    def goal_reward_weight_schedule(update_idx):
        if config["NUM_UPDATES"] <= 1:
            return jnp.array(0.0, dtype=jnp.float32)
        frac = 1.0 - (update_idx / (config["NUM_UPDATES"] - 1))
        return jnp.clip(frac, 0.0, 1.0)

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

    render_sim_steps = int(config.get("EVAL_RENDER_STEPS", 500))
    render_max_frames = int(config.get("EVAL_RENDER_MAX_FRAMES", 120))
    render_action_repeat = int(config.get("ACTION_REPEAT", 1))
    action_low = jnp.asarray(env.action_space(env_params).low)
    action_high = jnp.asarray(env.action_space(env_params).high)
    _render_obs_dim = base_obs_dim

    def _obs_norm_stats_for_render(final_env_state):
        obs_mean, obs_var = _extract_obs_norm_stats(final_env_state, _render_obs_dim)
        if obs_mean is None:
            obs_mean = jnp.zeros(_render_obs_dim)
            obs_var = jnp.ones(_render_obs_dim)
        return obs_mean, obs_var

    def _eval_render_action(params, obs, obs_mean, obs_var, goal):
        norm_obs = _normalize_eval_obs(obs, obs_mean, obs_var)
        policy_obs = _concat_goal(norm_obs, goal)
        pi, _ = network.apply(params, policy_obs)
        return jnp.clip(pi.mean(), action_low, action_high)

    # Batched eval used by the "competence_lp" teacher reward: roll out the
    # (deterministic) student conditioned on each proposed goal and measure the
    # success rate of reaching it within ``goal_reached_threshold``.
    _eval_env_reset = jax.vmap(base_env.reset)
    _eval_env_step = jax.vmap(base_env.step)

    def _eval_goal_competence(student_params, goals, obs_mean, obs_var, rng):
        """Per-goal success rate of the deterministic student.

        ``goals`` has shape ``[N, goal_dim]`` in normalized observation space.
        For each goal we run ``TEACHER_EVAL_EPISODES * TEACHER_EVAL_NUM_ENVS``
        independent fixed-horizon rollouts from fresh resets and return the
        fraction that reach the goal, as ``[N]`` in ``[0, 1]``.
        """
        n_goals = goals.shape[0]
        reps = teacher_eval_episodes * teacher_eval_num_envs
        batch = n_goals * reps
        # [N, reps, goal_dim] -> [N * reps, goal_dim] (goal index varies slowest).
        goals_b = jnp.broadcast_to(
            goals[:, None, :], (n_goals, reps, goals.shape[-1])
        ).reshape(batch, goals.shape[-1])
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, batch)
        state = _eval_env_reset(reset_rngs)

        def step_fn(carry, _):
            state, reached = carry
            action = _eval_render_action(
                student_params, state.obs, obs_mean, obs_var, goals_b
            )
            state = _eval_env_step(state, action)
            norm_obs = _normalize_eval_obs(state.obs, obs_mean, obs_var)
            goal_dist = jnp.sqrt(
                jnp.sum(jnp.square(_goal_xy_delta(norm_obs, goals_b)), axis=-1)
            )
            reached = jnp.logical_or(reached, goal_dist <= goal_reached_threshold)
            return (state, reached), None

        (_, reached), _ = jax.lax.scan(
            step_fn,
            (state, jnp.zeros((batch,), dtype=bool)),
            None,
            length=teacher_eval_horizon,
        )
        success = reached.astype(jnp.float32).reshape(n_goals, reps)
        return jnp.mean(success, axis=-1)

    def _render_rollout_impl(params, teacher_params, rng, obs_mean, obs_var):
        rng, reset_rng = jax.random.split(rng)
        state = base_env.reset(reset_rng)
        norm_obs0 = _normalize_eval_obs(state.obs, obs_mean, obs_var)
        eval_goal = _teacher_goal_det(teacher_params, norm_obs0, 0.0, params)

        def step_fn(carry, _):
            state, goal = carry
            action = _eval_render_action(params, state.obs, obs_mean, obs_var, goal)

            def repeat_step(s, __):
                return base_env.step(s, action), None

            state, _ = jax.lax.scan(
                repeat_step, state, None, length=render_action_repeat
            )
            # Refresh the goal once the current goal is reached.
            norm_obs = _normalize_eval_obs(state.obs, obs_mean, obs_var)
            goal_dist = jnp.sqrt(
                jnp.sum(jnp.square(_goal_xy_delta(norm_obs, goal)), axis=-1)
            )
            reached = goal_dist <= goal_reached_threshold
            goal = jnp.where(
                reached,
                _teacher_goal_det(teacher_params, norm_obs, 0.0, params),
                goal,
            )
            return (state, goal), state.pipeline_state

        _, pipeline_states = jax.lax.scan(
            step_fn, (state, eval_goal), None, length=render_sim_steps
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

    def render_eval_episode(params, teacher_params, rng, final_env_state):
        """Runs one eval rollout and logs rendered HTML to wandb."""
        if config.get("WANDB_MODE", "disabled") != "online":
            return
        try:
            obs_mean, obs_var = _obs_norm_stats_for_render(final_env_state)
            pipeline_states = _run_render_rollout(
                params, teacher_params, rng, obs_mean, obs_var
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

    def log_teacher_goal_viz_plot(goal_viz_buffer, agent_start_xy, final_env_state, step):
        """Plot the agent start and teacher-proposed goals; log to wandb."""
        try:
            import matplotlib.pyplot as plt

            buffer_host = jax.device_get(goal_viz_buffer)
            start_xy = np.asarray(jax.device_get(agent_start_xy)).reshape(-1)
            goals = _extract_teacher_goals(buffer_host)

            obs_mean, obs_var = _extract_obs_norm_stats(final_env_state, base_obs_dim)
            if config.get("NORMALIZE_ENV", False) and obs_mean is not None:
                obs_mean = jax.device_get(obs_mean)
                obs_var = jax.device_get(obs_var)
                start_xy = _denorm_xy(start_xy, obs_mean, obs_var)
                if goals.shape[0] > 0:
                    goals = _denorm_xy(goals, obs_mean, obs_var)

            if goals.shape[0] == 0:
                print("[log_teacher_goal_viz_plot] goal buffer empty; plotting start only.")

            exp_dir = wandb.run.dir if wandb.run is not None else os.getcwd()
            exp_name = (
                wandb.run.name
                if wandb.run is not None and wandb.run.name
                else f'purejaxrl_ppo_brax_{config["ENV_NAME"]}'
            )
            save_path = os.path.join(exp_dir, f"{exp_name}_teacher_goals_{step}.png")
            fig, _ = plot_teacher_goals(
                start_xy,
                goals,
                title=f'Teacher goals @ step {step} ({config["ENV_NAME"]})',
                save_path=save_path,
            )
            if (
                teacher_goal_viz_log_wandb
                and config.get("WANDB_MODE", "disabled") == "online"
            ):
                wandb.log({"teacher/goal_scatter": wandb.Image(fig)}, step=step)
            plt.close(fig)
        except Exception as err:
            print(f"[log_teacher_goal_viz_plot] skipped goal plot: {err}")
            traceback.print_exc()

    def log_teacher_softmax_viz_plot(
        teacher_params, student_params, ref_obs, avg_sr, final_env_state, step
    ):
        """Plot the teacher's softmax distribution over the goal grid; log to wandb."""
        try:
            import matplotlib.pyplot as plt

            ref_obs = jnp.asarray(jax.device_get(ref_obs)).reshape(-1)
            avg_sr = jnp.asarray(jax.device_get(avg_sr)).reshape(())
            pi, _ = _teacher_apply(
                teacher_params, _teacher_input(ref_obs, avg_sr), student_params
            )
            probs = np.asarray(jax.device_get(pi.probs)).reshape(-1)
            goal_grid_xy = np.asarray(
                jax.device_get(ref_obs[:2] + offset_grid)
            )
            start_xy = np.asarray(jax.device_get(ref_obs[:2])).reshape(-1)

            obs_mean, obs_var = _extract_obs_norm_stats(final_env_state, base_obs_dim)
            if config.get("NORMALIZE_ENV", False) and obs_mean is not None:
                obs_mean = jax.device_get(obs_mean)
                obs_var = jax.device_get(obs_var)
                goal_grid_xy = _denorm_xy(goal_grid_xy, obs_mean, obs_var)
                start_xy = _denorm_xy(start_xy, obs_mean, obs_var)

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
                teacher_num_offset_points,
                start_xy=start_xy,
                title=f'Teacher softmax @ step {step} ({config["ENV_NAME"]})',
                save_path=save_path,
            )
            if (
                teacher_goal_viz_log_wandb
                and config.get("WANDB_MODE", "disabled") == "online"
            ):
                wandb.log({"teacher/goal_softmax": wandb.Image(fig)}, step=step)
            plt.close(fig)
        except Exception as err:
            print(f"[log_teacher_softmax_viz_plot] skipped softmax plot: {err}")
            traceback.print_exc()

    def train(rng):
        wandb_log_every_updates = int(config.get("WANDB_LOG_EVERY_UPDATES", 10))
        if wandb_log_every_updates < 1:
            raise ValueError("WANDB_LOG_EVERY_UPDATES must be >= 1")

        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((policy_obs_dim,))
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

        # INIT TEACHER (PPO agent operating on the student-episode timeline)
        if config["ANNEAL_LR"]:
            teacher_tx = optax.chain(
                optax.clip_by_global_norm(teacher_max_grad_norm),
                optax.adam(learning_rate=teacher_linear_schedule, eps=1e-5),
            )
        else:
            teacher_tx = optax.chain(
                optax.clip_by_global_norm(teacher_max_grad_norm),
                optax.adam(teacher_lr, eps=1e-5),
            )
        teacher_train_state = TrainState.create(
            apply_fn=teacher_network.apply,
            params=teacher_params,
            tx=teacher_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        # jax.debug.print("obsv: {obsv}", obsv=obsv)
        avg_success_rate = jnp.zeros((config["NUM_ENVS"],), dtype=obsv.dtype)
        goal_penalty_norm_state = GoalPenaltyNormState(
            mean=jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32),
            var=jnp.ones((config["NUM_ENVS"],), dtype=jnp.float32),
            count=jnp.full((config["NUM_ENVS"],), 1e-4, dtype=jnp.float32),
        )
        # Teacher proposes the initial goal (sampled) from each env's initial obs.
        rng, _rng = jax.random.split(rng)
        teacher_rng, sample_rng = jax.random.split(_rng)
        (
            goal_batch,
            init_t_obs,
            init_t_action,
            init_t_log_prob,
            init_t_value,
        ) = _teacher_sample(
            teacher_train_state.params,
            obsv,
            avg_success_rate,
            train_state.params,
            sample_rng,
        )
        # Initial "before" competence for the first proposed goal, measured with
        # the freshly initialized student. Uses the post-reset obs-norm stats.
        if teacher_reward_type == "competence_lp":
            init_obs_mean, init_obs_var = _extract_obs_norm_stats(
                env_state, base_obs_dim
            )
            if init_obs_mean is None:
                init_obs_mean = jnp.zeros(base_obs_dim)
                init_obs_var = jnp.ones(base_obs_dim)
            teacher_rng, init_eval_rng = jax.random.split(teacher_rng)
            init_competence_before = _eval_goal_competence(
                train_state.params,
                goal_batch,
                init_obs_mean,
                init_obs_var,
                init_eval_rng,
            )
        else:
            init_competence_before = jnp.zeros(
                (config["NUM_ENVS"],), dtype=jnp.float32
            )
        pending = PendingTeacher(
            obs=init_t_obs,
            action=init_t_action,
            value=init_t_value,
            log_prob=init_t_log_prob,
            competence_before=init_competence_before,
            goal=goal_batch,
        )
        # Teacher rollout buffer: one slot per ended proposal event, plus a
        # trailing padding row that absorbs out-of-bounds / invalid writes.
        _buf_rows = teacher_batch_size + 1
        teacher_buffer = TeacherTransition(
            done=jnp.zeros((_buf_rows,)),
            action=jnp.zeros((_buf_rows,), dtype=jnp.int32),
            value=jnp.zeros((_buf_rows,)),
            reward=jnp.zeros((_buf_rows,)),
            log_prob=jnp.zeros((_buf_rows,)),
            obs=jnp.zeros((_buf_rows, base_obs_dim + 1)),
        )
        teacher_buffer_count = jnp.asarray(0, dtype=jnp.int32)

        # Agent starting x/y (mean across parallel envs) for the goal viz plot.
        agent_start_xy = jnp.mean(obsv[:, :2], axis=0)
        # Ring buffer of recent teacher-proposed goals (seeded with the first proposal).
        goal_viz_buffer = TeacherGoalVizBuffer(
            goals=jnp.zeros((teacher_goal_viz_buffer_size, goal_dim)),
            write_idx=jnp.int32(0),
            count=jnp.int32(0),
        )
        goal_viz_buffer = _append_teacher_goals(
            goal_viz_buffer, goal_batch, jnp.bool_(goal_conditioned)
        )

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            current_goal_reward_weight = (
                goal_reward_weight_schedule(update_idx)
                if anneal_goal_reward_weight
                else jnp.asarray(goal_reward_weight, dtype=jnp.float32)
            )
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    goal_batch,
                    goal_ep_returns,
                    episode_counts,
                    avg_success_rate,
                    reached_any_in_episode,
                    prev_episode_success,
                    prev_goal_ep_return,
                    rng,
                    teacher_train_state,
                    teacher_buffer,
                    teacher_buffer_count,
                    pending,
                    teacher_rng,
                    goal_penalty_norm_state,
                    goal_viz_buffer,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                policy_obs = _concat_goal(last_obs, goal_batch)
                pi, value = network.apply(train_state.params, policy_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                student_reward = reward
                task_reward_term = task_reward_weight * student_reward
                goal_reward_term = jnp.zeros_like(reward)
                # Unweighted (post-normalization) goal-reaching reward. The goal
                # value head regresses to this stream so GOAL_REWARD_WEIGHT only
                # trades off optimization in the actor advantage, not in the
                # critic target.
                goal_reward_unweighted = jnp.zeros_like(reward)
                reward = task_reward_term
                episode_success = jnp.zeros_like(reward)
                raw_goal_penalty = jnp.zeros_like(reward)
                # Competence-LP scratch values (filled in the goal-conditioned
                # branch when TEACHER_REWARD_TYPE == "competence_lp").
                comp_after = jnp.zeros_like(reward)
                comp_before_new = jnp.zeros_like(reward)
                comp_before_old = jnp.zeros_like(reward)
                if goal_conditioned:
                    if student_goal_reward_type == "sparse":
                        goal_dist = jnp.sqrt(
                            jnp.sum(jnp.square(_goal_xy_delta(obsv, goal_batch)), axis=-1)
                        )
                        goal_penalty = (goal_dist <= goal_reached_threshold).astype(reward.dtype)
                    elif student_goal_reward_type == "dense":
                        goal_penalty = -jnp.sum(
                            jnp.square(_goal_xy_delta(obsv, goal_batch)), axis=-1
                        )

                    # Keep the unnormalized penalty for logging and for the
                    # batch_minmax path (applied post-rollout, before GAE).
                    raw_goal_penalty = goal_penalty
                    if goal_penalty_norm_type == "running_std":
                        goal_penalty_norm_state = _update_goal_penalty_norm(
                            goal_penalty_norm_state, raw_goal_penalty
                        )
                        running_std = _goal_penalty_running_std(goal_penalty_norm_state)
                        goal_penalty = raw_goal_penalty / running_std

                    goal_reward_unweighted = goal_penalty
                    goal_reward_term = current_goal_reward_weight * goal_penalty
                    reward = task_reward_term + goal_reward_term
                    # End the current teacher proposal on done OR goal reached.
                    goal_dist = jnp.sqrt(
                        jnp.sum(jnp.square(_goal_xy_delta(obsv, goal_batch)), axis=-1)
                    )
                    reached = goal_dist <= goal_reached_threshold
                    reached_any_in_episode = jnp.logical_or(reached_any_in_episode, reached)
                    done_mask = done.astype(bool)
                    proposal_end_mask = jnp.logical_or(done_mask, reached)
                    completed_episode_counts = episode_counts + done_mask.astype(jnp.int32)
                    episode_success = reached_any_in_episode.astype(reward.dtype)
                    updated_avg_success_rate = avg_success_rate + success_rate_alpha * (
                        episode_success - avg_success_rate
                    )
                    avg_success_rate = jnp.where(
                        done_mask, updated_avg_success_rate, avg_success_rate
                    )
                    teacher_rng, sample_rng = jax.random.split(teacher_rng)
                    (
                        new_goals,
                        new_t_obs,
                        new_t_action,
                        new_t_log_prob,
                        new_t_value,
                    ) = _teacher_sample(
                        teacher_train_state.params,
                        obsv,
                        avg_success_rate,
                        train_state.params,
                        sample_rng,
                    )
                    goal_batch = jnp.where(proposal_end_mask[:, None], new_goals, goal_batch)
                    # Record the freshly proposed goals for visualization.
                    goal_viz_buffer = _append_teacher_goals(
                        goal_viz_buffer, new_goals, jnp.any(proposal_end_mask)
                    )
                    episode_counts = completed_episode_counts
                    reached_any_in_episode = jnp.where(
                        done_mask,
                        jnp.zeros_like(reached_any_in_episode),
                        reached_any_in_episode,
                    )
                    # COMPETENCE-LP: re-evaluate success on the SAME goal that was
                    # active (`pending.goal`, the "after" term) and on the freshly
                    # proposed goal (the next proposal's "before" term) whenever a
                    # proposal ends (done OR reached).
                    if teacher_reward_type == "competence_lp":
                        obs_mean, obs_var = _extract_obs_norm_stats(
                            env_state, base_obs_dim
                        )
                        if obs_mean is None:
                            obs_mean = jnp.zeros(base_obs_dim)
                            obs_var = jnp.ones(base_obs_dim)

                        def _do_competence_eval(operands):
                            old_goals, fresh_goals, student_params, e_rng = operands
                            r_after, r_before = jax.random.split(e_rng)
                            c_after = _eval_goal_competence(
                                student_params, old_goals, obs_mean, obs_var, r_after
                            )
                            c_before = _eval_goal_competence(
                                student_params, fresh_goals, obs_mean, obs_var, r_before
                            )
                            return c_after, c_before

                        def _skip_competence_eval(operands):
                            zeros = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32)
                            return zeros, zeros

                        teacher_rng, eval_rng = jax.random.split(teacher_rng)
                        comp_after, comp_before_new = jax.lax.cond(
                            jnp.any(proposal_end_mask),
                            _do_competence_eval,
                            _skip_competence_eval,
                            (pending.goal, new_goals, train_state.params, eval_rng),
                        )
                else:
                    done_mask = done.astype(bool)
                    proposal_end_mask = done_mask
                    episode_counts = episode_counts + done_mask.astype(jnp.int32)
                    reached_any_in_episode = jnp.where(
                        done_mask,
                        jnp.zeros_like(reached_any_in_episode),
                        reached_any_in_episode,
                    )
                goal_ep_returns = goal_ep_returns + goal_reward_term
                returned_goal_ep_returns = jnp.where(
                    done_mask, goal_ep_returns, jnp.zeros_like(goal_ep_returns)
                )
                goal_ep_returns = jnp.where(
                    done_mask, jnp.zeros_like(goal_ep_returns), goal_ep_returns
                )
                # TEACHER REWARD = learning progress on reaching the proposed goal.
                # Computed per env only on episode completion as the difference
                # between the current and previous episode's value.
                current_success = episode_success
                current_goal_return = returned_goal_ep_returns
                task_episode_return = jnp.where(
                    done_mask,
                    info["returned_episode_returns"],
                    jnp.zeros_like(reward),
                )
                weighted_task_return = teacher_task_return_weight * task_episode_return
                teacher_event_mask = (
                    proposal_end_mask if teacher_reward_type == "competence_lp" else done_mask
                )
                if teacher_reward_type == "success_rate":
                    progress = (current_success - prev_episode_success) + weighted_task_return
                    teacher_reward = jnp.where(
                        done_mask, progress, jnp.zeros_like(progress)
                    )
                elif teacher_reward_type == "goal_return":
                    progress = (current_goal_return - prev_goal_ep_return) + weighted_task_return
                    teacher_reward = jnp.where(
                        done_mask, progress, jnp.zeros_like(progress)
                    )
                else:  # competence_lp
                    # Same-goal learning progress: success rate after this
                    # episode's student updates minus the success rate measured
                    # when the goal was proposed.
                    comp_before_old = pending.competence_before
                    lp = comp_after - comp_before_old
                    if teacher_lp_absolute:
                        lp = jnp.abs(lp)
                    teacher_reward = jnp.where(
                        teacher_event_mask,
                        lp + weighted_task_return,
                        jnp.zeros_like(lp),
                    )
                prev_episode_success = jnp.where(
                    done_mask, current_success, prev_episode_success
                )
                prev_goal_ep_return = jnp.where(
                    done_mask, current_goal_return, prev_goal_ep_return
                )

                if goal_conditioned:
                    # TEACHER COLLECTION: emit one transition per proposal-end
                    # event (done OR reached in competence_lp mode), using the
                    # proposal that was active before rollover (`pending`).
                    event_int = teacher_event_mask.astype(jnp.int32)
                    event_rank = jnp.cumsum(event_int) - 1
                    raw_write_idx = teacher_buffer_count + event_rank
                    valid_event = teacher_event_mask & (raw_write_idx < teacher_batch_size)
                    safe_write_idx = jnp.where(
                        valid_event, raw_write_idx, teacher_batch_size
                    )
                    event_obs = jnp.where(
                        teacher_event_mask[:, None],
                        pending.obs,
                        teacher_buffer.obs[teacher_batch_size],
                    )
                    teacher_buffer = TeacherTransition(
                        done=teacher_buffer.done.at[safe_write_idx].set(
                            jnp.where(
                                teacher_event_mask,
                                jnp.ones_like(teacher_reward),
                                teacher_buffer.done[teacher_batch_size],
                            )
                        ),
                        action=teacher_buffer.action.at[safe_write_idx].set(
                            jnp.where(
                                teacher_event_mask,
                                pending.action,
                                teacher_buffer.action[teacher_batch_size],
                            )
                        ),
                        value=teacher_buffer.value.at[safe_write_idx].set(
                            jnp.where(
                                teacher_event_mask,
                                pending.value,
                                teacher_buffer.value[teacher_batch_size],
                            )
                        ),
                        reward=teacher_buffer.reward.at[safe_write_idx].set(
                            jnp.where(
                                teacher_event_mask,
                                teacher_reward,
                                teacher_buffer.reward[teacher_batch_size],
                            )
                        ),
                        log_prob=teacher_buffer.log_prob.at[safe_write_idx].set(
                            jnp.where(
                                teacher_event_mask,
                                pending.log_prob,
                                teacher_buffer.log_prob[teacher_batch_size],
                            )
                        ),
                        obs=teacher_buffer.obs.at[safe_write_idx].set(event_obs),
                    )
                    teacher_buffer_count = jnp.minimum(
                        teacher_buffer_count + valid_event.astype(jnp.int32).sum(),
                        jnp.asarray(teacher_batch_size, dtype=jnp.int32),
                    )
                    # Adopt the freshly sampled proposal for envs where the
                    # current proposal ended.
                    refresh_col = teacher_event_mask[:, None]
                    pending = PendingTeacher(
                        obs=jnp.where(refresh_col, new_t_obs, pending.obs),
                        action=jnp.where(teacher_event_mask, new_t_action, pending.action),
                        value=jnp.where(teacher_event_mask, new_t_value, pending.value),
                        log_prob=jnp.where(
                            teacher_event_mask, new_t_log_prob, pending.log_prob
                        ),
                        competence_before=jnp.where(
                            teacher_event_mask,
                            comp_before_new,
                            pending.competence_before,
                        ),
                        goal=jnp.where(refresh_col, new_goals, pending.goal),
                    )

                info = dict(info)
                info["task_reward_term"] = task_reward_term
                info["goal_reward_term"] = goal_reward_term
                info["goal_reward_unweighted"] = goal_reward_unweighted
                info["shaped_reward"] = reward
                # Raw (unnormalized) penalty and base env reward, kept so the
                # batch_minmax mode can rebuild the shaped reward post-rollout.
                info["goal_penalty_raw"] = raw_goal_penalty
                info["base_reward"] = task_reward_term
                info["returned_goal_reward_episode_returns"] = returned_goal_ep_returns
                info["returned_goal_reward_episode"] = done_mask
                info["teacher_reward"] = teacher_reward
                # Competence-LP diagnostics (meaningful only on boundary steps).
                info["teacher_competence_after"] = comp_after
                info["teacher_competence_before"] = comp_before_old
                transition = Transition(
                    done, action, value, reward, log_prob, policy_obs, info
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    goal_batch,
                    goal_ep_returns,
                    episode_counts,
                    avg_success_rate,
                    reached_any_in_episode,
                    prev_episode_success,
                    prev_goal_ep_return,
                    rng,
                    teacher_train_state,
                    teacher_buffer,
                    teacher_buffer_count,
                    pending,
                    teacher_rng,
                    goal_penalty_norm_state,
                    goal_viz_buffer,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                goal_batch,
                goal_ep_returns,
                episode_counts,
                avg_success_rate,
                reached_any_in_episode,
                prev_episode_success,
                prev_goal_ep_return,
                rng,
                teacher_train_state,
                teacher_buffer,
                teacher_buffer_count,
                pending,
                teacher_rng,
                goal_penalty_norm_state,
                goal_viz_buffer,
            ) = runner_state
            last_policy_obs = _concat_goal(last_obs, goal_batch)
            _, last_val = network.apply(train_state.params, last_policy_obs)

            # BATCH MIN-MAX NORMALIZATION (applied after rollout collection,
            # before GAE). Rebuilds the student shaped reward from the base env
            # reward plus the weight-scaled, min-max normalized goal penalty
            # over the whole just-collected rollout batch.
            if goal_conditioned and goal_penalty_norm_type == "batch_minmax":
                gp = traj_batch.info["goal_penalty_raw"]
                gp_min = jnp.min(gp)
                gp_max = jnp.max(gp)
                gp_norm = (gp - gp_min) / (gp_max - gp_min + goal_penalty_norm_eps)
                new_goal_reward_term = current_goal_reward_weight * gp_norm
                new_reward = traj_batch.info["base_reward"] + new_goal_reward_term
                new_info = dict(traj_batch.info)
                new_info["goal_reward_term"] = new_goal_reward_term
                new_info["goal_reward_unweighted"] = gp_norm
                new_info["shaped_reward"] = new_reward
                traj_batch = traj_batch._replace(reward=new_reward, info=new_info)

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

            def _calculate_gae_stream(rewards, values, dones, last_value):
                """GAE for a single reward/value stream (separate-critic mode)."""
                def _get_advantages(gae_and_next_value, xs):
                    gae, next_value = gae_and_next_value
                    done, value, reward = xs
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_value), last_value),
                    (dones, values, rewards),
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + values

            if use_separate_student_value_functions:
                # Two value heads stacked as (..., 2) = (task, goal). The goal
                # critic regresses to the UNWEIGHTED goal reward so that
                # GOAL_REWARD_WEIGHT only trades off optimization in the actor
                # advantage below, not in the critic targets.
                task_values = traj_batch.value[..., 0]
                goal_values = traj_batch.value[..., 1]
                base_reward = traj_batch.info["base_reward"]
                goal_reward_unweighted = traj_batch.info["goal_reward_unweighted"]
                adv_task, target_task = _calculate_gae_stream(
                    base_reward, task_values, traj_batch.done, last_val[..., 0]
                )
                adv_goal, target_goal = _calculate_gae_stream(
                    goal_reward_unweighted,
                    goal_values,
                    traj_batch.done,
                    last_val[..., 1],
                )
                advantages = adv_task + current_goal_reward_weight * adv_goal
                targets = jnp.stack([target_task, target_goal], axis=-1)
            else:
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
                        if use_separate_student_value_functions:
                            # value/targets are (..., 2) = (task, goal). The total
                            # critic loss is the UNWEIGHTED sum of both heads.
                            per_head = 0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            )
                            value_loss_task = per_head[..., 0].mean()
                            value_loss_goal = per_head[..., 1].mean()
                            value_loss = value_loss_task + value_loss_goal
                        else:
                            value_loss = (
                                0.5
                                * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )
                            value_loss_task = value_loss
                            value_loss_goal = jnp.zeros_like(value_loss)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        if normalize_student_advantage:
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
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            value_loss_task,
                            value_loss_goal,
                        )

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
            # import pdb; pdb.set_trace()
            total_loss = loss_info[0].mean()
            value_loss = loss_info[1][0].mean()
            actor_loss = loss_info[1][1].mean()
            entropy = loss_info[1][2].mean()
            value_loss_task = loss_info[1][3].mean()
            value_loss_goal = loss_info[1][4].mean()
            current_lr = (
                linear_schedule(
                    update_idx
                    * config["NUM_MINIBATCHES"]
                    * config["UPDATE_EPOCHS"]
                )
                if config["ANNEAL_LR"]
                else config["LR"]
            )

            # ----- TEACHER PPO UPDATE -----
            # Fires (via lax.cond) once TEACHER_BATCH_SIZE proposal-end events
            # have been collected; buffer/count resets afterwards.
            def _teacher_calculate_gae(traj, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + teacher_gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + teacher_gamma * teacher_gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj.value

            def _run_teacher_update(operands):
                t_train_state, t_buffer, t_count, t_rng, last_val = operands
                # Drop the trailing padding row and use the collected batch.
                traj = jax.tree_util.tree_map(
                    lambda x: x[:teacher_batch_size], t_buffer
                )
                advantages, targets = _teacher_calculate_gae(traj, last_val)

                def _t_update_epoch(update_state, unused):
                    def _t_update_minibatch(t_state, batch_info):
                        traj_b, gae_b, tgt_b = batch_info

                        def _t_loss_fn(params, traj_b, gae, targets):
                            # Recompute with the current student params so the
                            # learnable probing states keep receiving gradients.
                            pi, value = _teacher_apply(
                                params, traj_b.obs, train_state.params
                            )
                            prob_states = params["params"]["probing_states"]
                            # jax.debug.print("probe states, params: {prob_states}", prob_states=prob_states[0])
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
                            if normalize_teacher_advantage:
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

                    t_state, traj, advantages, targets, rng = update_state
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, teacher_batch_size)
                    batch = (traj, advantages, targets)
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((teacher_batch_size,) + x.shape[1:]),
                        batch,
                    )
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x, [teacher_num_minibatches, -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )
                    t_state, total_loss = jax.lax.scan(
                        _t_update_minibatch, t_state, minibatches
                    )
                    update_state = (t_state, traj, advantages, targets, rng)
                    return update_state, total_loss

                t_rng, epoch_rng = jax.random.split(t_rng)
                update_state = (t_train_state, traj, advantages, targets, epoch_rng)
                update_state, t_loss_info = jax.lax.scan(
                    _t_update_epoch, update_state, None, teacher_update_epochs
                )
                t_train_state = update_state[0]
                new_buffer = jax.tree_util.tree_map(jnp.zeros_like, t_buffer)
                new_count = jnp.zeros_like(t_count)
                metrics = (
                    t_loss_info[0].mean(),
                    t_loss_info[1][0].mean(),
                    t_loss_info[1][1].mean(),
                    t_loss_info[1][2].mean(),
                    jnp.array(1.0, dtype=jnp.float32),
                )
                return t_train_state, new_buffer, new_count, t_rng, metrics

            def _skip_teacher_update(operands):
                t_train_state, t_buffer, t_count, t_rng, last_val = operands
                z = jnp.array(0.0, dtype=jnp.float32)
                return t_train_state, t_buffer, t_count, t_rng, (z, z, z, z, z)

            teacher_should_update = teacher_buffer_count >= teacher_batch_size
            (
                teacher_train_state,
                teacher_buffer,
                teacher_buffer_count,
                teacher_rng,
                teacher_metrics,
            ) = jax.lax.cond(
                teacher_should_update,
                _run_teacher_update,
                _skip_teacher_update,
                (
                    teacher_train_state,
                    teacher_buffer,
                    teacher_buffer_count,
                    teacher_rng,
                    jnp.array(0.0, dtype=jnp.float32),
                ),
            )
            (
                teacher_total_loss,
                teacher_value_loss,
                teacher_actor_loss,
                teacher_entropy,
                teacher_did_update,
            ) = teacher_metrics

            # Effective teacher LR (mirrors student annealing). ``train_state.step``
            # is the optax gradient-step count, i.e. the schedule's ``count`` arg.
            teacher_current_lr = (
                teacher_linear_schedule(teacher_train_state.step)
                if config["ANNEAL_LR"]
                else teacher_lr
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
                        update_idx,
                        info,
                        total_loss,
                        value_loss,
                        actor_loss,
                        entropy,
                        current_lr,
                        teacher_current_lr,
                        current_goal_reward_weight,
                        teacher_total_loss,
                        teacher_value_loss,
                        teacher_actor_loss,
                        teacher_entropy,
                        teacher_did_update,
                        value_loss_task,
                        value_loss_goal,
                    ) = args
                    # Throttle logging to avoid very frequent wandb writes.
                    if (int(update_idx) + 1) % wandb_log_every_updates != 0:
                        return
                    # Keep all metrics on a consistent global step for stable wandb curves.
                    step = int(info["timestep"].max() * config["NUM_ENVS"])
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    # Only log when a student episode has completed.
                    if len(return_values) == 0:
                        return
                    goal_return_values = info["returned_goal_reward_episode_returns"][
                        info["returned_goal_reward_episode"]
                    ]
                    teacher_reward_values = info["teacher_reward"][
                        info["returned_goal_reward_episode"]
                    ]
                    payload = {
                        "episodic_return": float(return_values.mean()),
                        "total_loss": float(total_loss),
                        "value_loss": float(value_loss),
                        "actor_loss": float(actor_loss),
                        "entropy": float(entropy),
                        "learning_rate": float(current_lr),
                        "teacher/learning_rate": float(teacher_current_lr),
                        "teacher/did_update": float(teacher_did_update),
                        "task_reward_weight": float(task_reward_weight),
                        "goal_reward_weight": float(current_goal_reward_weight),
                        "task_reward_term_mean": float(info["task_reward_term"].mean()),
                        "goal_reward_term_mean": float(info["goal_reward_term"].mean()),
                        "shaped_reward_mean": float(info["shaped_reward"].mean()),
                        "goal_penalty_raw_mean": float(info["goal_penalty_raw"].mean()),
                        "goal_penalty_raw_std": float(info["goal_penalty_raw"].std()),
                        "goal_penalty_raw_min": float(info["goal_penalty_raw"].min()),
                        "goal_penalty_raw_max": float(info["goal_penalty_raw"].max()),
                    }
                    if use_separate_student_value_functions:
                        payload["value_loss_task"] = float(value_loss_task)
                        payload["value_loss_goal"] = float(value_loss_goal)
                    if len(goal_return_values) > 0:
                        payload["episodic_goal_shaping_return"] = float(
                            goal_return_values.mean()
                        )
                    if len(teacher_reward_values) > 0:
                        payload["teacher_learning_progress"] = float(
                            teacher_reward_values.mean()
                        )
                    if "teacher_competence_after" in info:
                        comp_after_vals = info["teacher_competence_after"][
                            info["returned_goal_reward_episode"]
                        ]
                        comp_before_vals = info["teacher_competence_before"][
                            info["returned_goal_reward_episode"]
                        ]
                        if len(comp_after_vals) > 0:
                            payload["teacher/competence_after"] = float(
                                comp_after_vals.mean()
                            )
                            payload["teacher/competence_before"] = float(
                                comp_before_vals.mean()
                            )
                    # Teacher losses are only meaningful on steps where it updated.
                    if float(teacher_did_update) > 0.0:
                        payload.update(
                            {
                                "teacher/total_loss": float(teacher_total_loss),
                                "teacher/value_loss": float(teacher_value_loss),
                                "teacher/actor_loss": float(teacher_actor_loss),
                                "teacher/entropy": float(teacher_entropy),
                            }
                        )
                    wandb.log(payload, step=step)

                jax.debug.callback(
                    wandb_callback,
                    (
                        update_idx,
                        metric,
                        total_loss,
                        value_loss,
                        actor_loss,
                        entropy,
                        current_lr,
                        teacher_current_lr,
                        current_goal_reward_weight,
                        teacher_total_loss,
                        teacher_value_loss,
                        teacher_actor_loss,
                        teacher_entropy,
                        teacher_did_update,
                        value_loss_task,
                        value_loss_goal,
                    ),
                )

                if goal_conditioned:

                    def goal_viz_callback(args):
                        (
                            update_idx,
                            info,
                            buf,
                            start_xy,
                            env_state,
                            t_params,
                            s_params,
                            ref_obs,
                            avg_sr,
                        ) = args
                        # import pdb; pdb.set_trace()
                        if (
                            int(update_idx) + 1
                        ) % teacher_goal_viz_log_every_updates != 0:
                            return
                        step = int(info["timestep"].max() * config["NUM_ENVS"])
                        log_teacher_goal_viz_plot(buf, start_xy, env_state, step)
                        log_teacher_softmax_viz_plot(
                            t_params, s_params, ref_obs, avg_sr, env_state, step
                        )

                    jax.debug.callback(
                        goal_viz_callback,
                        (
                            update_idx,
                            metric,
                            goal_viz_buffer,
                            agent_start_xy,
                            env_state,
                            teacher_train_state.params,
                            train_state.params,
                            obsv[0],
                            avg_success_rate.mean(),
                        ),
                    )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                goal_batch,
                goal_ep_returns,
                episode_counts,
                avg_success_rate,
                reached_any_in_episode,
                prev_episode_success,
                prev_goal_ep_return,
                rng,
                teacher_train_state,
                teacher_buffer,
                teacher_buffer_count,
                pending,
                teacher_rng,
                goal_penalty_norm_state,
                goal_viz_buffer,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        goal_ep_returns = jnp.zeros((config["NUM_ENVS"],), dtype=obsv.dtype)
        episode_counts = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
        reached_any_in_episode = jnp.zeros((config["NUM_ENVS"],), dtype=bool)
        prev_episode_success = jnp.zeros((config["NUM_ENVS"],), dtype=obsv.dtype)
        prev_goal_ep_return = jnp.zeros((config["NUM_ENVS"],), dtype=obsv.dtype)
        runner_state = (
            train_state,
            env_state,
            obsv,
            goal_batch,
            goal_ep_returns,
            episode_counts,
            avg_success_rate,
            reached_any_in_episode,
            prev_episode_success,
            prev_goal_ep_return,
            _rng,
            teacher_train_state,
            teacher_buffer,
            teacher_buffer_count,
            pending,
            teacher_rng,
            goal_penalty_norm_state,
            goal_viz_buffer,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {
            "runner_state": runner_state,
            "metrics": metric,
            "teacher_goal_viz": runner_state[-1],
            "agent_start_xy": agent_start_xy,
            "agent_start_obs": obsv[0],
        }

    return (
        train,
        render_eval_episode,
        log_teacher_goal_viz_plot,
        log_teacher_softmax_viz_plot,
    )


def main():
    config_obj = parse_config_from_cli()
    config = config_obj.to_dict()

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["PPO", "BRAX", config["ENV_NAME"], f"jax_{jax.__version__}"],
        name=f'purejaxrl_ppo_brax_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    (
        train_fn,
        render_eval_episode,
        log_teacher_goal_viz_plot,
        log_teacher_softmax_viz_plot,
    ) = make_train(config)
    print(
        f"Number of student updates: {config['NUM_UPDATES']}, "
        f"teacher: {config['TEACHER_NUM_UPDATES']}, "
        f"teacher/student ratio: {config['TEACHER_STUDENT_UPDATE_RATIO']:.6f}"
    )
    train_jit = jax.jit(train_fn)
    train_output = train_jit(rng)
    jax.block_until_ready(train_output["runner_state"][6])
    runner_state = train_output["runner_state"]
    final_train_state = runner_state[0]
    final_env_state = runner_state[1]
    final_rng = runner_state[10]
    final_teacher_train_state = runner_state[11]
    render_eval_episode(
        final_train_state.params,
        final_teacher_train_state.params,
        final_rng,
        final_env_state,
    )
    if config.get("GOAL_CONDITIONED", False):
        log_teacher_goal_viz_plot(
            train_output["teacher_goal_viz"],
            train_output["agent_start_xy"],
            final_env_state,
            step=int(config["TOTAL_TIMESTEPS"]),
        )
        log_teacher_softmax_viz_plot(
            final_teacher_train_state.params,
            final_train_state.params,
            train_output["agent_start_obs"],
            runner_state[6].mean(),
            final_env_state,
            step=int(config["TOTAL_TIMESTEPS"]),
        )


if __name__ == "__main__":
    main()
