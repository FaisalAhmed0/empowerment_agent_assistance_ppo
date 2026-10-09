"""Load a teacher model checkpoint saved under an experiment directory.

Expected layout (created when training with --SAVE_MODEL):
  EXP_DIR/
    config.json
    checkpoints/
      teacher/       # Orbax StandardCheckpointer of teacher TrainState
      teacher_ema/   # EMA teacher params (same TrainState layout)
      teacher_avg/   # Uniform-average teacher params (same TrainState layout);
                     # written at every max-log-sum save, overwritten at end of run
      teacher_max_log_sum_competence/  # teacher at max log(sum(competence))
      student/       # final student TrainState
      student_max_log_sum_competence/  # student paired with max-log-sum teacher
      obs_norm_stats.npz  # written with max-log-sum saves and at end of run

Example:
  python purejaxrl/load_teacher_model.py --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id>
  python purejaxrl/load_teacher_model.py --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> --use_ema
  python purejaxrl/load_teacher_model.py --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> --use_avg
  python purejaxrl/load_teacher_model.py --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> --use_max_log_sum_competence
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState

try:
    from purejaxrl.envs.factory import make_custom_env
except ImportError:
    from envs.factory import make_custom_env

try:
    from envs import ant_maze
except ImportError:
    from purejaxrl.envs import ant_maze

from wrappers import BraxGymnaxWrapper

try:
    from purejaxrl.ppo_lp_teacher import (
        ActorCritic,
        TeacherActorCritic,
        load_checkpoint,
    )
except ImportError:
    from ppo_lp_teacher import (
        ActorCritic,
        TeacherActorCritic,
        load_checkpoint,
    )

try:
    from purejaxrl.conditional_teacher_models import ConditionalTeacherActorCritic
except ImportError:
    from conditional_teacher_models import ConditionalTeacherActorCritic


@dataclass
class LoadTeacherConfig:
    exp_dir: str 
    checkpoint_dir: str = "checkpoints"
    seed: int = 30
    use_ema: bool = False
    use_avg: bool = False
    use_max_log_sum_competence: bool = False


def _teacher_checkpoint_subdir(
    use_ema: bool = False,
    use_avg: bool = False,
    use_max_log_sum_competence: bool = False,
) -> str:
    """Select teacher Orbax subdir.

    Precedence: max-log-sum competence > avg > EMA > teacher.
    """
    if use_max_log_sum_competence:
        return "teacher_max_log_sum_competence"
    if use_avg:
        return "teacher_avg"
    if use_ema:
        return "teacher_ema"
    return "teacher"


def _load_config(exp_dir: str) -> dict[str, Any]:
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing config.json in experiment directory: {config_path}"
        )
    with open(config_path, "r") as f:
        return json.load(f)


def load_experiment_config(exp_dir: str) -> dict[str, Any]:
    """Public wrapper for loading ``config.json`` from an experiment directory."""
    return _load_config(exp_dir)


def load_obs_norm_stats(
    exp_dir: str,
    checkpoint_dir: str = "checkpoints",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load observation normalization stats saved alongside checkpoints.

    Returns ``(mean, var)`` as numpy arrays, or ``None`` when the file is
    missing (e.g. older runs that predate this artifact).
    """
    stats_path = os.path.join(exp_dir, checkpoint_dir, "obs_norm_stats.npz")
    if not os.path.exists(stats_path):
        return None
    with np.load(stats_path) as data:
        mean = np.asarray(data["mean"]).copy()
        var = np.asarray(data["var"]).copy()
    return mean, var


def teacher_goal_set_from_config(
    config: dict[str, Any],
) -> tuple[np.ndarray, int, np.ndarray, list]:
    """Rebuild the teacher goal grid and competence goals used in training.

    Mirrors the ant branch of ``make_teacher_goal_set`` in the training
    scripts. Returns ``(goal_grid, num_points, competence_goals, maze_layout)``
    where ``maze_layout`` is the (scaled) layout containing the reset cell.
    """
    env_name = config["ENV_NAME"]
    if "ant" not in env_name:
        raise ValueError(f"Only ant maze envs are supported, got {env_name!r}")
    if "u_maze" in env_name:
        maze_layout = ant_maze.U_MAZE
        all_goals_layout = ant_maze.U_MAZE_ALL_STATES
        custom_goal = np.array([12.0, 8.0], dtype=np.float32)
    elif "big_maze" in env_name:
        maze_layout = ant_maze.BIG_MAZE
        all_goals_layout = ant_maze.BIG_MAZE_ALL_GOALS
        custom_goal = np.array([12.0, 8.0], dtype=np.float32)
    elif "hardest_maze" in env_name:
        maze_layout = ant_maze.HARDEST_MAZE
        all_goals_layout = ant_maze.HARDEST_MAZE_ALL_GOALS
        custom_goal = np.array([28.0, 40.0], dtype=np.float32)
    else:
        raise ValueError(f"Unknown maze layout: {env_name}")

    maze_scale_factor = int(config.get("MAZE_SCALE_FACTOR", 1))
    maze_layout = ant_maze.scale_maze_layout(maze_layout, maze_scale_factor)
    all_goals_layout = ant_maze.scale_maze_layout(all_goals_layout, maze_scale_factor)
    custom_goal = custom_goal * maze_scale_factor

    num_points = int(config["TEACHER_NUM_GOAL_POINTS"])
    min_x, max_x, min_y, max_y = ant_maze.get_maze_xy_bounds(maze_layout)
    xs = np.linspace(min_x, max_x, num_points)
    ys = np.linspace(min_y, max_y, num_points)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    goal_grid = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
    replace_idx = int(np.argmin(np.sum((goal_grid - custom_goal) ** 2, axis=-1)))
    goal_grid[replace_idx] = custom_goal

    competence_goals = np.asarray(
        ant_maze.all_possible_goals(all_goals_layout), dtype=np.float32
    )
    return goal_grid, num_points, competence_goals, maze_layout


def _num_competence(config: dict[str, Any]) -> int:
    return int(teacher_goal_set_from_config(config)[2].shape[0])


def _make_env(config: dict[str, Any]):
    env_kwargs = config.get("ENV_KWARGS", {}) or {}
    custom_env = make_custom_env(
        env_name=config["ENV_NAME"],
        backend=config.get("ENV_BACKEND"),
        env_kwargs=env_kwargs,
    )
    if custom_env is None:
        from brax import envs as brax_envs

        custom_env = brax_envs.get_environment(
            env_name=config["ENV_NAME"],
            backend=config.get("ENV_BACKEND", "positional"),
        )
    env = BraxGymnaxWrapper(
        env=custom_env,
        episode_length=config.get("EPISODE_LENGTH", 1000),
        action_repeat=config.get("ACTION_REPEAT", 1),
    )
    return env, None


def _teacher_obs_dim(config: dict[str, Any], base_obs_dim: int, action_dim: int) -> int:
    only_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    competence_and_env_goal = bool(
        config.get("TEACHER_CONDITION_ON_COMPETENCE_AND_ENV_GOAL", False)
    )
    condition_teacher_on_competence = (
        bool(config.get("CONDITION_TEACHER_ON_COMPETENCE", True))
        or only_competence
        or competence_and_env_goal
    )
    condition_teacher_on_action = bool(
        config.get("CONDITION_TEACHER_ON_ACTION", True)
    )
    teacher_obs_goal_only = (
        bool(config.get("TEACHER_OBS_GOAL_ONLY", False)) or competence_and_env_goal
    )
    num_competence = _num_competence(config)
    if only_competence:
        return num_competence
    state_obs_dim = 2 if teacher_obs_goal_only else base_obs_dim
    return (
        state_obs_dim
        + (num_competence if condition_teacher_on_competence else 0)
        + (action_dim if condition_teacher_on_action else 0)
    )


def _build_student_tx(config: dict[str, Any]):
    """Build an Optax transform matching the student TrainState checkpoint."""
    num_updates = max(
        1,
        int(config["TOTAL_TIMESTEPS"])
        // int(config["NUM_STEPS"])
        // int(config["NUM_ENVS"]),
    )
    num_minibatches = int(config.get("NUM_MINIBATCHES", 4))
    update_epochs = int(config.get("UPDATE_EPOCHS", 4))
    agent_total_opt_steps = num_updates * num_minibatches * update_epochs

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (num_minibatches * update_epochs)) / num_updates
        )
        return config["LR"] * frac

    if config.get("USE_OPTAX_LR_SCHEDULE", False):
        lr_schedule = optax.linear_schedule(
            init_value=config["LR"],
            end_value=0.0,
            transition_steps=max(int(agent_total_opt_steps), 1),
        )
    else:
        lr_schedule = linear_schedule

    agent_lr = lr_schedule if config.get("ANNEAL_LR", True) else config["LR"]
    if config.get("USE_ADAMW", False):
        agent_opt = optax.adamw(
            learning_rate=agent_lr,
            eps=1e-5,
            weight_decay=config.get("WEIGHT_DECAY", 1e-4),
        )
    else:
        agent_opt = optax.adam(learning_rate=agent_lr, eps=1e-5)
    return optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
        agent_opt,
    )


def _student_checkpoint_subdir(use_max_log_sum_competence: bool = False) -> str:
    """Select student Orbax subdir."""
    if use_max_log_sum_competence:
        return "student_max_log_sum_competence"
    return "student"


def load_student_model(
    exp_dir: str,
    checkpoint_dir: str = "checkpoints",
    seed: int = 0,
    use_max_log_sum_competence: bool = False,
) -> TrainState:
    """Rebuild ActorCritic and restore its TrainState from EXP_DIR.

    If ``use_max_log_sum_competence`` is set, loads
    ``student_max_log_sum_competence`` (paired with the max-log-sum teacher).
    """
    config = _load_config(exp_dir)
    student_subdir = _student_checkpoint_subdir(
        use_max_log_sum_competence=use_max_log_sum_competence,
    )
    student_ckpt_dir = os.path.join(exp_dir, checkpoint_dir, student_subdir)
    if not os.path.isdir(student_ckpt_dir):
        raise FileNotFoundError(
            f"Student checkpoint directory not found: {student_ckpt_dir}"
        )

    env, env_params = _make_env(config)
    action_dim = int(env.action_space(env_params).shape[0])
    condition_on_goal = bool(config.get("CONDITION_ON_GOAL", False))
    goal_dim = 2

    network = ActorCritic(
        action_dim,
        activation=config.get("ACTIVATION", "tanh"),
        hidden_dim=int(config.get("HIDDEN_DIM", 256)),
    )
    rng = jax.random.PRNGKey(seed)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    if condition_on_goal:
        init_x = jnp.concatenate([init_x, jnp.zeros((goal_dim,))], axis=-1)
    init_params = network.init(rng, init_x)
    template_train_state = TrainState.create(
        apply_fn=network.apply,
        params=init_params,
        tx=_build_student_tx(config),
    )
    return load_checkpoint(template_train_state, student_ckpt_dir)


def _build_teacher_tx(config: dict[str, Any]):
    teacher_num_minibatches = int(config.get("TEACHER_NUM_MINIBATCHES", 8))
    teacher_update_epochs = int(config.get("TEACHER_UPDATE_EPOCHS", 4))
    teacher_rollout_buffer_size = int(config.get("TEACHER_ROLLOUT_BUFFER_SIZE", 1))
    approx_total_episode_completions = (
        int(config["TOTAL_TIMESTEPS"]) // int(config.get("EPISODE_LENGTH", 1000))
    )
    teacher_num_updates = max(
        1, approx_total_episode_completions // teacher_rollout_buffer_size
    )
    teacher_total_opt_steps = (
        teacher_num_updates * teacher_num_minibatches * teacher_update_epochs
    )

    def teacher_linear_schedule(count):
        frac = (
            1.0
            - (count // (teacher_num_minibatches * teacher_update_epochs))
            / teacher_num_updates
        )
        frac = jnp.maximum(frac, 0.000000001)
        return config["TEACHER_LR"] * frac

    if config.get("USE_OPTAX_LR_SCHEDULE", False):
        teacher_lr_schedule = optax.linear_schedule(
            init_value=config["TEACHER_LR"],
            end_value=0.0,
            transition_steps=max(int(teacher_total_opt_steps), 1),
        )
    else:
        teacher_lr_schedule = teacher_linear_schedule

    teacher_lr = (
        teacher_lr_schedule
        if config.get("ANNEAL_LR", True)
        else config["TEACHER_LR"]
    )
    if config.get("TEACHER_USE_ADAMW", False):
        teacher_opt = optax.adamw(
            learning_rate=teacher_lr,
            eps=1e-5,
            weight_decay=config.get("TEACHER_WEIGHT_DECAY", 1e-4),
        )
    else:
        teacher_opt = optax.adam(learning_rate=teacher_lr, eps=1e-5)
    return optax.chain(
        optax.clip_by_global_norm(config.get("TEACHER_MAX_GRAD_NORM", 1.0)),
        teacher_opt,
    )


def load_teacher_model(
    exp_dir: str,
    checkpoint_dir: str = "checkpoints",
    seed: int = 0,
    use_ema: bool = False,
    use_avg: bool = False,
    use_max_log_sum_competence: bool = False,
) -> TrainState:
    """Rebuild teacher network and restore its TrainState from EXP_DIR.

    Uses ``ConditionalTeacherActorCritic`` when ``USE_CONDITIONAL_TEACHER`` is
    set in the experiment config; otherwise ``TeacherActorCritic``.

    If ``use_max_log_sum_competence`` is set, loads
    ``teacher_max_log_sum_competence`` (takes precedence over ``use_avg`` /
    ``use_ema``). Otherwise ``use_avg`` selects ``teacher_avg``, then
    ``use_ema`` selects ``teacher_ema``.
    """
    config = _load_config(exp_dir)
    teacher_subdir = _teacher_checkpoint_subdir(
        use_ema=use_ema,
        use_avg=use_avg,
        use_max_log_sum_competence=use_max_log_sum_competence,
    )
    teacher_ckpt_dir = os.path.join(exp_dir, checkpoint_dir, teacher_subdir)
    if not os.path.isdir(teacher_ckpt_dir):
        raise FileNotFoundError(
            f"Teacher checkpoint directory not found: {teacher_ckpt_dir}"
        )

    env, env_params = _make_env(config)
    base_obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).shape[0])
    teacher_num_goal_points = int(config["TEACHER_NUM_GOAL_POINTS"])
    only_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    competence_and_env_goal = bool(
        config.get("TEACHER_CONDITION_ON_COMPETENCE_AND_ENV_GOAL", False)
    )
    condition_teacher_on_competence = (
        bool(config.get("CONDITION_TEACHER_ON_COMPETENCE", True))
        or only_competence
        or competence_and_env_goal
    )
    condition_teacher_on_action = bool(
        config.get("CONDITION_TEACHER_ON_ACTION", True)
    )
    teacher_obs_goal_only = (
        bool(config.get("TEACHER_OBS_GOAL_ONLY", False)) or competence_and_env_goal
    )
    num_competence = _num_competence(config)
    teacher_obs_dim = _teacher_obs_dim(config, base_obs_dim, action_dim)

    if only_competence:
        teacher_net_obs_dim = 0
        teacher_net_competence_dim = num_competence
        teacher_net_action_dim = 0
    else:
        teacher_net_obs_dim = 2 if teacher_obs_goal_only else base_obs_dim
        teacher_net_competence_dim = (
            num_competence if condition_teacher_on_competence else 0
        )
        teacher_net_action_dim = (
            action_dim if condition_teacher_on_action else 0
        )

    if config.get("USE_CONDITIONAL_TEACHER", False):
        conditional_concatenate = bool(
            config.get("CONDITIONAL_TEACHER_CONCATENATE", False)
        )
        teacher_network = ConditionalTeacherActorCritic(
            num_actions=teacher_num_goal_points * teacher_num_goal_points,
            obs_dim=teacher_net_obs_dim,
            competence_dim=teacher_net_competence_dim,
            student_action_dim=teacher_net_action_dim,
            activation=config.get("TEACHER_ACTIVATION", "tanh"),
            hidden_dim=int(config.get("TEACHER_HIDDEN_DIM", 256)),
            add=not conditional_concatenate,
            concatenate=conditional_concatenate,
        )
    else:
        teacher_kwargs = dict(
            num_actions=teacher_num_goal_points * teacher_num_goal_points,
            obs_dim=teacher_net_obs_dim,
            competence_dim=teacher_net_competence_dim,
            activation=config.get("TEACHER_ACTIVATION", "tanh"),
            hidden_dim=int(config.get("TEACHER_HIDDEN_DIM", 256)),
            use_encoders=bool(config.get("TEACHER_USE_ENCODERS", True)),
        )
        # ppo_lp_teacher's TeacherActorCritic never conditions on the action.
        if "student_action_dim" in TeacherActorCritic.__dataclass_fields__:
            teacher_kwargs["student_action_dim"] = teacher_net_action_dim
        elif teacher_net_action_dim:
            raise ValueError(
                "Experiment conditions the teacher on the action, but "
                "TeacherActorCritic has no student_action_dim field"
            )
        teacher_network = TeacherActorCritic(**teacher_kwargs)

    rng = jax.random.PRNGKey(seed)
    teacher_init_params = teacher_network.init(
        rng, jnp.zeros((teacher_obs_dim,))
    )
    teacher_tx = _build_teacher_tx(config)
    template_train_state = TrainState.create(
        apply_fn=teacher_network.apply,
        params=teacher_init_params,
        tx=teacher_tx,
    )
    return load_checkpoint(template_train_state, teacher_ckpt_dir)


def main():
    args = tyro.cli(LoadTeacherConfig)
    teacher_train_state = load_teacher_model(
        exp_dir=args.exp_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        use_ema=args.use_ema,
        use_avg=args.use_avg,
        use_max_log_sum_competence=args.use_max_log_sum_competence,
    )
    num_leaves = sum(
        1 for _ in jax.tree_util.tree_leaves(teacher_train_state.params)
    )
    teacher_subdir = _teacher_checkpoint_subdir(
        use_ema=args.use_ema,
        use_avg=args.use_avg,
        use_max_log_sum_competence=args.use_max_log_sum_competence,
    )
    ckpt_path = os.path.join(args.exp_dir, args.checkpoint_dir, teacher_subdir)
    print(f"Loaded teacher TrainState from {ckpt_path}")
    print(f"Teacher params leaves: {num_leaves}")
    print(f"TrainState step: {int(teacher_train_state.step)}")


if __name__ == "__main__":
    main()