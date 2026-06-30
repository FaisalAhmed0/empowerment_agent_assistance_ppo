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
    ENV_NAME: str = "ant_u_maze"
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
    EVAL_RENDER_STEPS: int = 300
    EVAL_RENDER_MAX_FRAMES: int = 100
    EVAL_RENDER_HEIGHT: int = 360
    EVAL_RENDER_LOG_WANDB_HTML: bool = True
    COMMENT: str = ""
    ADD_GOAL_REWARD: bool = True
    CONDITION_ON_GOAL: bool = True
    GOAL_REACH_EPSILON: float = 0.5
    TEACHER_GOAL_X_MIN: float = 0.0
    TEACHER_GOAL_X_MAX: float = 12.0
    TEACHER_GOAL_Y_MIN: float = 0.0
    TEACHER_GOAL_Y_MAX: float = 4.0
    TEACHER_NUM_GOAL_POINTS: int = 30
    TEACHER_HIDDEN_DIM: int = 256
    SAVE_MODEL: bool = False
    checkpoint_dir: str = "checkpoints"
    GOAL_REWARD_COEF: float = 1.0
    NUM_EVAL_ENVS: int = 32
    CONDITION_TEACHER_ON_COMPETENCE: bool = True


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
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray




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
    all_goals = all_possible_goals()
    num_competence = int(all_goals.shape[0])
    condition_teacher_on_competence = config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
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

    def _eval_render_action(params, obs, obs_mean, obs_var):
        norm_obs = _normalize_eval_obs(obs, obs_mean, obs_var)
        pi, _ = network.apply(params, norm_obs)
        return jnp.clip(pi.mean(), action_low, action_high)

    def _render_rollout_impl(params, rng, obs_mean, obs_var):
        state = base_env.reset(rng)

        def step_fn(carry, _):
            state, _ = carry
            action = _eval_render_action(params, state.obs, obs_mean, obs_var)

            def repeat_step(s, __):
                return base_env.step(s, action), None

            state, _ = jax.lax.scan(
                repeat_step, state, None, length=render_action_repeat
            )
            return (state, None), state.pipeline_state

        _, pipeline_states = jax.lax.scan(
            step_fn, (state, None), None, length=render_sim_steps
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

    def render_eval_episode(params, rng, final_env_state):
        """Runs one eval rollout and logs rendered HTML to wandb."""
        if config.get("WANDB_MODE", "disabled") != "online":
            return
        try:
            obs_mean, obs_var = _obs_norm_stats_for_render(final_env_state)
            pipeline_states = _run_render_rollout(params, rng, obs_mean, obs_var)
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
        teacher_params = teacher_network.init(
            teacher_init_rng, jnp.zeros((teacher_obs_dim,))
        )

        def _build_teacher_input(obs, competence_vector):
            if not condition_teacher_on_competence:
                return obs
            comp_batch = jnp.broadcast_to(
                competence_vector, (obs.shape[0], competence_vector.shape[0])
            )
            return jnp.concatenate([obs, comp_batch], axis=-1)

        def sample_teacher_goals(obs, competence_vector, rng):
            teacher_obs = _build_teacher_input(obs, competence_vector)
            pi, _ = teacher_network.apply(teacher_params, teacher_obs)
            idx = pi.sample(seed=rng)
            return goal_grid[idx].astype(jnp.float32)

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
            jax.debug.print("obs_mean: {obs_mean}", obs_mean=obs_mean[0])
            jax.debug.print("obs_var: {obs_var}", obs_var=obs_var[0])

        # INIT ENV
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        if config["NORMALIZE_ENV"]:
            obsv, env_state = env.reset_with_stats(
                reset_rng, warmup_env_state, env_params
            )
            jax.debug.print(
                "post_reset_obs_mean: {obs_mean}", obs_mean=env_state.mean[0]
            )
            jax.debug.print(
                "post_reset_obs_var: {obs_var}", obs_var=env_state.var[0]
            )
        else:
            obsv, env_state = env.reset(reset_rng, env_params)

        competence_vector = jnp.zeros((num_competence,), dtype=obsv.dtype)
        raw_goals = sample_teacher_goals(
            obsv[..., :base_obs_dim], competence_vector, goal_rng
        )
        goals = raw_goals
        if condition_on_goal:
            if config["NORMALIZE_ENV"]:
                mean_xy = env_state.mean[..., :2]
                var_xy = env_state.var[..., :2]
                goals = (raw_goals - mean_xy) / jnp.sqrt(var_xy + 1e-8)
            obsv = jnp.concatenate([obsv, goals], axis=-1)
            
        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
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
                if condition_teacher_on_competence:
                    competence_vector = jax.lax.cond(
                        jnp.any(done),
                        lambda _: compute_competence_vector(
                            train_state.params, env_state
                        ),
                        lambda _: competence_vector,
                        operand=None,
                    )
                # jax.debug.print("competence_vector: {competence_vector}", competence_vector=competence_vector)
                teacher_goals = sample_teacher_goals(
                    obsv[..., :base_obs_dim], competence_vector, goal_rng
                )
                raw_goals = jnp.where(done[:, None], teacher_goals, raw_goals)
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
                    done, action, value, reward, task_reward, goal_reward, log_prob, last_obs, info
                )
                runner_state = (
                    train_state,
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
                    rng,
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
                goals,
                raw_goals,
                competence_vector,
                task_reward_sum,
                task_reward_sq_sum,
                goal_reward_sum,
                goal_reward_sq_sum,
                reward_count,
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
            # import pdb; pdb.set_trace()
            total_loss = loss_info[0].mean()
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
                    ) = args
                    now = time.perf_counter()
                    log_metrics = {}
                    if _wandb_timer["last_time"] is not None:
                        elapsed = now - _wandb_timer["last_time"]
                        if elapsed > 0:
                            log_metrics["sps"] = float(steps_per_update / elapsed)
                    _wandb_timer["last_time"] = now

                    # Keep all metrics on a consistent global step for stable wandb curves.
                    step = int(info["timestep"].max() * config["NUM_ENVS"])
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    if len(return_values) > 0:
                        log_metrics["episodic_return"] = float(return_values.mean())
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
                            "obs_norm_mean": float(obs_norm_mean),
                            "obs_norm_var": float(obs_norm_var),
                            "student_competence_mean": float(competence_mean),
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
                    ),
                )

            runner_state = (
                train_state,
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
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        reward_dtype = obsv.dtype
        runner_state = (
            train_state,
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
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {
            "runner_state": runner_state,
            "metrics": metric,
        }

    return train, render_eval_episode


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
    train_fn, render_eval_episode = make_train(config)
    train_jit = jax.jit(train_fn)
    train_output = train_jit(rng)
    jax.block_until_ready(train_output["runner_state"][11])
    final_train_state, final_env_state, _, _, _, _, _, _, _, _, _, final_rng = train_output[
        "runner_state"
    ]
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
    render_eval_episode(final_train_state.params, final_rng, final_env_state)


if __name__ == "__main__":
    main()
