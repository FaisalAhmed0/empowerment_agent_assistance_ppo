import os
import traceback
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
from envs.ant_maze import sample_random_goal
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
    from purejaxrl.rnd import RNDNetwork, init_rnd_state, rnd_step, train_predictor
except ImportError:
    from envs.factory import make_custom_env
    from rnd import RNDNetwork, init_rnd_state, rnd_step, train_predictor

import orbax.checkpoint as ocp

from wonderwords import RandomWord



@dataclass
class TrainConfig:
    LR: float = 3e-4
    NUM_ENVS: int = 245
    NUM_STEPS: int = 64
    TOTAL_TIMESTEPS: int = int(5e7)
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 16
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
    DEBUG: bool = False
    SEED: int = 30
    WANDB_MODE: str = "online"
    ENTITY: str = ""
    PROJECT: str = "purejaxrl"
    EVAL_RENDER_STEPS: int = 300
    EVAL_RENDER_MAX_FRAMES: int = 1000
    EVAL_RENDER_HEIGHT: int = 360
    EVAL_RENDER_LOG_WANDB_HTML: bool = True
    TRAIN_RENDER_FREQ: int = 200
    EVAL_FREQ: int = 10
    EVAL_NUM_ENVS: int = 50
    COMMENT: str = ""
    ADD_GOAL_REWARD: bool = False
    USE_ORACLE_REWARD: bool = False
    ORACLE_REWARD_COEF: float = 1.0
    CONDITION_ON_GOAL: bool = False
    GOAL_REACH_EPSILON: float = 0.5
    HIDDEN_DIM: int = 256
    SAVE_MODEL: bool = False
    checkpoint_dir: str = "checkpoints"
    USE_RND: bool = False
    RND_COEF: float = 1.0
    RND_LR: float = 1e-4
    RND_HIDDEN_DIM: int = 256
    RND_OUTPUT_DIM: int = 128
    TASK_REWARD_COEF: float = 1.0

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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    task_reward: jnp.ndarray
    goal_reward: jnp.ndarray
    rnd_reward: jnp.ndarray
    rnd_raw_intrinsic: jnp.ndarray
    rnd_obs: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    returned_goal_success: jnp.ndarray




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


def _inner_brax_state(state):
    """Walk nested wrappers until the inner Brax ``State`` is reached."""
    current = state
    while hasattr(current, "env_state") and not hasattr(current, "pipeline_state"):
        current = current.env_state
    return current


def _success_metric(wrapped_state):
    return _inner_brax_state(wrapped_state).metrics["success"]


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
        mean = mean[0]
        var = var[0]
    return stats_state.replace(mean=mean, var=var)


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
    """Evaluate agent on env goals (no teacher sampling).

    Conditions the policy on each env's own maze goal from ``org_obs[..., -2:]``
    when ``condition_on_goal`` is True. Returns ``(success_rate, episodic_return)``
    averaged over ``num_envs``.
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

    reset_rngs = jax.random.split(rng, num_envs)
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

    # Conditioning goal equals the environment goal (no teacher sampling).
    env_goals = org_obs[..., -2:]
    if condition_on_goal:
        if normalize_obs:
            goal_batch = _normalize_xy(env_goals, norm_mean, norm_var)
        else:
            goal_batch = env_goals

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


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    if config.get("USE_ORACLE_REWARD", False):
        env_kwargs["use_oracle_reward"] = True
        env_kwargs["oracle_reward_coef"] = config["ORACLE_REWARD_COEF"]
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
    use_rnd = config.get("USE_RND", False)
    rnd_network = RNDNetwork(
        hidden_dim=config.get("RND_HIDDEN_DIM", 256),
        output_dim=config.get("RND_OUTPUT_DIM", 128),
    )
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

    def render_eval_episode(params, rng, final_env_state):
        """Runs one eval rollout and logs rendered HTML to wandb."""
        if config.get("WANDB_MODE", "disabled") != "online":
            return
        try:
            obs_mean, obs_var = _obs_norm_stats_for_render(final_env_state)
            pipeline_states = _run_render_rollout(params, rng, obs_mean, obs_var)
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

        obs_dim = int(env.observation_space(env_params).shape[0])
        rng, rnd_rng = jax.random.split(rng)
        rnd_state = init_rnd_state(
            rnd_rng,
            rnd_network,
            obs_dim,
            config["NUM_ENVS"],
            config.get("RND_LR", 1e-4),
            max_grad_norm=config["MAX_GRAD_NORM"],
            dtype=init_x.dtype,
        )

        def evaluate_agent_on_env_goal(params, stats_state, rng):
            return evaluate_student_env_goal(
                env_2,
                base_env_2,
                network,
                params,
                config["EVAL_NUM_ENVS"],
                rng,
                max_steps=config.get("EPISODE_LENGTH", 1000),
                warmup_env_state=stats_state,
                normalize_obs=config["NORMALIZE_ENV"],
                condition_on_goal=condition_on_goal,
            )

        rng, goal_rng,  _rng = jax.random.split(rng, 3)

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
            # jax.debug.print(
            #     "post_reset_obs_mean: {obs_mean}", obs_mean=env_state.mean[0]
            # )
            # jax.debug.print(
            #     "post_reset_obs_var: {obs_var}", obs_var=env_state.var[0]
            # )
        else:
            obsv, env_state = env.reset(reset_rng, env_params)
        
        # if add_goal_reward:
        raw_goals = jax.vmap(sample_random_goal)(
            jax.random.split(goal_rng, config["NUM_ENVS"])
        ).astype(jnp.float32)
        goals = raw_goals
        if condition_on_goal:
            if config["NORMALIZE_ENV"]:
                mean_xy = env_state.mean[..., :2]
                var_xy = env_state.var[..., :2]
                goals = (raw_goals - mean_xy) / jnp.sqrt(var_xy + 1e-8)
            obsv = jnp.concatenate([obsv, goals], axis=-1)
            
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
                    env_state,
                    last_obs,
                    goals,
                    raw_goals,
                    task_reward_sum,
                    task_reward_sq_sum,
                    goal_reward_sum,
                    goal_reward_sq_sum,
                    reward_count,
                    ep_goal_success,
                    returned_ep_goal_success,
                    rnd_state,
                ) = body

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
                step_goal_success = _success_metric(env_state)
                ep_goal_success = jnp.maximum(ep_goal_success, step_goal_success)
                returned_ep_goal_success = (
                    returned_ep_goal_success * (1 - done) + ep_goal_success * done
                )
                ep_goal_success = ep_goal_success * (1 - done)
                task_reward = reward
                goal_reward = jnp.zeros_like(task_reward)
                if add_goal_reward:
                    if config["NORMALIZE_ENV"]:
                        dist = jnp.linalg.norm(
                            env_state.org_obs[..., :2] - raw_goals, axis=-1
                        )
                    else:
                        dist = jnp.linalg.norm(obsv[..., :2] - raw_goals, axis=-1)
                    goal_reward = (dist <= goal_reach_epsilon).astype(task_reward.dtype)
                    reward = task_reward + goal_reward

                if use_rnd:
                    rnd_obs = (
                        env_state.org_obs if config["NORMALIZE_ENV"] else obsv
                    )
                    rnd_state, rnd_reward, rnd_raw_intrinsic = rnd_step(
                        rnd_state,
                        rnd_network,
                        rnd_obs,
                        done,
                        config["GAMMA"],
                    )
                    reward = config["TASK_REWARD_COEF"] * reward + config["RND_COEF"] * rnd_reward
                else:
                    rnd_obs = jnp.zeros(
                        (config["NUM_ENVS"], obs_dim), dtype=task_reward.dtype
                    )
                    rnd_reward = jnp.zeros_like(task_reward)
                    rnd_raw_intrinsic = jnp.zeros_like(task_reward)

                sampled_raw_goals = jax.vmap(sample_random_goal)(
                    jax.random.split(goal_rng, config["NUM_ENVS"])
                )
                raw_goals = jnp.where(done[:, None], sampled_raw_goals, raw_goals)
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
                    rnd_reward,
                    rnd_raw_intrinsic,
                    rnd_obs,
                    log_prob,
                    last_obs,
                    info,
                    returned_ep_goal_success,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    goals,
                    raw_goals,
                    task_reward_sum,
                    task_reward_sq_sum,
                    goal_reward_sum,
                    goal_reward_sq_sum,
                    reward_count,
                    ep_goal_success,
                    returned_ep_goal_success,
                    rnd_state,
                )
                if enable_train_render:
                    runner_state = runner_state + (train_render_buf, rng)
                else:
                    runner_state = runner_state + (rng,)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            *body, rng = runner_state
            if enable_train_render:
                train_render_buf = body[-1]
                body = body[:-1]
            (
                train_state,
                env_state,
                last_obs,
                goals,
                raw_goals,
                task_reward_sum,
                task_reward_sq_sum,
                goal_reward_sum,
                goal_reward_sq_sum,
                reward_count,
                ep_goal_success,
                returned_ep_goal_success,
                rnd_state,
            ) = body
            _, last_val = network.apply(train_state.params, last_obs)

            if use_rnd:
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                rnd_obs_batch = traj_batch.rnd_obs.reshape(
                    (batch_size, obs_dim)
                )
                rnd_state, rnd_predictor_loss = train_predictor(
                    rnd_state, rnd_network, rnd_obs_batch
                )
            else:
                rnd_predictor_loss = jnp.array(0.0, dtype=last_val.dtype)

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
            rnd_reward_mean = traj_batch.rnd_reward.mean()
            rnd_reward_std = traj_batch.rnd_reward.std()
            rnd_raw_intrinsic_mean = traj_batch.rnd_raw_intrinsic.mean()
            rnd_rew_running_std = jnp.sqrt(rnd_state.rew_var + 1e-8)
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
                returned_episode = metric["returned_episode"]
                done_count = returned_episode.sum()
                train_goal_success_rate = jnp.where(
                    done_count > 0,
                    (traj_batch.returned_goal_success * returned_episode).sum()
                    / done_count,
                    jnp.nan,
                )

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
                        train_goal_success_rate,
                        rnd_reward_mean,
                        rnd_reward_std,
                        rnd_raw_intrinsic_mean,
                        rnd_rew_running_std,
                        rnd_predictor_loss,
                    ) = args
                    # Keep all metrics on a consistent global step for stable wandb curves.
                    step = int(info["timestep"].max() * config["NUM_ENVS"])
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    if len(return_values) > 0:
                        log_payload = {
                            "episodic_return": float(return_values.mean()),
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
                        }
                        if use_rnd:
                            log_payload.update(
                                {
                                    "rnd_reward_mean": float(rnd_reward_mean),
                                    "rnd_reward_std": float(rnd_reward_std),
                                    "rnd_raw_intrinsic_mean": float(
                                        rnd_raw_intrinsic_mean
                                    ),
                                    "rnd_rew_running_std": float(rnd_rew_running_std),
                                    "rnd_predictor_loss": float(rnd_predictor_loss),
                                }
                            )
                        if not jnp.isnan(train_goal_success_rate):
                            log_payload["train/goal_success_rate"] = float(
                                train_goal_success_rate
                            )
                        wandb.log(log_payload, step=step)

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
                        train_goal_success_rate,
                        rnd_reward_mean,
                        rnd_reward_std,
                        rnd_raw_intrinsic_mean,
                        rnd_rew_running_std,
                        rnd_predictor_loss,
                    ),
                )

            # Periodic eval on environment goals.
            eval_freq = int(config.get("EVAL_FREQ", 0))
            if eval_freq > 0:
                should_eval = (update_idx) % eval_freq == 0
                rng, eval_rng = jax.random.split(rng)

                def _run_eval(_):
                    # jax.debug.print("running env-goal eval")
                    success_rate, episodic_return = evaluate_agent_on_env_goal(
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

                def _skip_eval(_):
                    return jnp.array(0, dtype=jnp.int32)

                jax.lax.cond(
                    should_eval,
                    _run_eval,
                    _skip_eval,
                    operand=None,
                )

            if enable_train_render:
                should_render = (update_idx) % train_render_freq == 0
                should_log_train_render = jnp.logical_and(
                    should_render, train_render_buf.has_completed
                )

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
                env_state,
                last_obs,
                goals,
                raw_goals,
                task_reward_sum,
                task_reward_sq_sum,
                goal_reward_sum,
                goal_reward_sq_sum,
                reward_count,
                ep_goal_success,
                returned_ep_goal_success,
                rnd_state,
            )
            if enable_train_render:
                runner_state = runner_state + (train_render_buf, rng)
            else:
                runner_state = runner_state + (rng,)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        reward_dtype = obsv.dtype
        runner_state = (
            train_state,
            env_state,
            obsv,
            goals,
            raw_goals,
            jnp.array(0.0, dtype=reward_dtype),
            jnp.array(0.0, dtype=reward_dtype),
            jnp.array(0.0, dtype=reward_dtype),
            jnp.array(0.0, dtype=reward_dtype),
            jnp.array(0.0, dtype=reward_dtype),
            jnp.zeros((config["NUM_ENVS"],), dtype=reward_dtype),
            jnp.zeros((config["NUM_ENVS"],), dtype=reward_dtype),
            rnd_state,
        )
        if enable_train_render:
            runner_state = runner_state + (train_render_buf, _rng)
        else:
            runner_state = runner_state + (_rng,)
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

    scratch = os.environ.get("SCRATCH")
    random_name = RandomWord().word()
    random_id = np.random.randint(1000000000)
    while os.path.exists(
        os.path.join(scratch, "purejaxrl", f"{random_name}_{random_id}")
    ):
        random_name = RandomWord().word()
        random_id = np.random.randint(1000000000)

    config["EXP_DIR"] = os.path.join(scratch, "purejaxrl", f"{random_name}_{random_id}")

    print(f"Experiment directory: {config['EXP_DIR']}")
    os.makedirs(config["EXP_DIR"], exist_ok=True)

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
    rs = train_output["runner_state"]
    jax.block_until_ready(rs[-1])
    final_train_state = rs[0]
    final_env_state = rs[1]
    final_rng = rs[-1]
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
