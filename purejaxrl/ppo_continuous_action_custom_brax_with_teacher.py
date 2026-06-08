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
    NUM_STEPS: int = 32
    TOTAL_TIMESTEPS: int = int(1e8)
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 32
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.3
    ENT_COEF: float = 0.0
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTIVATION: str = "tanh"
    ENV_NAME: str = "ant_u_maze_single_goal"
    ENV_BACKEND: str | None = None
    EPISODE_LENGTH: int = 1000
    ACTION_REPEAT: int = 1
    # Pass as JSON from CLI, e.g. --env-kwargs '{"foo": 1}'.
    ENV_KWARGS: str = "{}"
    ANNEAL_LR: bool = True
    NORMALIZE_ENV: bool = True
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
    GOAL_REWARD_WEIGHT: float = 1.0
    ANNEAL_GOAL_REWARD_WEIGHT: bool = True
    STUDENT_GOAL_REWARD_TYPE: str="sparse" # this can either sparse or dense
    TEACHER_DELTA_LOW: float = -10.0
    TEACHER_DELTA_HIGH: float = 10.0
    TEACHER_NUM_PROBING_STATES: int = 100
    TEACHER_PROBE_AGG: str = "concat"
    TEACHER_SAMPLE_EVERY_N_EPISODES: int = 1
    GOAL_REACHED_THRESHOLD: float = 0.1
    SUCCESS_RATE_ALPHA: float = 0.05
    TEACHER_REWARD_TYPE: str = "goal_return"
    TEACHER_EPISODE_LENGTH: int = 8
    TEACHER_LR: float = 3e-4
    TEACHER_GAMMA: float = 1.0
    TEACHER_GAE_LAMBDA: float = 0.95
    TEACHER_CLIP_EPS: float = 0.3
    TEACHER_ENT_COEF: float = 0.0
    TEACHER_VF_COEF: float = 0.5
    TEACHER_MAX_GRAD_NORM: float = 0.5
    TEACHER_UPDATE_EPOCHS: int = 4
    TEACHER_NUM_MINIBATCHES: int = 4

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

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class TeacherGoalPolicy(nn.Module):
    """Stochastic PPO policy that maps an observation to a goal delta.

    The actor outputs a Gaussian over an unbounded 2D raw delta (x/y only); the
    raw sample is squashed externally (see ``_squash_delta``) so each component
    lies in ``[delta_low, delta_high]`` and the goal is
    ``reference_obs[..., :2] + delta_xy``.
    PPO trains on the raw (pre-squash) action, mirroring how the student uses the
    ``ClipAction`` wrapper.
    """

    obs_dim: int
    student_obs_dim: int
    num_probing_states: int
    delta_low: float = -1.0
    delta_high: float = 1.0
    probe_agg: str = "mean"
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, student_apply, student_params):
        # jax.debug.print("teacher input shape: {x}", x=x)
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # Learnable probing states ~ U(-1, 1), sized to the student's observation
        # so they can be fed through the student policy.
        probing_states = self.param(
            "probing_states",
            lambda key, shape: jax.random.uniform(key, shape, jnp.float32, -1.0, 1.0),
            (self.num_probing_states, self.student_obs_dim),
        )
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
        actor_in = jnp.concatenate([x, jax.lax.stop_gradient(probe_b)], axis=-1)
        critic_in = jnp.concatenate([x, probe_b], axis=-1)

        actor_h = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_in)
        actor_h = activation(actor_h)
        actor_h = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_h)
        actor_h = activation(actor_h)
        actor_mean = nn.Dense(
            self.obs_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_h)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.obs_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        critic_h = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_in)
        critic_h = activation(critic_h)
        critic_h = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
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
    """One teacher transition = one completed student episode."""

    done: jnp.ndarray
    action: jnp.ndarray  # raw (pre-squash) delta
    value: jnp.ndarray
    reward: jnp.ndarray  # learning progress for that episode
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # teacher input = [reference_obs, avg_success_rate]


class PendingTeacher(NamedTuple):
    """Per-env components of the goal proposal currently active for an episode."""

    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


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
    goal_reward_weight = config.get("GOAL_REWARD_WEIGHT", 0.0)
    anneal_goal_reward_weight = config.get("ANNEAL_GOAL_REWARD_WEIGHT", False)
    student_goal_reward_type = config.get("STUDENT_GOAL_REWARD_TYPE", "sparse")
    if student_goal_reward_type not in ("sparse", "dense"):
        raise ValueError(
            "STUDENT_GOAL_REWARD_TYPE must be 'sparse' or 'dense'"
        )
    teacher_sample_every_n_episodes = int(
        config.get("TEACHER_SAMPLE_EVERY_N_EPISODES", 1)
    )
    if teacher_sample_every_n_episodes < 1:
        raise ValueError("TEACHER_SAMPLE_EVERY_N_EPISODES must be >= 1")
    teacher_delta_low = config.get("TEACHER_DELTA_LOW", -1.0)
    teacher_delta_high = config.get("TEACHER_DELTA_HIGH", 1.0)
    goal_reached_threshold = config.get("GOAL_REACHED_THRESHOLD", 0.1)
    success_rate_alpha = config.get("SUCCESS_RATE_ALPHA", 0.05)
    teacher_reward_type = config.get("TEACHER_REWARD_TYPE", "success_rate")
    if teacher_reward_type not in ("success_rate", "goal_return"):
        raise ValueError(
            "TEACHER_REWARD_TYPE must be 'success_rate' or 'goal_return'"
        )
    base_obs_dim = int(env.observation_space(env_params).shape[0])
    goal_dim = 2
    policy_obs_dim = base_obs_dim + goal_dim if goal_conditioned else base_obs_dim
    teacher_num_probing_states = int(config.get("TEACHER_NUM_PROBING_STATES", 8))
    teacher_probe_agg = config.get("TEACHER_PROBE_AGG", "mean")

    # Teacher PPO hyper-parameters. The teacher acts on the student-episode
    # timeline: one transition per completed student episode and one teacher
    # episode (== one teacher rollout) every TEACHER_EPISODE_LENGTH episodes.
    teacher_episode_length = int(config.get("TEACHER_EPISODE_LENGTH", 8))
    if teacher_episode_length < 1:
        raise ValueError("TEACHER_EPISODE_LENGTH must be >= 1")
    teacher_lr = config.get("TEACHER_LR", config["LR"])
    teacher_gamma = config.get("TEACHER_GAMMA", config["GAMMA"])
    teacher_gae_lambda = config.get("TEACHER_GAE_LAMBDA", config["GAE_LAMBDA"])
    teacher_clip_eps = config.get("TEACHER_CLIP_EPS", config["CLIP_EPS"])
    teacher_ent_coef = config.get("TEACHER_ENT_COEF", config["ENT_COEF"])
    teacher_vf_coef = config.get("TEACHER_VF_COEF", config["VF_COEF"])
    teacher_max_grad_norm = config.get("TEACHER_MAX_GRAD_NORM", config["MAX_GRAD_NORM"])
    teacher_update_epochs = int(config.get("TEACHER_UPDATE_EPOCHS", config["UPDATE_EPOCHS"]))
    teacher_num_minibatches = int(config.get("TEACHER_NUM_MINIBATCHES", 1))
    teacher_batch_size = teacher_episode_length * config["NUM_ENVS"]
    if teacher_batch_size % teacher_num_minibatches != 0:
        raise ValueError(
            "TEACHER_EPISODE_LENGTH * NUM_ENVS must be divisible by "
            "TEACHER_NUM_MINIBATCHES"
        )
    config["TEACHER_MINIBATCH_SIZE"] = teacher_batch_size // teacher_num_minibatches

    network = ActorCritic(
        env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
    )

    # Trainable teacher policy: maps a reference observation to a goal delta.
    teacher_network = TeacherGoalPolicy(
        obs_dim=goal_dim,
        student_obs_dim=policy_obs_dim,
        num_probing_states=teacher_num_probing_states,
        delta_low=teacher_delta_low,
        delta_high=teacher_delta_high,
        probe_agg=teacher_probe_agg,
        activation=config["ACTIVATION"],
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

    def _squash_delta(raw):
        """Squash an unbounded raw delta into ``[delta_low, delta_high]``."""
        return teacher_delta_low + (teacher_delta_high - teacher_delta_low) * 0.5 * (
            jnp.tanh(raw) + 1.0
        )

    def _teacher_apply(teacher_params, teacher_input, student_params):
        return teacher_network.apply(
            teacher_params, teacher_input, network.apply, student_params
        )

    def _teacher_sample(
        teacher_params, reference_obs, avg_success_rate, student_params, rng
    ):
        """Sample a goal from the teacher policy.

        Returns ``(goal, teacher_input, raw_action, log_prob, value)`` so the
        proposal can be stored as a PPO transition.
        """
        teacher_input = _teacher_input(reference_obs, avg_success_rate)
        pi, value = _teacher_apply(teacher_params, teacher_input, student_params)
        raw_action = pi.sample(seed=rng)
        log_prob = pi.log_prob(raw_action)
        goal = reference_obs[..., :2] + _squash_delta(raw_action)
        return goal, teacher_input, raw_action, log_prob, value

    def _teacher_goal_det(teacher_params, reference_obs, avg_success_rate, student_params):
        """Deterministic goal (uses the policy mean) for eval/rendering."""
        pi, _ = _teacher_apply(
            teacher_params, _teacher_input(reference_obs, avg_success_rate), student_params
        )
        return reference_obs[..., :2] + _squash_delta(pi.mean())

    def _concat_goal(obs, goals):
        if not goal_conditioned:
            return obs
        return jnp.concatenate([obs, goals], axis=-1)

    def _goal_xy_delta(obs, goals):
        # Goal reaching is defined in the positional plane only.
        return obs[..., :2] - goals[..., :2]

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

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
        avg_success_rate = jnp.zeros((config["NUM_ENVS"],), dtype=obsv.dtype)
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
        pending = PendingTeacher(
            obs=init_t_obs,
            action=init_t_action,
            value=init_t_value,
            log_prob=init_t_log_prob,
        )
        # Teacher rollout buffer: one slot per completed student episode, plus a
        # trailing padding row that absorbs out-of-bounds / invalid writes.
        _buf_rows = teacher_episode_length + 1
        teacher_buffer = TeacherTransition(
            done=jnp.zeros((_buf_rows, config["NUM_ENVS"])),
            action=jnp.zeros((_buf_rows, config["NUM_ENVS"], goal_dim)),
            value=jnp.zeros((_buf_rows, config["NUM_ENVS"])),
            reward=jnp.zeros((_buf_rows, config["NUM_ENVS"])),
            log_prob=jnp.zeros((_buf_rows, config["NUM_ENVS"])),
            obs=jnp.zeros((_buf_rows, config["NUM_ENVS"], base_obs_dim + 1)),
        )
        teacher_buffer_count = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)

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
                goal_reward_term = jnp.zeros_like(reward)
                episode_success = jnp.zeros_like(reward)
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

                    goal_reward_term = current_goal_reward_weight * goal_penalty
                    reward = reward + goal_reward_term
                    # Refresh the goal when it is reached, or on done every N episodes.
                    goal_dist = jnp.sqrt(
                        jnp.sum(jnp.square(_goal_xy_delta(obsv, goal_batch)), axis=-1)
                    )
                    reached = goal_dist <= goal_reached_threshold
                    reached_any_in_episode = jnp.logical_or(reached_any_in_episode, reached)
                    done_mask = done.astype(bool)
                    completed_episode_counts = episode_counts + done_mask.astype(jnp.int32)
                    sample_on_done = done_mask & (
                        (completed_episode_counts % teacher_sample_every_n_episodes) == 0
                    )
                    episode_success = reached_any_in_episode.astype(reward.dtype)
                    updated_avg_success_rate = avg_success_rate + success_rate_alpha * (
                        episode_success - avg_success_rate
                    )
                    avg_success_rate = jnp.where(
                        done_mask, updated_avg_success_rate, avg_success_rate
                    )
                    refresh = sample_on_done
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
                    goal_batch = jnp.where(refresh[:, None], new_goals, goal_batch)
                    episode_counts = completed_episode_counts
                    reached_any_in_episode = jnp.where(
                        done_mask,
                        jnp.zeros_like(reached_any_in_episode),
                        reached_any_in_episode,
                    )
                else:
                    done_mask = done.astype(bool)
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
                if teacher_reward_type == "success_rate":
                    progress = (current_success - prev_episode_success)
                else:
                    progress = (current_goal_return - prev_goal_ep_return)
                teacher_reward = jnp.where(
                    done_mask, progress, jnp.zeros_like(progress)
                )
                prev_episode_success = jnp.where(
                    done_mask, current_success, prev_episode_success
                )
                prev_goal_ep_return = jnp.where(
                    done_mask, current_goal_return, prev_goal_ep_return
                )

                if goal_conditioned:
                    # TEACHER COLLECTION: emit one transition per env that just
                    # completed an episode, using the proposal that was active
                    # during it (the current `pending`). Writes land at row=count
                    # for that env; invalid / over-capacity writes are routed to
                    # the trailing padding row and discarded.
                    env_idx = jnp.arange(config["NUM_ENVS"])
                    can_record = done_mask & (
                        teacher_buffer_count < teacher_episode_length
                    )
                    write_row = jnp.where(
                        can_record, teacher_buffer_count, teacher_episode_length
                    )
                    teacher_buffer = TeacherTransition(
                        done=teacher_buffer.done.at[write_row, env_idx].set(
                            jnp.zeros_like(teacher_reward)
                        ),
                        action=teacher_buffer.action.at[write_row, env_idx].set(
                            pending.action
                        ),
                        value=teacher_buffer.value.at[write_row, env_idx].set(
                            pending.value
                        ),
                        reward=teacher_buffer.reward.at[write_row, env_idx].set(
                            teacher_reward
                        ),
                        log_prob=teacher_buffer.log_prob.at[write_row, env_idx].set(
                            pending.log_prob
                        ),
                        obs=teacher_buffer.obs.at[write_row, env_idx].set(pending.obs),
                    )
                    teacher_buffer_count = teacher_buffer_count + can_record.astype(
                        jnp.int32
                    )
                    # Adopt the freshly sampled proposal for envs that resampled.
                    refresh_col = refresh[:, None]
                    pending = PendingTeacher(
                        obs=jnp.where(refresh_col, new_t_obs, pending.obs),
                        action=jnp.where(refresh_col, new_t_action, pending.action),
                        value=jnp.where(refresh, new_t_value, pending.value),
                        log_prob=jnp.where(refresh, new_t_log_prob, pending.log_prob),
                    )

                info = dict(info)
                info["goal_reward_term"] = goal_reward_term
                info["shaped_reward"] = reward
                info["returned_goal_reward_episode_returns"] = returned_goal_ep_returns
                info["returned_goal_reward_episode"] = done_mask
                info["teacher_reward"] = teacher_reward
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
            ) = runner_state
            last_policy_obs = _concat_goal(last_obs, goal_batch)
            _, last_val = network.apply(train_state.params, last_policy_obs)

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
            # Fires (via lax.cond) once every env has collected a full teacher
            # episode (TEACHER_EPISODE_LENGTH student episodes). One teacher
            # rollout == one teacher episode; buffer/counts reset afterwards.
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
                t_train_state, t_buffer, t_count, t_rng, pend = operands
                # Drop the trailing padding row; the rollout is one teacher episode.
                traj = jax.tree_util.tree_map(
                    lambda x: x[:teacher_episode_length], t_buffer
                )
                # Truncation bootstrap: value of the next (currently pending) proposal.
                last_val = pend.value
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

                    t_state, traj, advantages, targets, rng = update_state
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, teacher_batch_size)
                    batch = (traj, advantages, targets)
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((teacher_batch_size,) + x.shape[2:]),
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
                t_train_state, t_buffer, t_count, t_rng, pend = operands
                z = jnp.array(0.0, dtype=jnp.float32)
                return t_train_state, t_buffer, t_count, t_rng, (z, z, z, z, z)

            teacher_should_update = jnp.all(
                teacher_buffer_count >= teacher_episode_length
            )
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
                    pending,
                ),
            )
            (
                teacher_total_loss,
                teacher_value_loss,
                teacher_actor_loss,
                teacher_entropy,
                teacher_did_update,
            ) = teacher_metrics

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
                        current_goal_reward_weight,
                        teacher_total_loss,
                        teacher_value_loss,
                        teacher_actor_loss,
                        teacher_entropy,
                        teacher_did_update,
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
                        "goal_reward_weight": float(current_goal_reward_weight),
                        "goal_reward_term_mean": float(info["goal_reward_term"].mean()),
                        "shaped_reward_mean": float(info["shaped_reward"].mean()),
                    }
                    if len(goal_return_values) > 0:
                        payload["episodic_goal_shaping_return"] = float(
                            goal_return_values.mean()
                        )
                    if len(teacher_reward_values) > 0:
                        payload["teacher_learning_progress"] = float(
                            teacher_reward_values.mean()
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
                        current_goal_reward_weight,
                        teacher_total_loss,
                        teacher_value_loss,
                        teacher_actor_loss,
                        teacher_entropy,
                        teacher_did_update,
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
    train_fn, render_eval_episode = make_train(config)
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


if __name__ == "__main__":
    main()
