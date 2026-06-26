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
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from gymnax.environments import spaces
import distrax
from wrappers import (
    GymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    ClipAction,
)
try:
    from purejaxrl.envs.multi_agent_ant_maze import MultiAgentAntMaze
except ImportError:
    from envs.multi_agent_ant_maze import MultiAgentAntMaze


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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# --------------------------------------------------------------------------- #
# Multi-agent environment wrappers (local to this file)
# --------------------------------------------------------------------------- #
class MultiAgentBraxGymnaxWrapper:
    """Wraps a MultiAgentAntMaze (one sim, N ants) for the gymnax-style API.

    Unlike the single-agent BraxGymnaxWrapper, ``step`` returns a *per-agent*
    reward vector ``(N,)`` (pulled from ``state.metrics['reward_agent_i']``) and a
    scalar ``done`` which, when the env is built with
    ``terminate_when_unhealthy=False``, is exactly the episode-length truncation
    flag produced by ``EpisodeWrapper``.
    """

    def __init__(self, base_env, episode_length=1000, action_repeat=1):
        self.n_agents = int(base_env.num_agents)
        self.per_agent_obs_dim = int(base_env._per_agent_obs_dim)
        self.act_per_agent = int(base_env._act_per_agent)
        env = EpisodeWrapper(
            base_env, episode_length=episode_length, action_repeat=action_repeat
        )
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = base_env.action_size
        self.observation_size = (base_env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        per_agent_reward = jnp.stack(
            [next_state.metrics[f"reward_agent_{i}"] for i in range(self.n_agents)],
            axis=-1,
        )
        done = next_state.done > 0.5
        return next_state.obs, next_state, per_agent_reward, done, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_size,),
        )


@struct.dataclass
class MultiAgentLogEnvState:
    env_state: Any
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: int


class MultiAgentLogWrapper(GymnaxWrapper):
    """Like LogWrapper but tracks a per-agent episode return ``(N,)``."""

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        n = self._env.n_agents
        state = MultiAgentLogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((n,)),
            episode_lengths=0.0,
            returned_episode_returns=jnp.zeros((n,)),
            returned_episode_lengths=0.0,
            timestep=0,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        done_f = jnp.float32(done)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = MultiAgentLogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done_f),
            episode_lengths=new_episode_length * (1 - done_f),
            returned_episode_returns=state.returned_episode_returns * (1 - done_f)
            + new_episode_return * done_f,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done_f)
            + new_episode_length * done_f,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


@struct.dataclass
class MultiAgentNormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: jnp.ndarray
    env_state: Any


class MultiAgentNormalizeVecReward(GymnaxWrapper):
    """Per-agent running reward normalization (reward shaped ``(NUM_ENVS, N)``)."""

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        n = self._env.n_agents
        state = MultiAgentNormalizeVecRewEnvState(
            mean=jnp.zeros((n,)),
            var=jnp.ones((n,)),
            count=1e-4,
            return_val=jnp.zeros((batch_count, n)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        done_f = jnp.float32(done)[:, None]
        return_val = state.return_val * self.gamma * (1 - done_f) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = MultiAgentNormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(new_var + 1e-8), done, info


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env_kwargs = config.get("ENV_KWARGS", {})

    # Build the multi-agent maze env directly so we can disable health-based
    # termination: with terminate_when_unhealthy=False the only episode boundary
    # is the episode-length truncation produced by EpisodeWrapper.
    env_name = config["ENV_NAME"]
    if env_name.startswith("ant_multi_"):
        maze_layout_name = env_name[len("ant_multi_") :]
    else:
        maze_layout_name = config.get("MAZE_LAYOUT_NAME", "u_maze")
    base_env = MultiAgentAntMaze(
        backend=config.get("ENV_BACKEND") or "spring",
        maze_layout_name=maze_layout_name,
        n_agents=int(env_kwargs.get("n_agents", 2)),
        dense_reward=bool(env_kwargs.get("dense_reward", True)),
        terminate_when_unhealthy=bool(config.get("TERMINATE_WHEN_UNHEALTHY", False)),
    )

    n_agents = int(base_env.num_agents)
    per_agent_obs_dim = int(base_env._per_agent_obs_dim)
    act_per_agent = int(base_env._act_per_agent)

    env = MultiAgentBraxGymnaxWrapper(
        base_env=base_env,
        episode_length=config.get("EPISODE_LENGTH", 1000),
        action_repeat=config.get("ACTION_REPEAT", 1),
    )
    env_params = None
    env = MultiAgentLogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = MultiAgentNormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # Each of the N agents owns an independent copy of this network.
    network = ActorCritic(act_per_agent, activation=config["ACTIVATION"])

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

    def _agent_mean_action(params, obs):
        # Keep the distribution inside the vmap; return only arrays.
        pi, _ = network.apply(params, obs)
        return pi.mean()

    def _eval_render_action(params, obs, obs_mean, obs_var):
        # params: leading-N stacked params. obs: full (N*d,) observation.
        norm_obs = _normalize_eval_obs(obs, obs_mean, obs_var)
        obs_split = norm_obs.reshape(n_agents, per_agent_obs_dim)
        mean_action = jax.vmap(_agent_mean_action)(params, obs_split)  # (N, act)
        full_action = mean_action.reshape(n_agents * act_per_agent)
        return jnp.clip(full_action, action_low, action_high)

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
        """Runs one eval rollout (all N policies) and logs rendered HTML to wandb."""
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
        # INIT N INDEPENDENT NETWORKS / OPTIMIZERS (leading agent axis)
        rng, _rng = jax.random.split(rng)
        agent_keys = jax.random.split(_rng, n_agents)
        init_x = jnp.zeros((per_agent_obs_dim,))
        network_params = jax.vmap(network.init, in_axes=(0, None))(agent_keys, init_x)
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

        def _make_train_state(params):
            return TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=tx,
            )

        # vmap over the agent axis -> params/opt_state/step all carry leading N.
        train_state = jax.vmap(_make_train_state)(network_params)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        def _split_obs(obs):
            # (NUM_ENVS, N*d) -> agent-first (N, NUM_ENVS, d)
            return obs.reshape(
                config["NUM_ENVS"], n_agents, per_agent_obs_dim
            ).transpose(1, 0, 2)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION (vmapped over the agent axis). The distrax
                # distribution is created and consumed inside the vmapped fn so
                # that only arrays cross the vmap boundary.
                obs_af = _split_obs(last_obs)  # (N, NUM_ENVS, d)
                rng, _rng = jax.random.split(rng)
                agent_keys = jax.random.split(_rng, n_agents)

                def _agent_act(params, obs, key):
                    pi, value = network.apply(params, obs)
                    action = pi.sample(seed=key)
                    log_prob = pi.log_prob(action)
                    return action, log_prob, value

                action, log_prob, value = jax.vmap(_agent_act)(
                    train_state.params, obs_af, agent_keys
                )  # action (N, NUM_ENVS, act); log_prob/value (N, NUM_ENVS)

                # Assemble concatenated env action (NUM_ENVS, N*act)
                full_action = action.transpose(1, 0, 2).reshape(
                    config["NUM_ENVS"], n_agents * act_per_agent
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, full_action, env_params
                )
                # reward (NUM_ENVS, N), done (NUM_ENVS,) -> agent-first
                reward_af = reward.T  # (N, NUM_ENVS)
                done_af = jnp.broadcast_to(
                    jnp.float32(done)[None, :], (n_agents, config["NUM_ENVS"])
                )
                transition = Transition(
                    done_af, action, value, reward_af, log_prob, obs_af, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE (per agent; agent axis rides along elementwise)
            train_state, env_state, last_obs, rng = runner_state
            last_obs_af = _split_obs(last_obs)

            def _agent_value(params, obs):
                _, value = network.apply(params, obs)
                return value

            last_val = jax.vmap(_agent_value)(train_state.params, last_obs_af)

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

            # UPDATE EACH AGENT INDEPENDENTLY (vmap over the agent axis)
            def _single_agent_update(
                train_state, obs, action, value_old, logp_old, advantages, targets, rng
            ):
                # All inputs are for ONE agent: shape (NUM_STEPS, NUM_ENVS, ...).
                def _update_epoch(update_state, unused):
                    (
                        train_state,
                        obs,
                        action,
                        value_old,
                        logp_old,
                        advantages,
                        targets,
                        rng,
                    ) = update_state

                    def _update_minbatch(train_state, batch_info):
                        obs_mb, action_mb, value_mb, logp_mb, adv_mb, tgt_mb = batch_info

                        def _loss_fn(params):
                            pi, value = network.apply(params, obs_mb)
                            log_prob = pi.log_prob(action_mb)

                            value_pred_clipped = value_mb + (value - value_mb).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"]
                            )
                            value_losses = jnp.square(value - tgt_mb)
                            value_losses_clipped = jnp.square(value_pred_clipped - tgt_mb)
                            value_loss = (
                                0.5
                                * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )

                            ratio = jnp.exp(log_prob - logp_mb)
                            gae = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                            entropy = pi.entropy().mean()

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(train_state.params)
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    rng, _rng = jax.random.split(rng)
                    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (obs, action, value_old, logp_old, advantages, targets)
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
                    update_state = (
                        train_state,
                        obs,
                        action,
                        value_old,
                        logp_old,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, total_loss

                update_state = (
                    train_state,
                    obs,
                    action,
                    value_old,
                    logp_old,
                    advantages,
                    targets,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                return update_state[0], loss_info

            # Move agent axis (currently axis 1) to the front for the vmap.
            def _agent_first(x):
                return jnp.moveaxis(x, 1, 0)

            rng, _rng = jax.random.split(rng)
            agent_rngs = jax.random.split(_rng, n_agents)
            train_state, loss_info = jax.vmap(_single_agent_update)(
                train_state,
                _agent_first(traj_batch.obs),
                _agent_first(traj_batch.action),
                _agent_first(traj_batch.value),
                _agent_first(traj_batch.log_prob),
                _agent_first(advantages),
                _agent_first(targets),
                agent_rngs,
            )

            metric = traj_batch.info
            # loss_info[0]: (N, UPDATE_EPOCHS, NUM_MINIBATCHES); aux losses likewise.
            per_agent_total_loss = loss_info[0].mean(axis=(1, 2))
            per_agent_value_loss = loss_info[1][0].mean(axis=(1, 2))
            per_agent_actor_loss = loss_info[1][1].mean(axis=(1, 2))
            per_agent_entropy = loss_info[1][2].mean(axis=(1, 2))
            # Average per-step reward per agent over this rollout. reward (T, N, E)
            per_agent_reward = traj_batch.reward.mean(axis=(0, 2))
            current_lr = (
                linear_schedule(
                    update_idx
                    * config["NUM_MINIBATCHES"]
                    * config["UPDATE_EPOCHS"]
                )
                if config["ANNEAL_LR"]
                else config["LR"]
            )

            if config.get("DEBUG"):

                def debug_callback(args):
                    info, per_agent_reward = args
                    returned = np.asarray(info["returned_episode"]).reshape(-1)
                    if returned.sum() == 0:
                        return
                    rets = np.asarray(info["returned_episode_returns"])
                    rets_flat = rets.reshape(-1, rets.shape[-1])[returned.astype(bool)]
                    per_agent_ret = rets_flat.mean(axis=0)
                    step = int(np.asarray(info["timestep"]).max() * config["NUM_ENVS"])
                    print(
                        f"global step={step} | "
                        f"return mean={per_agent_ret.mean():.3f} "
                        f"per-agent={np.round(per_agent_ret, 3).tolist()} | "
                        f"reward mean={float(np.asarray(per_agent_reward).mean()):.3f}"
                    )

                jax.debug.callback(debug_callback, (metric, per_agent_reward))

            if config.get("WANDB_MODE", "disabled") == "online":

                def wandb_callback(args):
                    (
                        info,
                        per_agent_reward,
                        per_agent_total_loss,
                        per_agent_value_loss,
                        per_agent_actor_loss,
                        per_agent_entropy,
                        current_lr,
                    ) = args
                    # Match the single-agent script: only write to wandb on
                    # updates where at least one episode completed. Otherwise the
                    # callback fires every update and the host sync + wandb write
                    # serialize the device pipeline (very high logging frequency).
                    returned = np.asarray(info["returned_episode"]).reshape(-1)
                    if returned.sum() == 0:
                        return

                    step = int(np.asarray(info["timestep"]).max() * config["NUM_ENVS"])
                    log_dict = {}

                    rets = np.asarray(info["returned_episode_returns"])
                    rets_flat = rets.reshape(-1, rets.shape[-1])[returned.astype(bool)]
                    per_agent_ret = rets_flat.mean(axis=0)
                    for i in range(n_agents):
                        log_dict[f"return/agent_{i}"] = float(per_agent_ret[i])
                    log_dict["return/mean"] = float(per_agent_ret.mean())

                    per_agent_reward = np.asarray(per_agent_reward)
                    for i in range(n_agents):
                        log_dict[f"reward/agent_{i}"] = float(per_agent_reward[i])
                        log_dict[f"loss/total_agent_{i}"] = float(
                            per_agent_total_loss[i]
                        )
                        log_dict[f"loss/value_agent_{i}"] = float(
                            per_agent_value_loss[i]
                        )
                        log_dict[f"loss/actor_agent_{i}"] = float(
                            per_agent_actor_loss[i]
                        )
                        log_dict[f"loss/entropy_agent_{i}"] = float(
                            per_agent_entropy[i]
                        )
                    log_dict["reward/mean"] = float(per_agent_reward.mean())
                    log_dict["loss/total_mean"] = float(np.mean(per_agent_total_loss))
                    log_dict["loss/value_mean"] = float(np.mean(per_agent_value_loss))
                    log_dict["loss/actor_mean"] = float(np.mean(per_agent_actor_loss))
                    log_dict["loss/entropy_mean"] = float(np.mean(per_agent_entropy))
                    log_dict["learning_rate"] = float(current_lr)
                    wandb.log(log_dict, step=step)

                jax.debug.callback(
                    wandb_callback,
                    (
                        metric,
                        per_agent_reward,
                        per_agent_total_loss,
                        per_agent_value_loss,
                        per_agent_actor_loss,
                        per_agent_entropy,
                        current_lr,
                    ),
                )

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {
            "runner_state": runner_state,
            "metrics": metric,
        }

    return train, render_eval_episode


def main():
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.3,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "ant_multi_u_maze_many_starts",
        "ENV_BACKEND": None,
        "EPISODE_LENGTH": 1000,
        "ACTION_REPEAT": 1,
        "ENV_KWARGS": {"n_agents": 2, "dense_reward": True},
        "TERMINATE_WHEN_UNHEALTHY": False,
        "ANNEAL_LR": True,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
        "SEED": 30,
        "WANDB_MODE": "online",  # set to "online" to activate wandb
        "ENTITY": "",
        "PROJECT": "purejaxrl",
        "EVAL_RENDER_STEPS": 300,
        "EVAL_RENDER_MAX_FRAMES": 100,
        "EVAL_RENDER_HEIGHT": 360,
        "EVAL_RENDER_LOG_WANDB_HTML": True,
    }

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["PPO", "BRAX", "MULTI_AGENT", config["ENV_NAME"], f"jax_{jax.__version__}"],
        name=f'purejaxrl_ppo_brax_multi_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn, render_eval_episode = make_train(config)
    train_jit = jax.jit(train_fn)
    train_output = train_jit(rng)
    jax.block_until_ready(train_output["runner_state"][3])
    final_train_state, final_env_state, _, final_rng = train_output["runner_state"]
    render_eval_episode(final_train_state.params, final_rng, final_env_state)


if __name__ == "__main__":
    main()
