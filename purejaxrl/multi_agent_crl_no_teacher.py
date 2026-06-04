"""Multi-agent CRL: independent replay buffer, actor, critic, and alpha per ant."""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Literal, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import pmap as brax_pmap
from brax.training import types
from brax.v1 import envs as envs_v1
from flax.struct import dataclass
from flax.training.train_state import TrainState
from jaxgcrl.agents.crl.crl import Transition, save_params
from jaxgcrl.agents.crl.losses import contrastive_loss_fn, energy_fn
from jaxgcrl.agents.crl.networks import Actor, Encoder
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]
_PMAP_AXIS_NAME = "i"


def _agents_per_device(n_agents: int, n_devices: int) -> int:
    return (n_agents + n_devices - 1) // n_devices


def _pad_and_shard_agent_params(params, n_agents: int, n_devices: int, agents_per_device: int):
    """Tree with leading axis ``(n_agents, ...)`` -> ``(n_devices, agents_per_device, ...)``."""
    padded_n = n_devices * agents_per_device
    pad = padded_n - n_agents
    if pad > 0:
        last = jax.tree_util.tree_map(lambda x: x[-1:], params)
        pad_tree = jax.tree_util.tree_map(lambda x: jnp.repeat(x, pad, axis=0), last)
        params = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), params, pad_tree)
    return jax.tree_util.tree_map(
        lambda x: x.reshape((n_devices, agents_per_device) + x.shape[1:]),
        params,
    )


def _merge_sharded_agent_params(params, n_agents: int):
    """``(n_devices, agents_per_device, ...)`` -> ``(n_agents, ...)``."""
    flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), params)
    return jax.tree_util.tree_map(lambda x: x[:n_agents], flat)


def compute_grad_norm(grad):
    flat_grads, _ = jax.flatten_util.ravel_pytree(grad)
    return jnp.linalg.norm(flat_grads)


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch_ma(buffer_config, transition, sample_key):
    """Like CRL ``flatten_batch`` but stores ``future_goal`` for actor (local obs has goal in tail)."""
    gamma, state_size, goal_indices = buffer_config
    # print(f"gamma: {gamma}")
    # print(f"state_size: {state_size}")
    # print(f"goal_indices: {goal_indices}")
    # import pdb; pdb.set_trace()
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount
    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state_full = jnp.take(transition.observation, goal_index[:-1], axis=0)
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state_full[:, goal_indices]
    future_state = future_state_full[:, :state_size]
    state = transition.observation[:-1, :state_size]
    new_obs = jnp.concatenate([state, goal], axis=1)
    future_goal = goal

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
        "future_goal": future_goal,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def _local_obs_from_joint_row(row: jnp.ndarray, agent_i: int, per_agent_obs_size: int) -> jnp.ndarray:
    """Extract ``[qpos_i, qvel_i, goal_i]`` block from packed joint observation."""
    start = agent_i * per_agent_obs_size
    return jax.lax.dynamic_slice_in_dim(row, start, per_agent_obs_size, axis=-1)


def _slice_joint_trajectory_for_agent(
    joint: Transition,
    agent_i: int,
    *,
    n_agents: int,
    per_agent_obs_size: int,
    per_agent_action_size: int,
) -> Transition:
    """(unroll, num_envs, ...) joint trajectory -> per-agent local transition."""
    obs = joint.observation
    act = joint.action
    obs_i = jax.vmap(jax.vmap(lambda row: _local_obs_from_joint_row(row, agent_i, per_agent_obs_size)))(
        obs
    )
    act_r = act.reshape(act.shape[:-1] + (n_agents, per_agent_action_size))
    act_i = act_r[..., agent_i, :]
    ext = joint.extras
    return Transition(
        observation=obs_i,
        action=act_i,
        reward=joint.reward,
        discount=joint.discount,
        extras=ext,
    )


@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


@dataclass
class MultiAgentCRLNoTeacher:
    """CRL with one buffer, actor, critic, and entropy coefficient per ant (no teacher)."""

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = True
    repr_dim: int = 64
    use_ln: bool = True
    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[..., None] = lambda *args: None,
        save_path=None,
    ):
        del randomization_fn
        self.check_config(config)
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        if local_devices_to_use > 1:
            logging.info(
                "multi-agent CRL: packed agent pmap over %s device(s)",
                local_devices_to_use,
            )
            return self._train_multi_device_pmap(
                config,
                train_env,
                eval_env,
                progress_fn,
                save_path,
                local_devices_to_use,
            )
        return self._train_single_device(
            config, train_env, eval_env, progress_fn, save_path
        )

    def _train_single_device(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        progress_fn: Callable[..., None] = lambda *args: None,
        save_path=None,
    ):
        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )
        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        action_size = train_env.action_size
        print(f"action_size: {action_size}")
        joint_state_dim = int(train_env.state_dim)
        print(f"joint_state_dim: {joint_state_dim}")
        per_agent_D = int(getattr(train_env, "per_agent_state_dim", joint_state_dim))
        print(f"per_agent_D: {per_agent_D}")
        n_agents = int(getattr(train_env, "num_agents", getattr(train_env, "_n_agents", 1)))
        print(f"n_agents: {n_agents}")
        assert n_agents >= 1
        assert joint_state_dim == n_agents * per_agent_D, (
            f"state_dim {joint_state_dim} != n_agents * per_agent_D ({n_agents} * {per_agent_D})"
        )
        per_agent_action_size = action_size // n_agents
        assert action_size % n_agents == 0, f"action_size {action_size} not divisible by n_agents {n_agents}"
        print(f"per_agent_action_size: {per_agent_action_size}")
        joint_obs_size = int(train_env.observation_size)
        per_agent_obs_size = per_agent_D + 2
        print(f"per_agent_obs_size: {per_agent_obs_size}")
        assert joint_obs_size == n_agents * per_agent_obs_size, (
            f"observation_size {joint_obs_size} != n_agents * per_agent_obs_size "
            f"({n_agents} * {per_agent_obs_size})"
        )
        goal_idx_tuple = tuple(train_env.goal_indices.tolist())
        print(f"goal_idx_tuple: {goal_idx_tuple}")
        buffer_config_local = (self.discounting, per_agent_D, goal_idx_tuple)
        print(f"buffer_config_local: {buffer_config_local}")
        # import pdb; pdb.set_trace()
        target_entropy = -0.5 * per_agent_action_size

        def stack_param_trees(*trees):
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)

        actor_keys = jax.random.split(actor_key, n_agents)
        sa_keys = jax.random.split(sa_key, n_agents)
        g_keys = jax.random.split(g_key, n_agents)

        actor = Actor(
            action_size=per_agent_action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_params_stacked = stack_param_trees(
            *[actor.init(k, np.ones([1, per_agent_obs_size])) for k in actor_keys]
        )
        # NOTE: I think the training state should be separate for each agent.
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params_stacked,
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder_params_stacked = stack_param_trees(
            *[sa_encoder.init(sk, np.ones([1, per_agent_D + per_agent_action_size])) for sk in sa_keys]
        )
        g_encoder_params_stacked = stack_param_trees(
            *[g_encoder.init(gk, np.ones([1, 2])) for gk in g_keys]
        )
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params_stacked, "g_encoder": g_encoder_params_stacked},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        log_alpha_vec = jnp.zeros((n_agents,), dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha_vec},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        dummy_obs = jnp.zeros((per_agent_obs_size,))
        dummy_action = jnp.zeros((per_agent_action_size,))
        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_keys = jax.random.split(buffer_key, n_agents)
        buffer_states = tuple(jax.jit(replay_buffer.init)(k) for k in buffer_keys)

        def stacked_local_obs(joint_obs: jnp.ndarray) -> jnp.ndarray:
            """joint_obs (B, O) -> (n_agents, B, per_agent_obs_size)."""
            return jnp.stack(
                [
                    jax.vmap(
                        lambda row, i=i: _local_obs_from_joint_row(row, i, per_agent_obs_size)
                    )(joint_obs)
                    for i in range(n_agents)
                ],
                axis=0,
            )

        def multi_actor_stochastic(actor_params_stacked, joint_obs, rng):
            loc = stacked_local_obs(joint_obs)
            means, log_stds = jax.vmap(actor.apply, in_axes=(0, 0))(actor_params_stacked, loc)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(rng, shape=means.shape)
            actions_ag = nn.tanh(means + stds * noise)
            bsz = joint_obs.shape[0]
            return jnp.swapaxes(actions_ag, 0, 1).reshape(bsz, action_size)

        def multi_actor_mean(actor_params_stacked, joint_obs):
            loc = stacked_local_obs(joint_obs)
            means, _ = jax.vmap(actor.apply, in_axes=(0, 0))(actor_params_stacked, loc)
            bsz = joint_obs.shape[0]
            return jnp.swapaxes(nn.tanh(means), 0, 1).reshape(bsz, action_size)

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            actions = multi_actor_mean(training_state.actor_state.params, env_state.obs)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def actor_step(actor_state, env, env_state, key, extra_fields):
            loc = stacked_local_obs(env_state.obs)
            means_st, log_stds_st = jax.vmap(actor.apply, in_axes=(0, 0))(actor_state.params, loc)
            stds_st = jnp.exp(log_stds_st)
            _, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, shape=means_st.shape)
            actions_ag = nn.tanh(means_st + stds_st * noise)
            bsz = env_state.obs.shape[0]
            actions = jnp.swapaxes(actions_ag, 0, 1).reshape(bsz, action_size)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def insert_all_buffers(bss: Tuple[Any, ...], joint_tr: Transition) -> Tuple[Any, ...]:
            new_states = []
            for i in range(n_agents):
                tr_i = _slice_joint_trajectory_for_agent(
                    joint_tr,
                    i,
                    n_agents=n_agents,
                    per_agent_obs_size=per_agent_obs_size,
                    per_agent_action_size=per_agent_action_size,
                )
                new_states.append(replay_buffer.insert(bss[i], tr_i))
            return tuple(new_states)

        @jax.jit
        def get_experience(actor_state, env_state, buffer_states_tuple, key):
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    actor_state,
                    train_env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), transition

            (env_state, _), joint_data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)
            new_buffers = insert_all_buffers(buffer_states_tuple, joint_data)
            return env_state, new_buffers

        def prefill_replay_buffer(training_state, env_state, buffer_states_tuple, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_states_tuple, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_states_tuple = get_experience(
                    training_state.actor_state,
                    env_state,
                    buffer_states_tuple,
                    key,
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_states_tuple, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_states_tuple, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        def sample_all_buffers(bss: Tuple[Any, ...]):
            out_bs = []
            out_tr = []
            for i in range(n_agents):
                bs, tr = replay_buffer.sample(bss[i])
                out_bs.append(bs)
                out_tr.append(tr)
            return tuple(out_bs), tuple(out_tr)

        energy_name = self.energy_fn
        contrastive_name = self.contrastive_loss_fn

        @jax.jit
        def update_networks(carry, transitions_stacked):
            """transitions_stacked: pytree with leading axis (n_agents, batch, ...)."""
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)
            actor_keys_b = jax.random.split(actor_key, n_agents)
            critic_keys_b = jax.random.split(critic_key, n_agents)

            # critic_frozen = jax.lax.stop_gradient(training_state.critic_state.params)
            alpha_vec = training_state.alpha_state.params["log_alpha"]

            def actor_loss_single(actor_p, critic_p, la_i, tr, key_b):
                obs = tr.observation
                state = obs[:, :per_agent_D]
                goal = tr.extras["future_goal"]
                observation = jnp.concatenate([state, goal], axis=-1)
                means, log_stds = actor.apply(actor_p, observation)
                stds = jnp.exp(log_stds)
                x_ts = means + stds * jax.random.normal(key_b, shape=means.shape, dtype=means.dtype)
                action = nn.tanh(x_ts)
                log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
                log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
                log_prob = log_prob.sum(-1)

                sa_encoder_params, g_encoder_params = critic_p["sa_encoder"], critic_p["g_encoder"]
                sa_repr = sa_encoder.apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
                g_repr = g_encoder.apply(g_encoder_params, goal)
                qf_pi = energy_fn(energy_name, sa_repr, g_repr)
                loss = jnp.mean(jnp.exp(la_i) * log_prob - qf_pi)
                log_prob_mean = jnp.mean(log_prob)
                return loss, log_prob_mean

            def actor_mean_loss(actor_params):
                losses, lp_means = jax.vmap(actor_loss_single, in_axes=(0, 0, 0, 0, 0))(
                    actor_params,
                    training_state.critic_state.params,
                    alpha_vec,
                    transitions_stacked,
                    actor_keys_b,
                )
                return jnp.mean(losses), lp_means

            (actor_loss_val, log_prob_means), actor_grad = jax.value_and_grad(actor_mean_loss, has_aux=True)(
                training_state.actor_state.params
            )
            new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

            def alpha_loss_fn(alpha_params):
                av = alpha_params["log_alpha"]
                return jnp.mean(jnp.exp(av) * jax.lax.stop_gradient(-log_prob_means - target_entropy))

            alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss_fn)(training_state.alpha_state.params)
            new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)
            training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

            def critic_one(critic_p, tr, key_b):
                del key_b
                state = tr.observation[:, :per_agent_D]
                action = tr.action
                sa_encoder_params, g_encoder_params = critic_p["sa_encoder"], critic_p["g_encoder"]
                sa_repr = sa_encoder.apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
                g_repr = g_encoder.apply(g_encoder_params, tr.observation[:, per_agent_D:])
                logits = energy_fn(energy_name, sa_repr[:, None, :], g_repr[None, :, :])
                cl = contrastive_loss_fn(contrastive_name, logits)
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                cl = cl + self.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)
                return cl

            def critic_mean_loss(critic_params):
                cls = jax.vmap(critic_one, in_axes=(0, 0, 0))(
                    critic_params,
                    transitions_stacked,
                    critic_keys_b,
                )
                return jnp.mean(cls)

            critic_loss_val, critic_grad = jax.value_and_grad(critic_mean_loss)(training_state.critic_state.params)
            new_critic_state = training_state.critic_state.apply_gradients(grads=critic_grad)
            training_state = training_state.replace(critic_state=new_critic_state)
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            actor_metrics = {
                "entropy_per_agent": -log_prob_means,
                "actor_loss": actor_loss_val,
                "alpha_loss": alpha_loss_val,
                "log_alpha_per_agent": new_alpha_state.params["log_alpha"],
                "actor_grad_norm": compute_grad_norm(actor_grad),
            }
            critic_metrics = {
                "critic_loss": critic_loss_val,
                "critic_grad_norm": compute_grad_norm(critic_grad),
                "categorical_accuracy": jnp.array(0.0),
                "logits_pos": jnp.array(0.0),
                "logits_neg": jnp.array(0.0),
                "logsumexp": jnp.array(0.0),
            }
            metrics = {**actor_metrics, **critic_metrics}
            return (training_state, key), metrics

        def stack_transitions(tr_list: Tuple[Transition, ...]) -> Transition:
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *tr_list)

        @jax.jit
        def training_step(training_state, env_state, buffer_states_tuple, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)
            env_state, buffer_states_tuple = get_experience(
                training_state.actor_state,
                env_state,
                buffer_states_tuple,
                experience_key1,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            buffer_states_tuple, tr_list = sample_all_buffers(buffer_states_tuple)
            agent_sample_keys = jax.random.split(sampling_key, n_agents + 1)
            transitions_per_agent = []
            for i in range(n_agents):
                tr = tr_list[i]
                sk_use = jax.random.split(agent_sample_keys[i], tr.observation.shape[0])
                tr_flat = jax.vmap(flatten_batch_ma, in_axes=(None, 0, 0))(
                    buffer_config_local,
                    tr,
                    sk_use,
                )
                tr_flat = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), tr_flat)
                perm = jax.random.permutation(jax.random.fold_in(experience_key2, i), len(tr_flat.observation))
                tr_flat = jax.tree_util.tree_map(lambda x: x[perm], tr_flat)
                tr_flat = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                    tr_flat,
                )
                transitions_per_agent.append(tr_flat)

            transitions_stacked = stack_transitions(tuple(transitions_per_agent))
            transitions_stacked = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions_stacked)
            (
                (training_state, _),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), transitions_stacked)

            return (training_state, env_state, buffer_states_tuple), metrics

        @jax.jit
        def training_epoch(training_state, env_state, buffer_states_tuple, key):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                ((ts, es, bs), metrics) = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_states_tuple, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_states_tuple, key),
                (),
                length=num_training_steps_per_epoch,
            )
            return training_state, env_state, buffer_states_tuple, metrics

        key, prefill_key = jax.random.split(key, 2)
        training_state, env_state, buffer_states, _ = prefill_replay_buffer(
            training_state, env_state, buffer_states, prefill_key
        )

        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        def expand_per_agent_metrics(flat: dict) -> dict:
            out = {}
            for name, value in flat.items():
                if hasattr(value, "shape") and tuple(value.shape) == (n_agents,):
                    short = name[: -len("_per_agent")] if name.endswith("_per_agent") else name
                    for i in range(n_agents):
                        out[f"training/agent_{i}/{short}"] = value[i]
                    out[f"training/{short}_mean"] = jnp.mean(value)
                    # Keep single-agent metric names available for dashboards.
                    if short == "entropy":
                        out["training/entropy"] = out[f"training/{short}_mean"]
                    elif short == "log_alpha":
                        out["training/log_alpha"] = out[f"training/{short}_mean"]
                else:
                    out[f"training/{name}"] = value
            return out

        training_walltime = 0.0
        logging.info("starting training (multi-agent CRL)....")
        params = None
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_states, metrics = training_epoch(
                training_state, env_state, buffer_states, epoch_key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            metrics = {k: v for k, v in metrics.items()}
            metrics = expand_per_agent_metrics(metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **metrics,
            }
            current_step = int(training_state.env_steps.item())
            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0

            def make_policy(param_stacked):
                def policy(obs, rng):
                    return multi_actor_stochastic(param_stacked, obs, rng)

                return policy

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if save_path:
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps
        logging.info("total steps: %s", total_steps)
        if params is None:
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )

        def make_policy(param_stacked):
            def policy(obs, rng):
                return multi_actor_stochastic(param_stacked, obs, rng)

            return policy

        return make_policy, params, metrics

    def _train_multi_device_pmap(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]],
        progress_fn: Callable[..., None],
        save_path,
        local_devices_to_use: int,
    ):
        """Packed agent sharding: ``pmap`` over devices, ``vmap`` over local agents, one action gather per step."""
        n_devices = local_devices_to_use
        devices = jax.local_devices()[:n_devices]
        device_count = n_devices * jax.process_count()
        assert config.num_envs % device_count == 0, (
            f"num_envs ({config.num_envs}) must be divisible by device_count ({device_count})"
        )
        num_envs_per_device = config.num_envs // device_count

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )
        assert num_training_steps_per_epoch > 0

        action_size = train_env.action_size
        joint_state_dim = int(train_env.state_dim)
        per_agent_D = int(getattr(train_env, "per_agent_state_dim", joint_state_dim))
        n_agents = int(getattr(train_env, "num_agents", getattr(train_env, "_n_agents", 1)))
        per_agent_action_size = action_size // n_agents
        per_agent_obs_size = per_agent_D + 2
        agents_per_device = _agents_per_device(n_agents, n_devices)
        goal_idx_tuple = tuple(train_env.goal_indices.tolist())
        buffer_config_local = (self.discounting, per_agent_D, goal_idx_tuple)
        target_entropy = -0.5 * per_agent_action_size

        assert (num_envs_per_device * (config.episode_length - 1)) % self.batch_size == 0, (
            "num_envs_per_device * (episode_length - 1) must be divisible by batch_size"
        )

        logging.info("n_agents=%s n_devices=%s agents_per_device=%s", n_agents, n_devices, agents_per_device)
        logging.info("num_envs_per_device=%s", num_envs_per_device)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

        def stack_param_trees(*trees):
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)

        actor_keys = jax.random.split(actor_key, n_agents)
        sa_keys = jax.random.split(sa_key, n_agents)
        g_keys = jax.random.split(g_key, n_agents)

        actor = Actor(
            action_size=per_agent_action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )

        actor_params_full = stack_param_trees(
            *[actor.init(k, np.ones([1, per_agent_obs_size])) for k in actor_keys]
        )
        sa_full = stack_param_trees(
            *[sa_encoder.init(sk, np.ones([1, per_agent_D + per_agent_action_size])) for sk in sa_keys]
        )
        g_full = stack_param_trees(*[g_encoder.init(gk, np.ones([1, 2])) for gk in g_keys])
        log_alpha_full = jnp.zeros((n_agents,), dtype=jnp.float32)

        actor_sharded = _pad_and_shard_agent_params(actor_params_full, n_agents, n_devices, agents_per_device)
        sa_sharded = _pad_and_shard_agent_params(sa_full, n_agents, n_devices, agents_per_device)
        g_sharded = _pad_and_shard_agent_params(g_full, n_agents, n_devices, agents_per_device)
        log_alpha_sharded = _pad_and_shard_agent_params(
            log_alpha_full[:, None], n_agents, n_devices, agents_per_device
        )[..., 0]

        actor_tx = optax.adam(learning_rate=self.policy_lr)
        critic_tx = optax.adam(learning_rate=self.critic_lr)
        alpha_tx = optax.adam(learning_rate=self.alpha_lr)

        def _shard_train_state():
            shards = []
            for d in range(n_devices):
                critic_params = {
                    "sa_encoder": jax.tree_util.tree_map(lambda x: x[d], sa_sharded),
                    "g_encoder": jax.tree_util.tree_map(lambda x: x[d], g_sharded),
                }
                shards.append(
                    TrainingState(
                        env_steps=jnp.zeros(()),
                        gradient_steps=jnp.zeros(()),
                        actor_state=TrainState.create(
                            apply_fn=actor.apply,
                            params=jax.tree_util.tree_map(lambda x: x[d], actor_sharded),
                            tx=actor_tx,
                        ),
                        critic_state=TrainState.create(
                            apply_fn=None,
                            params=critic_params,
                            tx=critic_tx,
                        ),
                        alpha_state=TrainState.create(
                            apply_fn=None,
                            params={"log_alpha": log_alpha_sharded[d]},
                            tx=alpha_tx,
                        ),
                    )
                )
            return jax.device_put_sharded(shards, devices)

        training_state = _shard_train_state()

        dummy_transition = Transition(
            observation=jnp.zeros((per_agent_obs_size,)),
            action=jnp.zeros((per_agent_action_size,)),
            reward=0.0,
            discount=0.0,
            extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=num_envs_per_device,
                episode_length=config.episode_length,
            )
        )

        def init_device_buffers(rb_key):
            keys = jax.random.split(rb_key, agents_per_device)
            return jax.vmap(replay_buffer.init)(keys)

        buffer_states = jax.pmap(init_device_buffers, axis_name=_PMAP_AXIS_NAME)(
            jax.random.split(buffer_key, n_devices)
        )

        env_keys = jax.random.split(env_key, config.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (n_devices, num_envs_per_device) + env_keys.shape[1:])
        env_state = jax.pmap(train_env.reset, axis_name=_PMAP_AXIS_NAME)(env_keys)
        train_env.step = jax.jit(train_env.step)

        energy_name = self.energy_fn
        contrastive_name = self.contrastive_loss_fn

        def _assemble_joint_actions(actions_ag):
            gathered = jax.lax.all_gather(actions_ag, axis_name=_PMAP_AXIS_NAME)
            batch = actions_ag.shape[1]
            merged = gathered.transpose(2, 0, 1, 3).reshape(batch, -1)
            return merged[:, :action_size]

        def _partial_stochastic_actions(actor_params, joint_obs, rng, agent_start):
            def local_obs(local_a):
                g = agent_start + local_a

                def row_obs(row):
                    return jax.lax.cond(
                        g < n_agents,
                        lambda: _local_obs_from_joint_row(row, g, per_agent_obs_size),
                        lambda: jnp.zeros((per_agent_obs_size,), dtype=row.dtype),
                    )

                return jax.vmap(row_obs)(joint_obs)

            loc = jax.vmap(local_obs)(jnp.arange(agents_per_device))
            means, log_stds = jax.vmap(actor.apply, in_axes=(0, 0))(actor_params, loc)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(rng, shape=means.shape)
            return nn.tanh(means + stds * noise)

        def _insert_device_buffers(bss, joint_tr, agent_start):
            def insert_one(local_i, bs):
                g = agent_start + local_i

                def do_insert(_):
                    tr_i = _slice_joint_trajectory_for_agent(
                        joint_tr,
                        g,
                        n_agents=n_agents,
                        per_agent_obs_size=per_agent_obs_size,
                        per_agent_action_size=per_agent_action_size,
                    )
                    return replay_buffer.insert(bs, tr_i)

                return jax.lax.cond(g < n_agents, do_insert, lambda _: bs, operand=None)

            return jax.vmap(insert_one)(jnp.arange(agents_per_device), bss)

        def actor_step_pmap(actor_state, env_state, key):
            agent_start = jax.lax.axis_index(_PMAP_AXIS_NAME) * agents_per_device
            actions_ag = _partial_stochastic_actions(
                actor_state.params, env_state.obs, key, agent_start
            )
            actions = _assemble_joint_actions(actions_ag)
            nstate = train_env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in ("truncation", "traj_id")}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def get_experience(actor_state, env_state, buffer_states, key):
            def f(carry, _):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step_pmap(actor_state, env_state, current_key)
                return (env_state, next_key), transition

            agent_start = jax.lax.axis_index(_PMAP_AXIS_NAME) * agents_per_device
            (env_state, _), joint_data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)
            new_buffers = _insert_device_buffers(buffer_states, joint_data, agent_start)
            return env_state, new_buffers

        def prefill_replay_buffer(training_state, env_state, buffer_states, key):
            def f(carry, _):
                training_state, env_state, buffer_states, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_states = get_experience(
                    training_state.actor_state, env_state, buffer_states, key
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_states, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_states, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

        def _merge_training_state_for_eval(ts):
            actor_p = _merge_sharded_agent_params(ts.actor_state.params, n_agents)
            critic_p = {
                "sa_encoder": _merge_sharded_agent_params(ts.critic_state.params["sa_encoder"], n_agents),
                "g_encoder": _merge_sharded_agent_params(ts.critic_state.params["g_encoder"], n_agents),
            }
            log_alpha = ts.alpha_state.params["log_alpha"].reshape(-1)[:n_agents]
            env_steps = ts.env_steps.reshape(-1)[0] if jnp.ndim(ts.env_steps) > 0 else ts.env_steps
            grad_steps = (
                ts.gradient_steps.reshape(-1)[0] if jnp.ndim(ts.gradient_steps) > 0 else ts.gradient_steps
            )
            return TrainingState(
                env_steps=env_steps,
                gradient_steps=grad_steps,
                actor_state=TrainState.create(
                    apply_fn=actor.apply,
                    params=actor_p,
                    tx=optax.adam(learning_rate=self.policy_lr),
                ),
                critic_state=TrainState.create(
                    apply_fn=None,
                    params=critic_p,
                    tx=optax.adam(learning_rate=self.critic_lr),
                ),
                alpha_state=TrainState.create(
                    apply_fn=None,
                    params={"log_alpha": log_alpha},
                    tx=optax.adam(learning_rate=self.alpha_lr),
                ),
            )

        def stacked_local_obs_all(actor_params_stacked, joint_obs):
            return jnp.stack(
                [
                    jax.vmap(lambda row, i=i: _local_obs_from_joint_row(row, i, per_agent_obs_size))(
                        joint_obs
                    )
                    for i in range(n_agents)
                ],
                axis=0,
            )

        def multi_actor_stochastic(actor_params_stacked, joint_obs, rng):
            loc = stacked_local_obs_all(actor_params_stacked, joint_obs)
            means, log_stds = jax.vmap(actor.apply, in_axes=(0, 0))(actor_params_stacked, loc)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(rng, shape=means.shape)
            actions_ag = nn.tanh(means + stds * noise)
            bsz = joint_obs.shape[0]
            return jnp.swapaxes(actions_ag, 0, 1).reshape(bsz, action_size)

        def multi_actor_mean(actor_params_stacked, joint_obs):
            loc = stacked_local_obs_all(actor_params_stacked, joint_obs)
            means, _ = jax.vmap(actor.apply, in_axes=(0, 0))(actor_params_stacked, loc)
            bsz = joint_obs.shape[0]
            return jnp.swapaxes(nn.tanh(means), 0, 1).reshape(bsz, action_size)

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            actions = multi_actor_mean(training_state.actor_state.params, env_state.obs)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def update_networks(carry, transitions_stacked):
            training_state, key = carry
            agent_start = jax.lax.axis_index(_PMAP_AXIS_NAME) * agents_per_device
            agent_mask = (jnp.arange(agents_per_device) + agent_start) < n_agents
            key, critic_key, actor_key = jax.random.split(key, 3)
            actor_keys_b = jax.random.split(actor_key, agents_per_device)
            critic_keys_b = jax.random.split(critic_key, agents_per_device)
            alpha_vec = training_state.alpha_state.params["log_alpha"]

            def actor_loss_single(actor_p, critic_p, la_i, tr, key_b):
                obs = tr.observation
                state = obs[:, :per_agent_D]
                goal = tr.extras["future_goal"]
                observation = jnp.concatenate([state, goal], axis=-1)
                means, log_stds = actor.apply(actor_p, observation)
                stds = jnp.exp(log_stds)
                x_ts = means + stds * jax.random.normal(key_b, shape=means.shape, dtype=means.dtype)
                action = nn.tanh(x_ts)
                log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
                log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
                log_prob = log_prob.sum(-1)
                sa_encoder_params, g_encoder_params = critic_p["sa_encoder"], critic_p["g_encoder"]
                sa_repr = sa_encoder.apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
                g_repr = g_encoder.apply(g_encoder_params, goal)
                qf_pi = energy_fn(energy_name, sa_repr, g_repr)
                loss = jnp.mean(jnp.exp(la_i) * log_prob - qf_pi)
                return loss, jnp.mean(log_prob)

            def actor_mean_loss(actor_params):
                losses, lp_means = jax.vmap(actor_loss_single, in_axes=(0, 0, 0, 0, 0))(
                    actor_params,
                    training_state.critic_state.params,
                    alpha_vec,
                    transitions_stacked,
                    actor_keys_b,
                )
                mask = agent_mask.astype(losses.dtype)
                denom = jnp.maximum(jnp.sum(mask), 1.0)
                return jnp.sum(losses * mask) / denom, lp_means

            (actor_loss_val, log_prob_means), actor_grad = jax.value_and_grad(actor_mean_loss, has_aux=True)(
                training_state.actor_state.params
            )
            new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

            def alpha_loss_fn(alpha_params):
                av = alpha_params["log_alpha"]
                mask = agent_mask.astype(av.dtype)
                term = jnp.exp(av) * jax.lax.stop_gradient(-log_prob_means - target_entropy)
                return jnp.sum(term * mask) / jnp.maximum(jnp.sum(mask), 1.0)

            alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss_fn)(training_state.alpha_state.params)
            new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)
            training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

            def critic_one(critic_p, tr, key_b):
                del key_b
                state = tr.observation[:, :per_agent_D]
                action = tr.action
                sa_encoder_params, g_encoder_params = critic_p["sa_encoder"], critic_p["g_encoder"]
                sa_repr = sa_encoder.apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
                g_repr = g_encoder.apply(g_encoder_params, tr.observation[:, per_agent_D:])
                logits = energy_fn(energy_name, sa_repr[:, None, :], g_repr[None, :, :])
                cl = contrastive_loss_fn(contrastive_name, logits)
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                return cl + self.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

            def critic_mean_loss(critic_params):
                cls = jax.vmap(critic_one, in_axes=(0, 0, 0))(
                    critic_params, transitions_stacked, critic_keys_b
                )
                mask = agent_mask.astype(cls.dtype)
                return jnp.sum(cls * mask) / jnp.maximum(jnp.sum(mask), 1.0)

            critic_loss_val, critic_grad = jax.value_and_grad(critic_mean_loss)(training_state.critic_state.params)
            new_critic_state = training_state.critic_state.apply_gradients(grads=critic_grad)
            training_state = training_state.replace(
                critic_state=new_critic_state,
                gradient_steps=training_state.gradient_steps + 1,
            )
            metrics = {
                "entropy_per_agent": -log_prob_means,
                "actor_loss": actor_loss_val,
                "alpha_loss": alpha_loss_val,
                "log_alpha_per_agent": new_alpha_state.params["log_alpha"],
                "actor_grad_norm": compute_grad_norm(actor_grad),
                "critic_loss": critic_loss_val,
                "critic_grad_norm": compute_grad_norm(critic_grad),
                "categorical_accuracy": jnp.array(0.0),
                "logits_pos": jnp.array(0.0),
                "logits_neg": jnp.array(0.0),
                "logsumexp": jnp.array(0.0),
            }
            return (training_state, key), metrics

        def training_step(training_state, env_state, buffer_states, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)
            env_state, buffer_states = get_experience(
                training_state.actor_state, env_state, buffer_states, experience_key1
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            def sample_one(bs):
                return replay_buffer.sample(bs)

            buffer_states, tr_local = jax.vmap(sample_one)(buffer_states)
            agent_start = jax.lax.axis_index(_PMAP_AXIS_NAME) * agents_per_device
            transitions_local = []
            for local_i in range(agents_per_device):
                g = agent_start + local_i
                tr = jax.tree_util.tree_map(lambda x: x[local_i], tr_local)
                sk_use = jax.random.split(
                    jax.random.fold_in(sampling_key, g), tr.observation.shape[0]
                )
                tr_flat = jax.vmap(flatten_batch_ma, in_axes=(None, 0, 0))(
                    buffer_config_local, tr, sk_use
                )
                tr_flat = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), tr_flat
                )
                perm = jax.random.permutation(jax.random.fold_in(experience_key2, g), len(tr_flat.observation))
                tr_flat = jax.tree_util.tree_map(lambda x: x[perm], tr_flat)
                tr_flat = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                    tr_flat,
                )
                transitions_local.append(tr_flat)

            transitions_stacked = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *transitions_local
            )
            transitions_stacked = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), transitions_stacked
            )
            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, training_key), transitions_stacked
            )
            return (training_state, env_state, buffer_states), metrics

        def training_epoch(training_state, env_state, buffer_states, key):
            def f(carry, _):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_states, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_states, key),
                (),
                length=num_training_steps_per_epoch,
            )
            return training_state, env_state, buffer_states, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        key, prefill_key = jax.random.split(key, 2)
        prefill_keys = jax.random.split(prefill_key, n_devices)
        training_state, env_state, buffer_states, _ = prefill_replay_buffer(
            training_state, env_state, buffer_states, prefill_keys
        )

        eval_ts_host = _merge_training_state_for_eval(training_state)

        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        def expand_sharded_per_agent_metrics(flat: dict) -> dict:
            out = {}
            for name, value in flat.items():
                if name.endswith("_per_agent") and hasattr(value, "shape"):
                    if len(value.shape) == 2 and value.shape == (n_devices, agents_per_device):
                        per_agent = value.reshape(-1)[:n_agents]
                    elif tuple(value.shape) == (n_agents,):
                        per_agent = value
                    else:
                        out[f"training/{name}"] = value
                        continue
                    short = name[: -len("_per_agent")]
                    for i in range(n_agents):
                        out[f"training/agent_{i}/{short}"] = per_agent[i]
                    out[f"training/{short}_mean"] = jnp.mean(per_agent)
                    if short == "entropy":
                        out["training/entropy"] = out[f"training/{short}_mean"]
                    elif short == "log_alpha":
                        out["training/log_alpha"] = out[f"training/{short}_mean"]
                else:
                    out[f"training/{name}"] = value
            return out

        training_walltime = 0.0
        logging.info("starting training (multi-agent CRL, packed pmap)....")
        params = None
        current_step = 0
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)
            epoch_keys = jax.random.split(epoch_key, n_devices)
            training_state, env_state, buffer_states, metrics = training_epoch(
                training_state, env_state, buffer_states, epoch_keys
            )
            metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            metrics = {k: v for k, v in metrics.items()}
            metrics = expand_sharded_per_agent_metrics(metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            ts_host = _merge_training_state_for_eval(training_state)
            current_step = int(ts_host.env_steps.item())
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": current_step,
                **metrics,
            }
            metrics = evaluator.run_evaluation(ts_host, metrics)
            eval_ts_host = ts_host

            do_render = ne % config.visualization_interval == 0
            merged_actor = ts_host.actor_state.params

            def make_policy(param_stacked):
                def policy(obs, rng):
                    return multi_actor_stochastic(param_stacked, obs, rng)

                return policy

            progress_fn(
                current_step,
                metrics,
                make_policy,
                merged_actor,
                unwrapped_env,
                do_render=do_render,
            )

            if save_path:
                params = (
                    ts_host.alpha_state.params,
                    ts_host.actor_state.params,
                    ts_host.critic_state.params,
                )
                save_params(f"{save_path}/step_{current_step}.pkl", params)

        # assert current_step >= config.total_env_steps
        if params is None:
            params = (
                eval_ts_host.alpha_state.params,
                eval_ts_host.actor_state.params,
                eval_ts_host.critic_state.params,
            )

        def make_policy(param_stacked):
            def policy(obs, rng):
                return multi_actor_stochastic(param_stacked, obs, rng)

            return policy

        brax_pmap.synchronize_hosts()
        return make_policy, params, metrics