"""Random Network Distillation (RND) for intrinsic exploration."""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


class RNDNetwork(nn.Module):
    hidden_dim: int = 256
    output_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(
            self.output_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        return x


class RNDState(NamedTuple):
    target_params: dict
    predictor_state: TrainState
    obs_mean: jnp.ndarray
    obs_var: jnp.ndarray
    obs_count: float
    rew_mean: float
    rew_var: float
    rew_count: float
    return_val: jnp.ndarray


def update_obs_stats(mean, var, count, obs):
    batch_mean = jnp.mean(obs, axis=0)
    batch_var = jnp.var(obs, axis=0)
    batch_count = obs.shape[0]

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = m2 / tot_count
    return new_mean, new_var, tot_count


def normalize_obs(obs, mean, var):
    return (obs - mean) / jnp.sqrt(var + 1e-8)


def compute_intrinsic_reward(target_params, predictor_params, network, norm_obs):
    target_features = network.apply(target_params, norm_obs)
    predictor_features = network.apply(predictor_params, norm_obs)
    return jnp.mean(jnp.square(target_features - predictor_features), axis=-1)


def rnd_step(
    rnd_state: RNDState,
    network: RNDNetwork,
    raw_obs: jnp.ndarray,
    done: jnp.ndarray,
    gamma: float,
):
    """Update RND stats and return normalized intrinsic reward."""
    obs_mean, obs_var, obs_count = update_obs_stats(
        rnd_state.obs_mean, rnd_state.obs_var, rnd_state.obs_count, raw_obs
    )
    norm_obs = normalize_obs(raw_obs, obs_mean, obs_var)

    raw_intrinsic = compute_intrinsic_reward(
        rnd_state.target_params,
        rnd_state.predictor_state.params,
        network,
        norm_obs,
    )

    return_val = rnd_state.return_val * gamma * (1.0 - done) + raw_intrinsic
    batch_mean = jnp.mean(return_val)
    batch_var = jnp.var(return_val)
    batch_count = raw_obs.shape[0]

    delta = batch_mean - rnd_state.rew_mean
    tot_count = rnd_state.rew_count + batch_count

    rew_mean = rnd_state.rew_mean + delta * batch_count / tot_count
    m_a = rnd_state.rew_var * rnd_state.rew_count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * rnd_state.rew_count * batch_count / tot_count
    rew_var = m2 / tot_count

    normalized_intrinsic = raw_intrinsic / jnp.sqrt(rew_var + 1e-8)

    new_rnd_state = rnd_state._replace(
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_count=obs_count,
        rew_mean=rew_mean,
        rew_var=rew_var,
        rew_count=tot_count,
        return_val=return_val,
    )
    return new_rnd_state, normalized_intrinsic, raw_intrinsic


def init_rnd_state(
    rng,
    network: RNDNetwork,
    obs_dim: int,
    num_envs: int,
    lr: float,
    max_grad_norm: float = 1.0,
    dtype=jnp.float32,
):
    rng, target_rng, predictor_rng = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((1, obs_dim), dtype=dtype)
    target_params = network.init(target_rng, dummy_obs)
    predictor_params = network.init(predictor_rng, dummy_obs)
    predictor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    predictor_state = TrainState.create(
        apply_fn=network.apply,
        params=predictor_params,
        tx=predictor_tx,
    )
    return RNDState(
        target_params=target_params,
        predictor_state=predictor_state,
        obs_mean=jnp.zeros((obs_dim,), dtype=dtype),
        obs_var=jnp.ones((obs_dim,), dtype=dtype),
        obs_count=1e-4,
        rew_mean=0.0,
        rew_var=1.0,
        rew_count=1e-4,
        return_val=jnp.zeros((num_envs,), dtype=dtype),
    )


def train_predictor(rnd_state: RNDState, network: RNDNetwork, raw_obs_batch: jnp.ndarray):
    """Train predictor with Adam (via TrainState) on MSE loss against fixed target."""

    def _loss_fn(predictor_params, norm_obs):
        target_features = jax.lax.stop_gradient(
            network.apply(rnd_state.target_params, norm_obs)
        )
        predictor_features = network.apply(predictor_params, norm_obs)
        return jnp.mean(jnp.square(target_features - predictor_features))

    norm_obs = normalize_obs(raw_obs_batch, rnd_state.obs_mean, rnd_state.obs_var)
    loss, grads = jax.value_and_grad(_loss_fn)(
        rnd_state.predictor_state.params, norm_obs
    )
    predictor_state = rnd_state.predictor_state.apply_gradients(grads=grads)
    return rnd_state._replace(predictor_state=predictor_state), loss
