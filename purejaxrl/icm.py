"""Intrinsic Curiosity Module (ICM) for intrinsic exploration (Pathak et al.)."""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


class _MLP(nn.Module):
    hidden_dim: int
    output_dim: int

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


class ICMNetwork(nn.Module):
    """Feature encoder + forward model + inverse model."""

    action_dim: int
    hidden_dim: int = 256
    feature_dim: int = 128

    def setup(self):
        self.encoder = _MLP(self.hidden_dim, self.feature_dim)
        self.forward_model = _MLP(self.hidden_dim, self.feature_dim)
        self.inverse_model = _MLP(self.hidden_dim, self.action_dim)

    def __call__(self, obs, next_obs, action):
        features = self.encoder(obs)
        next_features = self.encoder(next_obs)
        pred_next_features = self.forward_model(
            jnp.concatenate([features, action], axis=-1)
        )
        pred_action = self.inverse_model(
            jnp.concatenate([features, next_features], axis=-1)
        )
        return features, next_features, pred_next_features, pred_action


class ICMState(NamedTuple):
    train_state: TrainState
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


def compute_intrinsic_reward(params, network, norm_obs, norm_next_obs, action):
    _, next_features, pred_next_features, _ = network.apply(
        params, norm_obs, norm_next_obs, action
    )
    next_features = jax.lax.stop_gradient(next_features)
    return jnp.mean(jnp.square(pred_next_features - next_features), axis=-1)


def icm_step(
    icm_state: ICMState,
    network: ICMNetwork,
    raw_obs: jnp.ndarray,
    raw_next_obs: jnp.ndarray,
    action: jnp.ndarray,
    done: jnp.ndarray,
    gamma: float,
):
    """Update ICM stats and return normalized intrinsic reward."""
    obs_mean, obs_var, obs_count = update_obs_stats(
        icm_state.obs_mean, icm_state.obs_var, icm_state.obs_count, raw_obs
    )
    obs_mean, obs_var, obs_count = update_obs_stats(
        obs_mean, obs_var, obs_count, raw_next_obs
    )
    norm_obs = normalize_obs(raw_obs, obs_mean, obs_var)
    norm_next_obs = normalize_obs(raw_next_obs, obs_mean, obs_var)

    raw_intrinsic = compute_intrinsic_reward(
        icm_state.train_state.params,
        network,
        norm_obs,
        norm_next_obs,
        action,
    )

    return_val = icm_state.return_val * gamma * (1.0 - done) + raw_intrinsic
    batch_mean = jnp.mean(return_val)
    batch_var = jnp.var(return_val)
    batch_count = raw_obs.shape[0]

    delta = batch_mean - icm_state.rew_mean
    tot_count = icm_state.rew_count + batch_count

    rew_mean = icm_state.rew_mean + delta * batch_count / tot_count
    m_a = icm_state.rew_var * icm_state.rew_count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * icm_state.rew_count * batch_count / tot_count
    rew_var = m2 / tot_count

    normalized_intrinsic = raw_intrinsic / jnp.sqrt(rew_var + 1e-8)

    new_icm_state = icm_state._replace(
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_count=obs_count,
        rew_mean=rew_mean,
        rew_var=rew_var,
        rew_count=tot_count,
        return_val=return_val,
    )
    return new_icm_state, normalized_intrinsic, raw_intrinsic


def init_icm_state(
    rng,
    network: ICMNetwork,
    obs_dim: int,
    action_dim: int,
    num_envs: int,
    lr: float,
    max_grad_norm: float = 1.0,
    dtype=jnp.float32,
):
    del action_dim  # already encoded in network
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, obs_dim), dtype=dtype)
    dummy_action = jnp.zeros((1, network.action_dim), dtype=dtype)
    params = network.init(init_rng, dummy_obs, dummy_obs, dummy_action)
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    return ICMState(
        train_state=train_state,
        obs_mean=jnp.zeros((obs_dim,), dtype=dtype),
        obs_var=jnp.ones((obs_dim,), dtype=dtype),
        obs_count=1e-4,
        rew_mean=0.0,
        rew_var=1.0,
        rew_count=1e-4,
        return_val=jnp.zeros((num_envs,), dtype=dtype),
    )


def train_icm(
    icm_state: ICMState,
    network: ICMNetwork,
    raw_obs_batch: jnp.ndarray,
    raw_next_obs_batch: jnp.ndarray,
    action_batch: jnp.ndarray,
    beta: float = 0.2,
):
    """Train ICM with Adam on beta * L_forward + (1 - beta) * L_inverse."""

    def _loss_fn(params, norm_obs, norm_next_obs, action):
        _, next_features, pred_next_features, pred_action = network.apply(
            params, norm_obs, norm_next_obs, action
        )
        # Inverse loss trains the encoder; forward loss uses stop-grad on next features.
        forward_loss = jnp.mean(
            jnp.square(pred_next_features - jax.lax.stop_gradient(next_features))
        )
        inverse_loss = jnp.mean(jnp.square(pred_action - action))
        return beta * forward_loss + (1.0 - beta) * inverse_loss

    norm_obs = normalize_obs(raw_obs_batch, icm_state.obs_mean, icm_state.obs_var)
    norm_next_obs = normalize_obs(
        raw_next_obs_batch, icm_state.obs_mean, icm_state.obs_var
    )
    loss, grads = jax.value_and_grad(_loss_fn)(
        icm_state.train_state.params, norm_obs, norm_next_obs, action_batch
    )
    train_state = icm_state.train_state.apply_gradients(grads=grads)
    return icm_state._replace(train_state=train_state), loss
