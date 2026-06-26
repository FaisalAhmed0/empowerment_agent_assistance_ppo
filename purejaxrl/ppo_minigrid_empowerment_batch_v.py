import os
from dataclasses import asdict, dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import tyro
import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
import navix as nx
from wrappers import LogWrapper, NavixGymnaxWrapper


@dataclass
class TrainConfig:
    LR: float = 2.5e-4
    NUM_ENVS: int = 128
    NUM_STEPS: int = 128
    TOTAL_TIMESTEPS: int = 50_000_000
    UPDATE_EPOCHS: int = 1
    NUM_MINIBATCHES: int = 8
    GAMMA: float = 0.99
    GAMMA_CL: float = 0.7
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.3
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTIVATION: str = "relu"
    ENV_NAME: str = "Navix-DoorKey-16x16-v0"
    OBS_EMBED_DIM: int = 16
    OBS_TAG_VOCAB_SIZE: int = 11
    OBS_COLOR_VOCAB_SIZE: int = 6
    OBS_STATE_VOCAB_SIZE: int = 4
    EMPOWERMENT_REPR_DIM: int = 64
    EMPOWERMENT_ENERGY_FN: str = "norm"
    EMPOWERMENT_CONTRASTIVE_LOSS: str = "fwd_infonce"
    EMPOWERMENT_LR: float = 2.5e-4
    EMPOWERMENT_MAX_GRAD_NORM: float = 0.5
    ANNEAL_LR: bool = True
    DEBUG: bool = True
    SEED: int = 30
    RESET_SEED: int = 0
    EVAL_RESET_SEED: int = 0
    WANDB_MODE: str = "online"
    ENTITY: str = ""
    PROJECT: str = "purejaxrl"
    RENDER_ENABLED: bool = True
    RENDER_NUM_STEPS: int = 500
    RENDER_FPS: int = 10
    RENDER_OUT_PATH: str = "artifacts/minigrid_eval.mp4"
    RENDER_DETERMINISTIC: bool = False
    RENDER_NUM_VIDEOS_DURING_TRAINING: int = 0
    TEACHER_HEATMAP_OUT_PATH: str = "artifacts/teacher_softmax.png"
    TEACHER_HEATMAP_EVERY_N_UPDATES: int = 100
    TEACHER_UNIFORM_RANDOM: bool = False
    TEACHER_REWARD_TYPE: str = "competence_lp"
    TEACHER_LP_ABSOLUTE: bool = False
    TEACHER_SUCCESS_BONUS: float = 1.0
    TEACHER_EVAL_EPISODES: int = 1
    TEACHER_EVAL_NUM_ENVS: int = 10
    TEACHER_BATCH_SIZE: int = 512
    TEACHER_LR: float = 2.5e-4
    TEACHER_GAMMA: float = 1.0
    TEACHER_GAE_LAMBDA: float = 0.999
    TEACHER_CLIP_EPS: float = 0.3
    TEACHER_ENT_COEF: float = 0.05
    TEACHER_VF_COEF: float = 0.5
    TEACHER_MAX_GRAD_NORM: float = 0.5
    TEACHER_UPDATE_EPOCHS: int = 2
    TEACHER_NUM_MINIBATCHES: int = 4
    CL_BUFFER_SIZE: int = 128
    GOAL_REWARD_WEIGHT: float = 1.0


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    obs_embed_dim: int = 16
    obs_vocab_sizes: Sequence[int] = (11, 6, 4)

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Goal-conditioned input: the first 3 channels are the symbolic Minigrid
        # observation (integer grid of shape (H, W, 3)); the trailing channel is a
        # one-hot goal map proposed by the teacher (1 at the goal cell, 0 else).
        # Embed each symbolic channel independently with its own table, then
        # append the goal map as a float spatial feature so the CNN trunk sees
        # both the learned observation representation and the goal location.
        symbolic = x[..., :3].astype(jnp.int32)
        goal_map = x[..., 3:].astype(jnp.float32)
        tag_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[0],
            features=self.obs_embed_dim,
            name="tag_embed",
        )(symbolic[..., 0])
        color_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[1],
            features=self.obs_embed_dim,
            name="color_embed",
        )(symbolic[..., 1])
        state_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[2],
            features=self.obs_embed_dim,
            name="state_embed",
        )(symbolic[..., 2])
        x = jnp.concatenate(
            (tag_embed, color_embed, state_embed, goal_map), axis=-1
        )

        embedding = nn.Conv(
            32,
            kernel_size=(2, 2),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        embedding = activation(embedding)
        embedding = nn.Conv(
            64,
            kernel_size=(2, 2),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = activation(embedding)
        # Flatten spatial + channel dims while preserving any leading batch dims
        # (handles both unbatched (H, W, C) and batched (..., H, W, C) inputs).
        embedding = embedding.reshape((*embedding.shape[:-3], -1))
        embedding = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = activation(embedding)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            embedding
        )

        return pi, jnp.squeeze(critic, axis=-1)


class TeacherSpatialPolicy(nn.Module):
    """CNN teacher that proposes a grid location via a spatial categorical head.

    The teacher consumes the same symbolic Minigrid observations as the student
    (shape ``(..., H, W, C)``) and runs a fully-convolutional trunk that keeps the
    spatial resolution fixed (``padding="SAME"``). The final convolution emits a
    single-channel logits map of shape ``(..., H, W, 1)`` -- i.e. one score per
    grid cell, same height/width as the input grid.

    For sampling, the spatial logits are flattened to ``(..., H * W)`` and used to
    parameterize a ``distrax.Categorical`` over flat cell indices. A sampled index
    is later decoded back to ``(row, col)`` coordinates via ``flat_index_to_rowcol``.

    A scalar value head is computed from the shared conv trunk for use as a critic
    in teacher PPO.
    """

    activation: str = "tanh"
    obs_embed_dim: int = 16
    obs_vocab_sizes: Sequence[int] = (11, 6, 4)

    @nn.compact
    def __call__(self, x):
        """Return ``(logits_map, pi, value)``.

        - ``logits_map``: ``(..., H, W, 1)`` per-cell logits aligned to the grid.
        - ``pi``: ``distrax.Categorical`` over the ``H * W`` flattened grid cells.
        - ``value``: ``(...)`` scalar state-value estimate (critic head).
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Match the student's preprocessing so both networks see the same inputs:
        # embed each integer channel independently (with the teacher's own tables)
        # and concatenate before the CNN.
        x = x.astype(jnp.int32)
        tag_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[0],
            features=self.obs_embed_dim,
            name="tag_embed",
        )(x[..., 0])
        color_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[1],
            features=self.obs_embed_dim,
            name="color_embed",
        )(x[..., 1])
        state_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[2],
            features=self.obs_embed_dim,
            name="state_embed",
        )(x[..., 2])
        x = jnp.concatenate((tag_embed, color_embed, state_embed), axis=-1)

        embedding = nn.Conv(
            32,
            kernel_size=(3, 3),
            padding="SAME",
        )(x)
        embedding = activation(embedding)
        embedding = nn.Conv(
            32,
            kernel_size=(3, 3),
            padding="SAME",
        )(embedding)
        embedding = activation(embedding)
        embedding = nn.Conv(
            32,
            kernel_size=(3, 3),
            padding="SAME",
        )(embedding)
        embedding = activation(embedding)

        # Final 1x1 conv collapses to a single output channel while preserving the
        # (H, W) spatial layout, giving one logit per grid cell.
        logits_map = nn.Conv(
            1,
            kernel_size=(3, 3),
            padding="SAME",
        )(embedding)
        # Flatten the spatial grid (drop the trailing singleton channel) into a
        # flat categorical over H * W cells, preserving any leading batch dims.
        flat_logits = logits_map.reshape((*logits_map.shape[:-3], -1))
        pi = distrax.Categorical(logits=flat_logits)

        # VALUE HEAD: flatten the shared conv embedding (spatial + channel dims)
        # and project to a scalar state-value, preserving any leading batch dims.
        critic_embedding = embedding.reshape((*embedding.shape[:-3], -1))
        critic_embedding = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_embedding)
        critic_embedding = activation(critic_embedding)
        value = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic_embedding)

        return logits_map, pi, jnp.squeeze(value, axis=-1)


class EmpowermentCNNEncoder(nn.Module):
    """CNN encoder that maps symbolic Minigrid inputs to a fixed-size vector."""

    repr_dim: int = 64
    activation: str = "tanh"
    obs_embed_dim: int = 16
    obs_vocab_sizes: Sequence[int] = (11, 6, 4)
    conv_channels: Sequence[int] = (32, 64)

    @nn.compact
    def __call__(self, symbolic_obs, extra_channels=None, post_flat_extra=None):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        symbolic_obs = symbolic_obs.astype(jnp.int32)
        tag_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[0],
            features=self.obs_embed_dim,
            name="tag_embed",
        )(symbolic_obs[..., 0])
        color_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[1],
            features=self.obs_embed_dim,
            name="color_embed",
        )(symbolic_obs[..., 1])
        state_embed = nn.Embed(
            num_embeddings=self.obs_vocab_sizes[2],
            features=self.obs_embed_dim,
            name="state_embed",
        )(symbolic_obs[..., 2])
        x = jnp.concatenate((tag_embed, color_embed, state_embed), axis=-1)
        if extra_channels is not None:
            x = jnp.concatenate((x, extra_channels.astype(jnp.float32)), axis=-1)

        for i, channels in enumerate(self.conv_channels):
            x = nn.Conv(
                channels,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"conv_{i}",
            )(x)
            x = activation(x)

        x = x.reshape((*x.shape[:-3], -1))
        if post_flat_extra is not None:
            x = jnp.concatenate((x, post_flat_extra.astype(jnp.float32)), axis=-1)
        x = nn.Dense(
            self.repr_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        return x


class EmpowermentModel(nn.Module):
    """Three-branch empowerment model returning encoder representations."""

    action_dim: int
    repr_dim: int = 64
    activation: str = "tanh"
    obs_embed_dim: int = 16
    obs_vocab_sizes: Sequence[int] = (11, 6, 4)

    @nn.compact
    def __call__(self, obs, goal_map, action, future_obs):
        obs_symbolic = obs[..., :3].astype(jnp.int32)
        future_symbolic = future_obs[..., :3].astype(jnp.int32)

        goal_map = goal_map.astype(jnp.float32)
        if goal_map.ndim == obs_symbolic.ndim - 1:
            goal_map = goal_map[..., None]

        action_one_hot = jax.nn.one_hot(
            action.astype(jnp.int32), self.action_dim, dtype=jnp.float32
        )

        # Encoder A: CNN sees only (obs + goal), then action is concatenated
        # after flattening and before the MLP projection to repr_dim.
        obs_goal_action_repr = EmpowermentCNNEncoder(
            repr_dim=self.repr_dim,
            activation=self.activation,
            obs_embed_dim=self.obs_embed_dim,
            obs_vocab_sizes=self.obs_vocab_sizes,
            name="obs_goal_action_encoder",
        )(obs_symbolic, goal_map, action_one_hot)
        # Encoder B: same conditioning style as the student input
        # (symbolic obs + one-hot spatial goal map).
        obs_goal_repr = EmpowermentCNNEncoder(
            repr_dim=self.repr_dim,
            activation=self.activation,
            obs_embed_dim=self.obs_embed_dim,
            obs_vocab_sizes=self.obs_vocab_sizes,
            name="obs_goal_encoder",
        )(obs_symbolic, goal_map)
        # Encoder C: future observation only (no goal map).
        future_obs_repr = EmpowermentCNNEncoder(
            repr_dim=self.repr_dim,
            activation=self.activation,
            obs_embed_dim=self.obs_embed_dim,
            obs_vocab_sizes=self.obs_vocab_sizes,
            name="future_obs_encoder",
        )(future_symbolic, None)

        return obs_goal_action_repr, obs_goal_repr, future_obs_repr


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class TeacherTransition(NamedTuple):
    """One teacher transition = one student update step on a proposed goal."""

    done: jnp.ndarray
    action: jnp.ndarray  # flat goal-cell index proposed by the teacher
    value: jnp.ndarray
    reward: jnp.ndarray  # empowerment reward for the proposed goal
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # raw symbolic obs the teacher conditioned on


class PendingTeacher(NamedTuple):
    """Per-env goal proposal currently active until the student ends it."""

    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    # The proposed flat goal-cell index of the currently active proposal.
    goal: jnp.ndarray


class EpisodeStore(NamedTuple):
    """Window buffer used for future-state sampling."""

    obs: jnp.ndarray
    goal: jnp.ndarray
    action: jnp.ndarray
    done: jnp.ndarray
    future_obs: jnp.ndarray


def sample_future_states(rng, obs, dones, gamma):
    """Sample one discounted future state per timestep and environment.

    Parameters
    ----------
    rng : jax.random.PRNGKey
    obs : jnp.ndarray, shape (T, N, ...)
        Buffered observations for T timesteps across N environments.
    dones : jnp.ndarray, shape (T, N)
        Terminal flags aligned with ``obs``.
    gamma : float
        Discount factor for geometric sampling over future timesteps.
    """
    max_steps = obs.shape[0]
    num_envs = obs.shape[1]
    all_indices = jnp.arange(max_steps, dtype=jnp.int32)
    env_indices = jnp.arange(num_envs, dtype=jnp.int32)
    rngs = jax.random.split(rng, max_steps)

    dones = dones.astype(bool).at[-1].set(jnp.ones((num_envs,), dtype=bool))
    gamma = jnp.asarray(gamma, dtype=jnp.float32)

    def _sample_one_step(_, inputs):
        i, rng_i = inputs
        # First done at index >= i, per env. Mask positions < i then take the
        # first True via argmax (dones[-1] is forced True so this is well-defined).
        # Avoids dynamic slicing (dones[i:]), which JAX disallows under jit.
        valid_done = dones & (all_indices[:, None] >= i)
        first_done_after_i = jnp.argmax(valid_done, axis=0)
        diff = all_indices - i
        mask = (all_indices[:, None] >= i) & (
            all_indices[:, None] <= first_done_after_i[None, :]
        )
        diff = jnp.maximum(diff, 0)
        probs = jnp.power(gamma, diff.astype(jnp.float32))[:, None]
        probs = jnp.where(mask, probs, 0.0)
        probs_sum = jnp.sum(probs, axis=0, keepdims=True)
        probs = probs / jnp.maximum(probs_sum, 1e-12)

        # jax.random.categorical expects classes on the last axis; transpose to (N, T).
        sampled_t = jax.random.categorical(
            rng_i, jnp.log(jnp.clip(probs.T, a_min=1e-20, a_max=1.0)), axis=-1
        )
        future_obs_i = obs[sampled_t, env_indices]
        return None, future_obs_i

    _, future_obs = jax.lax.scan(
        _sample_one_step,
        None,
        (all_indices, rngs),
    )
    return future_obs


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def flat_index_to_rowcol(flat_index, grid_width):
    """Decode a flat grid-cell index into ``(row, col)`` coordinates.

    The teacher's categorical is defined over a row-major flatten of an
    ``(H, W)`` grid (matching ``reshape(..., -1)`` on the spatial logits), so the
    inverse mapping is ``row = idx // W`` and ``col = idx % W``. Works on scalars
    or arbitrarily-batched integer arrays.
    """
    flat_index = jnp.asarray(flat_index)
    row = flat_index // grid_width
    col = flat_index % grid_width
    return row, col


def make_goal_map(flat_index, grid_h, grid_w):
    """Build a one-hot goal map over the (H, W) grid from a flat cell index.

    ``flat_index`` may be a scalar or arbitrarily-batched integer array indexing
    the row-major flattened ``H * W`` grid (matching the teacher's categorical).
    The returned array has shape ``(..., H, W)`` with a single ``1`` at the goal
    cell and ``0`` elsewhere, preserving any leading batch dims of ``flat_index``.
    """
    one_hot = jax.nn.one_hot(flat_index, grid_h * grid_w)
    return one_hot.reshape(one_hot.shape[:-1] + (grid_h, grid_w))


def condition_obs_on_goal(obs, flat_index, grid_h, grid_w):
    """Append the teacher's one-hot goal map as an extra channel to ``obs``.

    ``obs`` has shape ``(..., H, W, C)``; the returned conditioned observation has
    shape ``(..., H, W, C + 1)`` where the trailing channel is the one-hot goal
    map (cast to the observation dtype so the combined tensor stays homogeneous).
    """
    goal_map = make_goal_map(flat_index, grid_h, grid_w).astype(obs.dtype)
    return jnp.concatenate([obs, goal_map[..., None]], axis=-1)


def gather_goal_cell(obs, flat_index, grid_h, grid_w):
    """Return the symbolic content at the goal cell as shape ``(..., C)``.

    Works for both single ``(H, W, C)`` and batched ``(..., H, W, C)`` obs by
    flattening the spatial grid and gathering along it with ``flat_index`` (which
    must share the leading batch dims of ``obs``). Used to snapshot the goal cell
    when a goal is issued and to detect later changes (goal reached).
    """
    channels = obs.shape[-1]
    obs_flat = obs.reshape(obs.shape[:-3] + (grid_h * grid_w, channels))
    idx = jnp.asarray(flat_index)[..., None, None]
    idx = jnp.broadcast_to(idx, idx.shape[:-1] + (channels,))
    cell = jnp.take_along_axis(obs_flat, idx, axis=-2)
    return cell[..., 0, :]


def sample_teacher_location(pi, grid_width, rng=None, deterministic=False):
    """Sample a grid location from the teacher's flattened categorical.

    Returns ``(flat_index, (row, col), log_prob)`` where ``flat_index`` indexes
    the row-major flattened ``H * W`` grid. When ``deterministic`` is True the
    distribution mode (argmax) is used and ``rng`` may be omitted; otherwise a
    stochastic sample is drawn with ``rng`` (required).
    """
    if deterministic:
        flat_index = pi.mode()
    else:
        if rng is None:
            raise ValueError("rng must be provided when deterministic=False")
        flat_index = pi.sample(seed=rng)
    log_prob = pi.log_prob(flat_index)
    row, col = flat_index_to_rowcol(flat_index, grid_width)
    return flat_index, (row, col), log_prob


def plot_teacher_softmax(
    probs, grid_h, grid_w, *, agent_rowcol=None, title=None, save_path=None
):
    """Heatmap of the teacher's categorical distribution over the (H, W) grid.

    ``probs`` is the flat ``H * W`` softmax probability vector (row-major, matching
    the teacher's flattened logits). It is reshaped to ``(grid_h, grid_w)`` and
    drawn with ``imshow`` so cell ``(row, col)`` maps directly to grid position
    ``(row, col)``. The most-likely cell is annotated, and an optional
    ``agent_rowcol`` marker can be overlaid. Returns ``(fig, ax)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = np.asarray(probs).reshape(grid_h, grid_w)

    fig, ax = plt.subplots(figsize=(6, 6))
    mesh = ax.imshow(probs, cmap="viridis", origin="upper", interpolation="nearest")
    fig.colorbar(mesh, ax=ax, label="P(goal cell)")

    # Annotate the argmax (most likely) cell.
    arg = int(np.argmax(probs))
    best_row, best_col = arg // grid_w, arg % grid_w
    ax.scatter(
        best_col,
        best_row,
        s=180,
        marker="x",
        c="tab:red",
        linewidths=2.0,
        zorder=3,
        label="argmax cell",
    )

    if agent_rowcol is not None:
        agent_rowcol = np.asarray(agent_rowcol).reshape(-1)
        ax.scatter(
            agent_rowcol[1],
            agent_rowcol[0],
            s=200,
            marker="*",
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
            label="agent",
        )

    ax.set_xticks(np.arange(grid_w))
    ax.set_yticks(np.arange(grid_h))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.legend(loc="upper right", framealpha=0.7)
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def save_teacher_softmax_snapshot(
    probs,
    grid_h,
    grid_w,
    out_path,
    *,
    title="Teacher goal-cell distribution",
    wandb_step=None,
    wandb_key="teacher/goal_softmax",
):
    """Save and optionally log a teacher goal-softmax heatmap snapshot."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fig, _ = plot_teacher_softmax(
        probs,
        grid_h,
        grid_w,
        title=title,
        save_path=out_path,
    )
    print(f"[teacher] saved softmax heatmap to {out_path}")
    if wandb_step is not None:
        wandb.log({wandb_key: wandb.Image(fig)}, step=int(wandb_step))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


def overlay_goal_on_frames(frames, goal_indices, grid_h, grid_w):
    """Highlight the teacher goal cell on each rendered RGB frame.

    Draws a semi-transparent gold fill over the goal tile and a solid
    orange-red border around it so the selected cell is immediately visible
    in saved evaluation videos.

    Parameters
    ----------
    frames : np.ndarray, shape (T, H_px, W_px, 3), dtype uint8
        RGB frames produced by ``nx.observations.rgb``.
    goal_indices : array-like, shape (T,)
        Flat row-major grid-cell indices (the teacher's ``goal_idx`` at each
        timestep).
    grid_h, grid_w : int
        Number of grid cells along height and width axes respectively.

    Returns
    -------
    np.ndarray
        A copy of ``frames`` with the goal cell highlighted on every frame.
    """
    import numpy as _np

    goal_indices = _np.asarray(goal_indices)
    frames = frames.copy()
    T, H_px, W_px = frames.shape[:3]
    # Use exact partition edges so overlay stays correct even if frame size is
    # not perfectly divisible by grid resolution.
    row_edges = _np.linspace(0, H_px, grid_h + 1, dtype=_np.int32)
    col_edges = _np.linspace(0, W_px, grid_w + 1, dtype=_np.int32)
    tile_h = max(1, int(_np.median(_np.diff(row_edges))))
    tile_w = max(1, int(_np.median(_np.diff(col_edges))))

    highlight_color = _np.array([255, 215, 0], dtype=_np.float32)  # gold
    alpha = 0.4
    border_color = _np.array([255, 69, 0], dtype=_np.uint8)  # orange-red
    border_px = max(2, min(tile_h, tile_w) // 10)

    for t in range(T):
        row = int(goal_indices[t]) // grid_w
        col = int(goal_indices[t]) % grid_w
        r0, r1 = row_edges[row], row_edges[row + 1]
        c0, c1 = col_edges[col], col_edges[col + 1]
        # Alpha-blend a gold fill over the goal tile.
        tile = frames[t, r0:r1, c0:c1].astype(_np.float32)
        frames[t, r0:r1, c0:c1] = _np.clip(
            tile * (1 - alpha) + highlight_color * alpha, 0, 255
        ).astype(_np.uint8)
        # Draw an orange-red border around the tile.
        frames[t, r0 : r0 + border_px, c0:c1] = border_color
        frames[t, r1 - border_px : r1, c0:c1] = border_color
        frames[t, r0:r1, c0 : c0 + border_px] = border_color
        frames[t, r0:r1, c1 - border_px : c1] = border_color

    return frames


def plot_render_episode_returns(
    rewards, dones, *, title=None, save_path=None, linewidth=2.0, marker="o"
):
    """Plot per-episode returns for one rendered rollout.

    Parameters
    ----------
    rewards : array-like, shape (T,)
        Step rewards from the rendered rollout.
    dones : array-like, shape (T,)
        Episode-terminal flags aligned with ``rewards``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones).reshape(-1).astype(bool)
    assert rewards.shape[0] == dones.shape[0], "rewards/dones length mismatch"

    episode_returns = []
    running_return = 0.0
    for r, d in zip(rewards, dones):
        running_return += float(r)
        if d:
            episode_returns.append(running_return)
            running_return = 0.0
    if not episode_returns:
        episode_returns.append(float(rewards.sum()))

    x = np.arange(1, len(episode_returns) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, episode_returns, linewidth=linewidth, marker=marker)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Episode return")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax, episode_returns


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config.setdefault("GAMMA_CL", 0.99)
    config.setdefault("CL_BUFFER_SIZE", config["NUM_STEPS"])
    config.setdefault("EMPOWERMENT_ENERGY_FN", "l2")
    config.setdefault("EMPOWERMENT_CONTRASTIVE_LOSS", "fwd_infonce")
    config.setdefault("EMPOWERMENT_LR", 2.5e-4)
    config.setdefault("EMPOWERMENT_MAX_GRAD_NORM", 0.5)
    assert config["CL_BUFFER_SIZE"] > 0, "CL_BUFFER_SIZE must be positive"
    assert (
        config["CL_BUFFER_SIZE"] % config["NUM_STEPS"] == 0
    ), "CL_BUFFER_SIZE must be divisible by NUM_STEPS"
    print(f"Number of updates: {config['NUM_UPDATES']}")
    # Teacher PPO + empowerment reward settings. The teacher proposes a goal
    # cell; its reward is the empowerment reward computed after the rollout from
    # the empowerment encoders (energy difference relative to the achieved future
    # state). One teacher transition is recorded per proposal-end event (done OR
    # goal reached); the teacher PPO update runs once TEACHER_BATCH_SIZE
    # transitions accumulate. The legacy competence/LP knobs below are kept only
    # for CLI backward-compatibility and are no longer used by the reward.
    config.setdefault("TEACHER_REWARD_TYPE", "empowerment")
    config.setdefault("TEACHER_LP_ABSOLUTE", False)
    config.setdefault("TEACHER_SUCCESS_BONUS", 1.0)
    config.setdefault("TEACHER_EVAL_EPISODES", 1)
    config.setdefault("TEACHER_EVAL_NUM_ENVS", 4)
    config.setdefault("TEACHER_BATCH_SIZE", 512)
    config.setdefault("TEACHER_LR", 2.5e-4)
    config.setdefault("TEACHER_GAMMA", 0.99)
    config.setdefault("TEACHER_GAE_LAMBDA", 0.95)
    config.setdefault("TEACHER_CLIP_EPS", 0.2)
    config.setdefault("TEACHER_ENT_COEF", 0.01)
    config.setdefault("TEACHER_VF_COEF", 0.5)
    config.setdefault("TEACHER_MAX_GRAD_NORM", 0.5)
    config.setdefault("TEACHER_UPDATE_EPOCHS", 4)
    config.setdefault("TEACHER_NUM_MINIBATCHES", 4)
    # Read grid dims from a throwaway env reset before building the real env.
    _probe_env, env_params = NavixGymnaxWrapper(config["ENV_NAME"]), None
    _probe_obsv, _ = _probe_env.reset(jax.random.PRNGKey(0), env_params)
    print("shape of symbolic observation (kept spatial for CNN)")
    jax.debug.print("obsv.shape: {obsv}", obsv=_probe_obsv.shape)
    teacher_grid_h, teacher_grid_w = int(_probe_obsv.shape[0]), int(_probe_obsv.shape[1])
    config["TEACHER_GRID_H"] = teacher_grid_h
    config["TEACHER_GRID_W"] = teacher_grid_w
    config["TEACHER_NUM_CELLS"] = teacher_grid_h * teacher_grid_w
    # Fixed-horizon episode length: disable early success termination so all
    # envs end together every H = 4 * grid_h * grid_w steps, giving a stable
    # episode horizon for the student.
    config["FIXED_EPISODE_LENGTH"] = 4 * teacher_grid_h * teacher_grid_w
    # Default eval horizon to one full fixed episode; can be overridden in config.
    config.setdefault("TEACHER_EVAL_HORIZON", config["FIXED_EPISODE_LENGTH"])
    assert config["TEACHER_BATCH_SIZE"] > 0, "TEACHER_BATCH_SIZE must be > 0"
    assert (
        config["TEACHER_BATCH_SIZE"] % config["TEACHER_NUM_MINIBATCHES"] == 0
    ), "teacher batch size must be divisible by TEACHER_NUM_MINIBATCHES"
    teacher_uniform_random = bool(config.get("TEACHER_UNIFORM_RANDOM", False))
    teacher_num_cells = int(config["TEACHER_NUM_CELLS"])
    env = NavixGymnaxWrapper(
        config["ENV_NAME"],
        max_steps=config["FIXED_EPISODE_LENGTH"],
        disable_termination=True,
    )
    env = LogWrapper(env)

    def render_eval_episode(
        params, rng, teacher_params, out_path=None, num_steps=None, fps=None,
        deterministic=None
    ):
        """Roll out the trained policy and save a video of the agent.

        The policy consumes the same flattened symbolic observations it was
        trained on, while frames are produced from the underlying Navix state via
        ``nx.observations.rgb`` so the saved video shows the full grid.

        Actions are sampled stochastically from the policy distribution by
        default. Set ``deterministic=True`` (or ``config["RENDER_DETERMINISTIC"]``)
        to instead take the distribution mode (argmax) at every step.
        """
        num_steps = (
            int(config.get("RENDER_NUM_STEPS", 200)) if num_steps is None else int(num_steps)
        )
        fps = int(config.get("RENDER_FPS", 10)) if fps is None else int(fps)
        if out_path is None:
            out_path = config.get("RENDER_OUT_PATH", "artifacts/minigrid_eval.mp4")
        if deterministic is None:
            deterministic = bool(config.get("RENDER_DETERMINISTIC", False))

        network = ActorCritic(
            env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            obs_embed_dim=config["OBS_EMBED_DIM"],
            obs_vocab_sizes=(
                config["OBS_TAG_VOCAB_SIZE"],
                config["OBS_COLOR_VOCAB_SIZE"],
                config["OBS_STATE_VOCAB_SIZE"],
            ),
        )
        teacher_network = TeacherSpatialPolicy(
            activation=config["ACTIVATION"],
            obs_embed_dim=config["OBS_EMBED_DIM"],
            obs_vocab_sizes=(
                config["OBS_TAG_VOCAB_SIZE"],
                config["OBS_COLOR_VOCAB_SIZE"],
                config["OBS_STATE_VOCAB_SIZE"],
            ),
        )
        grid_h, grid_w = config["TEACHER_GRID_H"], config["TEACHER_GRID_W"]

        def _sample_goal(obs, rng):
            if teacher_uniform_random:
                flat_shape = obs.shape[:-3]
                return jax.random.randint(
                    rng,
                    shape=flat_shape,
                    minval=0,
                    maxval=teacher_num_cells,
                    dtype=jnp.int32,
                )
            _, pi, _ = teacher_network.apply(teacher_params, obs)
            return pi.sample(seed=rng)

        def _rollout(params, rng):
            # Always start the eval rollout from the same fixed state, and reuse
            # that exact reset on every episode boundary so each episode in the
            # rendered video begins from the identical start state.
            reset_rng = jax.random.PRNGKey(
                config.get("EVAL_RESET_SEED", config["RESET_SEED"])
            )
            fixed_reset_obs, fixed_reset_state = env.reset(reset_rng, env_params)
            obs, state = fixed_reset_obs, fixed_reset_state
            rng, goal_rng = jax.random.split(rng)
            goal_idx = _sample_goal(obs, goal_rng)
            goal_ref = gather_goal_cell(obs, goal_idx, grid_h, grid_w)

            def _step(carry, _):
                obs, state, goal_idx, goal_ref, rng = carry
                cond_obs = condition_obs_on_goal(obs, goal_idx, grid_h, grid_w)
                pi, _ = network.apply(params, cond_obs)
                rng, sample_rng, step_rng, teacher_rng = jax.random.split(rng, 4)
                if deterministic:
                    action = pi.mode()
                else:
                    action = pi.sample(seed=sample_rng)
                obs, state, reward, done, info = env.step(
                    step_rng, state, action, env_params
                )

                # Override Navix's varying internal auto-reset so every episode
                # restarts from the same fixed state.
                obs = jax.tree_util.tree_map(
                    lambda r, x: jnp.where(done, r, x), fixed_reset_obs, obs
                )
                state = jax.tree_util.tree_map(
                    lambda r, x: jnp.where(done, r, x), fixed_reset_state, state
                )

                # Resample the teacher goal only on goal completion (the goal cell
                # content changed vs. the snapshot) or episode end.
                current_cell = gather_goal_cell(obs, goal_idx, grid_h, grid_w)
                goal_reached = jnp.any(jnp.not_equal(current_cell, goal_ref))
                resample = jnp.logical_or(done, goal_reached)
                new_idx = _sample_goal(obs, teacher_rng)
                goal_idx = jnp.where(resample, new_idx, goal_idx)
                new_ref = gather_goal_cell(obs, goal_idx, grid_h, grid_w)
                goal_ref = jnp.where(resample, new_ref, goal_ref)
                return (obs, state, goal_idx, goal_ref, rng), (
                    state,
                    goal_idx,
                    reward,
                    done,
                )

            _, (states, goal_indices, rewards, dones) = jax.lax.scan(
                _step, (obs, state, goal_idx, goal_ref, rng), None, length=num_steps
            )
            # `states` is the LogEnvState stacked over time; the wrapped Navix
            # timestep (and its renderable state) lives in `env_state`.
            # `goal_indices` is the flat goal cell index at each timestep.
            frames = jax.vmap(nx.observations.rgb)(states.env_state.state)
            return frames, goal_indices, rewards, dones

        frames, goal_indices, rewards, dones = jax.jit(_rollout)(params, rng)
        frames = np.asarray(jax.device_get(frames)).astype(np.uint8)
        goal_indices = np.asarray(jax.device_get(goal_indices))
        rewards = np.asarray(jax.device_get(rewards))
        dones = np.asarray(jax.device_get(dones))
        frames = overlay_goal_on_frames(frames, goal_indices, grid_h, grid_w)
        unique_goals = np.unique(goal_indices).size
        goal_rows = (goal_indices // grid_w).astype(np.int32)
        goal_cols = (goal_indices % grid_w).astype(np.int32)
        print(
            "[render_eval_episode] goal trace: "
            f"{unique_goals} unique cells over {len(goal_indices)} steps; "
            f"first 10 (row,col)={list(zip(goal_rows[:10], goal_cols[:10]))}"
        )
        if unique_goals <= 1:
            print(
                "[render_eval_episode] warning: sampled goal stayed in one cell. "
                "If unexpected, inspect teacher softmax entropy and render seed."
            )

        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        try:
            import imageio.v2 as imageio

            if out_path.endswith(".gif"):
                imageio.mimsave(out_path, list(frames), duration=1.0 / max(fps, 1))
            else:
                imageio.mimsave(out_path, list(frames), fps=fps, macro_block_size=1)
        except Exception as err:  # pragma: no cover - fallback path
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            gif_path = os.path.splitext(out_path)[0] + ".gif"
            fig = plt.figure()
            plt.axis("off")
            im = plt.imshow(frames[0])

            def _update(i):
                im.set_array(frames[i])
                return (im,)

            anim = animation.FuncAnimation(
                fig, _update, frames=len(frames), interval=1000 / max(fps, 1), blit=True
            )
            anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
            plt.close(fig)
            out_path = gif_path
            print(f"[render_eval_episode] imageio failed ({err}); used matplotlib fallback")

        returns_plot_path = os.path.splitext(out_path)[0] + "_episode_returns.png"
        ret_fig, _, episode_returns = plot_render_episode_returns(
            rewards,
            dones,
            title="Rendered rollout episode returns",
            save_path=returns_plot_path,
        )
        import matplotlib.pyplot as plt

        plt.close(ret_fig)
        print(
            f"[render_eval_episode] saved agent video to {out_path} ({len(frames)} frames)"
        )
        print(
            "[render_eval_episode] saved episode-return plot to "
            f"{returns_plot_path} (episodes={len(episode_returns)})"
        )
        return out_path

    def linear_schedule(count):
        update_idx = count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        # Make the student LR linear from LR at update 0 to exactly 0 at the
        # last training update (index NUM_UPDATES - 1).
        if config["NUM_UPDATES"] <= 1:
            frac = 0.0
        else:
            frac = 1.0 - (update_idx / (config["NUM_UPDATES"] - 1))
            frac = jnp.maximum(frac, 0.0)
        return config["LR"] * frac

    def teacher_linear_schedule(count):
        # The teacher updates after accumulating TEACHER_BATCH_SIZE event
        # transitions (done OR goal_reached). Use an upper-bound estimate from
        # total student env-steps for LR annealing.
        total_student_env_steps = (
            int(config["NUM_UPDATES"])
            * int(config["NUM_STEPS"])
            * int(config["NUM_ENVS"])
        )
        n = max(total_student_env_steps // int(config["TEACHER_BATCH_SIZE"]), 1)
        teacher_steps_per_update = (
            config["TEACHER_NUM_MINIBATCHES"] * config["TEACHER_UPDATE_EPOCHS"]
        )
        total_teacher_opt_steps = max(n * teacher_steps_per_update, 1)
        # Make the teacher LR linear from TEACHER_LR at optimizer step 0 to
        # exactly 0 at the final planned teacher optimizer step.
        if total_teacher_opt_steps <= 1:
            frac = 0.0
        else:
            frac = 1.0 - (count / (total_teacher_opt_steps - 1))
            frac = jnp.clip(frac, 0.0, 1.0)
        return config["TEACHER_LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            obs_embed_dim=config["OBS_EMBED_DIM"],
            obs_vocab_sizes=(
                config["OBS_TAG_VOCAB_SIZE"],
                config["OBS_COLOR_VOCAB_SIZE"],
                config["OBS_STATE_VOCAB_SIZE"],
            ),
        )
        rng, _rng = jax.random.split(rng)
        # The student consumes goal-conditioned observations: the symbolic grid
        # plus one extra channel holding the teacher's one-hot goal map.
        obs_shape = env.observation_space(env_params).shape
        init_x = jnp.zeros(obs_shape[:-1] + (obs_shape[-1] + 1,))
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

        # INIT TEACHER
        # The teacher is a CNN that proposes a goal cell over the (H, W) grid and
        # is trained with its own PPO loop. Its reward is the empowerment reward
        # computed after each rollout from the empowerment encoders (the energy
        # difference between the obs+goal+action and obs+goal encoders relative to
        # the achieved future state).
        grid_h = config["TEACHER_GRID_H"]
        grid_w = config["TEACHER_GRID_W"]
        teacher_gamma = config["TEACHER_GAMMA"]
        teacher_gae_lambda = config["TEACHER_GAE_LAMBDA"]
        teacher_clip_eps = config["TEACHER_CLIP_EPS"]
        teacher_vf_coef = config["TEACHER_VF_COEF"]
        teacher_ent_coef = config["TEACHER_ENT_COEF"]
        teacher_num_minibatches = config["TEACHER_NUM_MINIBATCHES"]
        teacher_update_epochs = config["TEACHER_UPDATE_EPOCHS"]
        teacher_batch_size = config["TEACHER_BATCH_SIZE"]
        teacher_lr = config["TEACHER_LR"]
        goal_reward_weight = config["GOAL_REWARD_WEIGHT"]
        gamma_cl = config["GAMMA_CL"]
        cl_buffer_size = int(config["CL_BUFFER_SIZE"])
        empowerment_energy_name = config["EMPOWERMENT_ENERGY_FN"]
        empowerment_loss_name = config["EMPOWERMENT_CONTRASTIVE_LOSS"]
        empowerment_lr = config["EMPOWERMENT_LR"]
        empowerment_max_grad_norm = config["EMPOWERMENT_MAX_GRAD_NORM"]
        teacher_network = TeacherSpatialPolicy(
            activation=config["ACTIVATION"],
            obs_embed_dim=config["OBS_EMBED_DIM"],
            obs_vocab_sizes=(
                config["OBS_TAG_VOCAB_SIZE"],
                config["OBS_COLOR_VOCAB_SIZE"],
                config["OBS_STATE_VOCAB_SIZE"],
            ),
        )
        rng, _teacher_rng = jax.random.split(rng)
        teacher_init_x = jnp.zeros(env.observation_space(env_params).shape)
        teacher_params = teacher_network.init(_teacher_rng, teacher_init_x)
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
            params=teacher_params,
            tx=teacher_tx,
        )

        # INIT EMPOWERMENT ENCODERS.
        empowerment_network = EmpowermentModel(
            action_dim=env.action_space(env_params).n,
            repr_dim=config["EMPOWERMENT_REPR_DIM"],
            activation=config["ACTIVATION"],
            obs_embed_dim=config["OBS_EMBED_DIM"],
            obs_vocab_sizes=(
                config["OBS_TAG_VOCAB_SIZE"],
                config["OBS_COLOR_VOCAB_SIZE"],
                config["OBS_STATE_VOCAB_SIZE"],
            ),
        )
        rng, _emp_rng = jax.random.split(rng)
        emp_init_obs = jnp.zeros(env.observation_space(env_params).shape)
        emp_init_goal = jnp.zeros(obs_shape[:-1] + (1,), dtype=jnp.float32)
        emp_init_action = jnp.zeros((), dtype=jnp.int32)
        emp_init_future_obs = jnp.zeros(env.observation_space(env_params).shape)
        empowerment_params = empowerment_network.init(
            _emp_rng,
            emp_init_obs,
            emp_init_goal,
            emp_init_action,
            emp_init_future_obs,
        )
        emp_repr_oga, emp_repr_og, emp_repr_future = empowerment_network.apply(
            empowerment_params,
            emp_init_obs,
            emp_init_goal,
            emp_init_action,
            emp_init_future_obs,
        )
        expected_repr_shape = (config["EMPOWERMENT_REPR_DIM"],)
        if (
            emp_repr_oga.shape != expected_repr_shape
            or emp_repr_og.shape != expected_repr_shape
            or emp_repr_future.shape != expected_repr_shape
        ):
            raise ValueError(
                "Empowerment encoder output shape mismatch: "
                f"got {(emp_repr_oga.shape, emp_repr_og.shape, emp_repr_future.shape)}, "
                f"expected {expected_repr_shape} for each branch."
            )
        empowerment_tx = optax.chain(
            optax.clip_by_global_norm(empowerment_max_grad_norm),
            optax.adam(empowerment_lr, eps=1e-5),
        )
        empowerment_train_state = TrainState.create(
            apply_fn=empowerment_network.apply,
            params=empowerment_params,
            tx=empowerment_tx,
        )
        # Validate configured contrastive components once before JIT.
        _dummy_repr = jnp.zeros((2, config["EMPOWERMENT_REPR_DIM"]), dtype=jnp.float32)
        _dummy_logits = energy_fn(
            empowerment_energy_name, _dummy_repr[:, None, :], _dummy_repr[None, :, :]
        )
        if _dummy_logits.shape != (2, 2):
            raise ValueError(
                f"Empowerment energy_fn must produce pairwise logits of shape (2, 2), got {_dummy_logits.shape}."
            )
        _dummy_loss = contrastive_loss_fn(empowerment_loss_name, _dummy_logits)
        if _dummy_loss.shape != ():
            raise ValueError(
                f"Empowerment contrastive_loss_fn must return a scalar, got shape {_dummy_loss.shape}."
            )

        def teacher_apply(params, obs):
            """Apply the teacher CNN, returning ``(logits_map, pi, value)``."""
            return teacher_network.apply(params, obs)

        def teacher_sample(params, obs, rng=None, deterministic=False):
            """Sample a goal cell from the teacher for the given observation(s).

            Returns ``(flat_index, (row, col), log_prob, value)`` over the
            (H, W) grid, where ``value`` is the teacher critic's estimate.
            """
            if teacher_uniform_random:
                flat_shape = obs.shape[:-3]
                if deterministic:
                    flat_index = jnp.zeros(flat_shape, dtype=jnp.int32)
                else:
                    if rng is None:
                        raise ValueError(
                            "rng must be provided for stochastic uniform teacher sampling"
                        )
                    flat_index = jax.random.randint(
                        rng,
                        shape=flat_shape,
                        minval=0,
                        maxval=teacher_num_cells,
                        dtype=jnp.int32,
                    )
                row, col = flat_index_to_rowcol(flat_index, config["TEACHER_GRID_W"])
                log_prob = jnp.full(
                    flat_shape,
                    -jnp.log(jnp.asarray(teacher_num_cells, dtype=jnp.float32)),
                    dtype=jnp.float32,
                )
                value = jnp.zeros(flat_shape, dtype=jnp.float32)
                return flat_index, (row, col), log_prob, value

            _, pi, value = teacher_network.apply(params, obs)
            flat_index, (row, col), log_prob = sample_teacher_location(
                pi, config["TEACHER_GRID_W"], rng=rng, deterministic=deterministic
            )
            return flat_index, (row, col), log_prob, value

        # INIT ENV
        # Use one fixed reset key, broadcast across all envs, so every env starts
        # from the exact same initial state. The same key is reused on every
        # episode reset (see _env_step) to keep start states identical.
        reset_key = jax.random.PRNGKey(config["RESET_SEED"])
        reset_rng = jnp.broadcast_to(
            reset_key, (config["NUM_ENVS"],) + reset_key.shape
        )
        fixed_reset_obsv, fixed_reset_env_state = jax.vmap(
            env.reset, in_axes=(0, None)
        )(reset_rng, env_params)
        obsv, env_state = fixed_reset_obsv, fixed_reset_env_state

        def condition(obs, flat_index):
            """Append the current teacher goal map to ``obs`` for the student."""
            return condition_obs_on_goal(obs, flat_index, grid_h, grid_w)

        # INIT TEACHER PROPOSAL + BUFFER
        # Sample one goal cell per env from the teacher on the fixed reset obs and
        # snapshot the goal cell content (for reach detection). One teacher
        # transition is recorded for each env event (done OR goal_reached); the
        # teacher PPO update runs once ``TEACHER_BATCH_SIZE`` transitions
        # accumulate. Each transition's reward is the empowerment reward computed
        # after the rollout (see the post-rollout block in ``_update_step``).
        rng, _goal_rng = jax.random.split(rng)
        init_goal_idx, _, init_log_prob, init_value = teacher_sample(
            teacher_train_state.params, obsv, rng=_goal_rng
        )
        goal_ref = gather_goal_cell(obsv, init_goal_idx, grid_h, grid_w)
        pending = PendingTeacher(
            obs=obsv,
            action=init_goal_idx,
            value=init_value,
            log_prob=init_log_prob,
            goal=init_goal_idx,
        )
        teacher_obs_shape = env.observation_space(env_params).shape
        # Flat event buffer with one padding slot so out-of-range writes (events
        # beyond capacity) can be safely absorbed and later discarded.
        _tbuf_rows = teacher_batch_size + 1
        teacher_buffer = TeacherTransition(
            done=jnp.zeros((_tbuf_rows,)),
            action=jnp.zeros((_tbuf_rows,), dtype=jnp.int32),
            value=jnp.zeros((_tbuf_rows,)),
            reward=jnp.zeros((_tbuf_rows,)),
            log_prob=jnp.zeros((_tbuf_rows,)),
            obs=jnp.zeros((_tbuf_rows,) + teacher_obs_shape),
        )
        teacher_buffer_count = jnp.array(0, dtype=jnp.int32)
        cl_buffer = EpisodeStore(
            obs=jnp.zeros(
                (cl_buffer_size, config["NUM_ENVS"]) + teacher_obs_shape,
                dtype=fixed_reset_obsv.dtype,
            ),
            goal=jnp.zeros(
                (cl_buffer_size, config["NUM_ENVS"], grid_h, grid_w),
                dtype=fixed_reset_obsv.dtype,
            ),
            action=jnp.zeros((cl_buffer_size, config["NUM_ENVS"]), dtype=jnp.int32),
            done=jnp.zeros((cl_buffer_size, config["NUM_ENVS"]), dtype=bool),
            future_obs=jnp.zeros(
                (cl_buffer_size, config["NUM_ENVS"]) + teacher_obs_shape,
                dtype=fixed_reset_obsv.dtype,
            ),
        )
        cl_buffer_ptr = jnp.array(0, dtype=jnp.int32)
        completed_episode = jax.tree_util.tree_map(jnp.zeros_like, cl_buffer)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            (
                train_state,
                teacher_train_state,
                empowerment_train_state,
                env_state,
                last_obs,
                pending,
                goal_ref,
                success_since_boundary,
                teacher_buffer,
                teacher_buffer_count,
                cl_buffer,
                cl_buffer_ptr,
                completed_episode,
                rng,
            ) = runner_state

            # COLLECT TRAJECTORIES
            # goal_idx resamples on episode-done OR teacher-goal-reached so the
            # student always has an active goal to chase. pending tracks the
            # currently active teacher proposal per env and rolls on each event;
            # one teacher transition is emitted per proposal-end event.
            def _env_step(inner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    goal_idx,
                    goal_ref,
                    pending,
                    success_since_boundary,
                    rng,
                ) = inner_state

                # Condition the student on the current teacher goal by appending
                # the one-hot goal map as an extra observation channel.
                cond_obs = condition(last_obs, goal_idx)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, cond_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, env_reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                student_task_success_step = jnp.equal(env_reward, 1)
                success_since_boundary = jnp.logical_or(
                    success_since_boundary, student_task_success_step
                )

                # Force every episode reset back to the exact same start state.
                # With disabled termination the episode ends only at max_steps,
                # so all envs reset together here.
                def _select_on_done(reset_leaf, leaf):
                    mask = jnp.reshape(
                        done, (done.shape[0],) + (1,) * (leaf.ndim - 1)
                    )
                    return jnp.where(mask, reset_leaf, leaf)

                obsv = jax.tree_util.tree_map(
                    _select_on_done, fixed_reset_obsv, obsv
                )
                reset_inner_state = jax.tree_util.tree_map(
                    _select_on_done,
                    fixed_reset_env_state.env_state,
                    env_state.env_state,
                )
                env_state = env_state.replace(env_state=reset_inner_state)

                # GOAL REACHED CHECK
                # The goal is "reached" when the symbolic content at the goal cell
                # differs from the snapshot taken when the goal was issued.
                current_cell = gather_goal_cell(obsv, goal_idx, grid_h, grid_w)
                goal_reached_mask = jnp.any(
                    jnp.not_equal(current_cell, goal_ref), axis=-1
                )
                goal_reached = goal_reward_weight * goal_reached_mask.astype(
                    env_reward.dtype
                )
                # The student keeps its task reward plus the goal-reaching bonus.
                reward = 0*env_reward + goal_reached
                transition = Transition(
                    done, action, value, reward, log_prob, cond_obs, info
                )

                # SAMPLE NEXT TEACHER PROPOSAL from the current (frozen) teacher.
                # Used for the student resample and for rolling the pending
                # proposal whenever the current proposal ends (done OR reached).
                rng, _t_sample_rng = jax.random.split(rng)
                new_t_goal_idx, _, new_t_log_prob, new_t_value = teacher_sample(
                    teacher_train_state.params, obsv, rng=_t_sample_rng
                )

                # PROPOSAL-END EVENT: the active teacher proposal terminates when
                # the student reaches its goal or the episode ends. One teacher
                # transition is recorded per event; its empowerment reward is
                # computed after the rollout, once future states are available.
                done_mask = done.astype(bool)
                event_mask = jnp.logical_or(done_mask, goal_reached_mask)

                # Capture the ending proposal's teacher PPO fields (the proposal
                # active during this step) before rolling to the new proposal.
                te_obs = pending.obs
                te_action = pending.action
                te_value = pending.value
                te_log_prob = pending.log_prob

                # RESAMPLE STUDENT GOAL on proposal-end events.
                goal_idx = jnp.where(event_mask, new_t_goal_idx, goal_idx)
                new_ref = gather_goal_cell(obsv, goal_idx, grid_h, grid_w)
                goal_ref = jnp.where(event_mask[:, None], new_ref, goal_ref)

                # Reset the per-proposal task-success tracker on events.
                success_since_boundary = jnp.where(
                    event_mask,
                    jnp.zeros_like(success_since_boundary),
                    success_since_boundary,
                )

                # ROLL PENDING PROPOSAL: adopt the newly sampled teacher proposal
                # for the envs whose proposal ended this step.
                pending = PendingTeacher(
                    obs=jnp.where(
                        event_mask[:, None, None, None], obsv, pending.obs
                    ),
                    action=jnp.where(event_mask, new_t_goal_idx, pending.action),
                    value=jnp.where(event_mask, new_t_value, pending.value),
                    log_prob=jnp.where(
                        event_mask, new_t_log_prob, pending.log_prob
                    ),
                    goal=jnp.where(event_mask, new_t_goal_idx, pending.goal),
                )

                inner_state = (
                    train_state,
                    env_state,
                    obsv,
                    goal_idx,
                    goal_ref,
                    pending,
                    success_since_boundary,
                    rng,
                )
                return inner_state, (
                    transition,
                    goal_reached,
                    event_mask,
                    te_obs,
                    te_action,
                    te_value,
                    te_log_prob,
                )

            inner_state = (
                train_state,
                env_state,
                last_obs,
                pending.goal,
                goal_ref,
                pending,
                success_since_boundary,
                rng,
            )
            inner_state, (
                traj_batch,
                goal_reached_traj,
                event_mask_traj,
                te_obs_traj,
                te_action_traj,
                te_value_traj,
                te_log_prob_traj,
            ) = jax.lax.scan(
                _env_step, inner_state, None, config["NUM_STEPS"]
            )
            # Fraction of steps where the teacher's goal cell changed = goal reached.
            goal_success_rate = (goal_reached_traj > 0).astype(jnp.float32).mean()

            # Unpack student/teacher rollout state. The teacher buffer and its
            # count persist across rollouts and remain in the outer scope.
            (
                train_state,
                env_state,
                last_obs,
                goal_idx,
                goal_ref,
                pending,
                success_since_boundary,
                rng,
            ) = inner_state

            # ----- EMPOWERMENT TEACHER REWARD (post-rollout) -----
            # The empowerment reward needs the future state reached after each
            # step, so it is computed here, after the rollout but before any
            # update. For every student step we form the tuple
            # (obs, goal, action, future_obs) and score it with the (frozen)
            # empowerment encoders:
            #     reward = energy(enc_sag, enc_g) - energy(enc_sg, enc_g)
            # The reward is detached so no gradient flows into the encoders.
            rng, _future_rng = jax.random.split(rng)
            teacher_future_obs = sample_future_states(
                _future_rng,
                traj_batch.obs[..., :3],
                traj_batch.done,
                gamma_cl,
            )
            # Flatten (NUM_STEPS, NUM_ENVS) into a single batch dim for the
            # empowerment encoders (their CNN expects one leading batch dim),
            # then reshape the per-sample reward back to (NUM_STEPS, NUM_ENVS).
            _emp_lead = traj_batch.obs.shape[:2]
            _emp_flat = _emp_lead[0] * _emp_lead[1]
            emp_obs = traj_batch.obs[..., :3].reshape(
                (_emp_flat,) + teacher_obs_shape
            )
            emp_goal = traj_batch.obs[..., 3].reshape(
                (_emp_flat,) + traj_batch.obs.shape[2:4]
            )
            emp_action = traj_batch.action.reshape((_emp_flat,))
            emp_future = teacher_future_obs.reshape(
                (_emp_flat,) + teacher_obs_shape
            )
            emp_repr_sag, emp_repr_sg, emp_repr_g = empowerment_network.apply(
                empowerment_train_state.params,
                emp_obs,
                emp_goal,
                emp_action,
                emp_future,
            )
            emp_energy_sag = energy_fn(
                empowerment_energy_name, emp_repr_sag, emp_repr_g
            )
            emp_energy_sg = energy_fn(
                empowerment_energy_name, emp_repr_sg, emp_repr_g
            )
            empowerment_reward_traj = jax.lax.stop_gradient(
                (emp_energy_sag - emp_energy_sg).reshape(_emp_lead)
            )

            # ----- APPEND TEACHER TRANSITIONS (event-based) -----
            # Flatten over (steps, envs) and append one teacher transition per
            # proposal-end event into the persistent flat buffer, offset by the
            # running count. Each event's reward is this step's empowerment
            # reward; ``done=1`` marks every teacher transition terminal so the
            # teacher PPO update treats each proposal independently.
            flat_event = event_mask_traj.reshape(-1)
            flat_reward = empowerment_reward_traj.reshape(-1)
            flat_obs = te_obs_traj.reshape((-1,) + teacher_obs_shape)
            flat_action = te_action_traj.reshape(-1)
            flat_value = te_value_traj.reshape(-1)
            flat_log_prob = te_log_prob_traj.reshape(-1)

            event_rank = jnp.cumsum(flat_event.astype(jnp.int32)) - 1
            raw_write_idx = teacher_buffer_count + event_rank
            valid_event = flat_event & (raw_write_idx < teacher_batch_size)
            safe_write_idx = jnp.where(
                valid_event, raw_write_idx, teacher_batch_size
            )
            teacher_buffer = TeacherTransition(
                done=teacher_buffer.done.at[safe_write_idx].set(
                    valid_event.astype(teacher_buffer.done.dtype)
                ),
                action=teacher_buffer.action.at[safe_write_idx].set(
                    flat_action.astype(teacher_buffer.action.dtype)
                ),
                value=teacher_buffer.value.at[safe_write_idx].set(
                    flat_value.astype(teacher_buffer.value.dtype)
                ),
                reward=teacher_buffer.reward.at[safe_write_idx].set(
                    flat_reward.astype(teacher_buffer.reward.dtype)
                ),
                log_prob=teacher_buffer.log_prob.at[safe_write_idx].set(
                    flat_log_prob.astype(teacher_buffer.log_prob.dtype)
                ),
                obs=teacher_buffer.obs.at[safe_write_idx].set(
                    flat_obs.astype(teacher_buffer.obs.dtype)
                ),
            )
            teacher_buffer_count = jnp.minimum(
                teacher_buffer_count + valid_event.astype(jnp.int32).sum(),
                jnp.asarray(teacher_batch_size, dtype=jnp.int32),
            )

            # CALCULATE ADVANTAGE — bootstrap with the last goal_idx (which may
            # have been resampled during the rollout, not necessarily pending.goal).
            cond_last_obs = condition(last_obs, goal_idx)
            _, last_val = network.apply(train_state.params, cond_last_obs)

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
                # Batching and Shuffling
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
                # Mini-batch Updates
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
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            total_loss = jnp.mean(loss_info[0])
            value_loss = jnp.mean(loss_info[1][0])
            actor_loss = jnp.mean(loss_info[1][1])
            entropy = jnp.mean(loss_info[1][2])

            # ----- TEACHER PPO UPDATE -----
            # The teacher buffer was filled (post-rollout) with one transition
            # per proposal-end event, each carrying its empowerment reward. Fires
            # once TEACHER_BATCH_SIZE transitions are collected; buffer/count
            # reset after. With done=1 per transition the teacher GAE reduces to
            # advantage = reward - value (each proposal scored independently).
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
                t_state, t_buffer, t_count, t_rng, last_val = operands
                # Slice valid transitions only (drop the padding slot at the end).
                traj = jax.tree_util.tree_map(
                    lambda x: x[:teacher_batch_size], t_buffer
                )
                advantages, targets = _teacher_calculate_gae(traj, last_val)

                def _t_update_epoch(update_state, unused):
                    def _t_update_minibatch(t_state, batch_info):
                        traj_b, gae_b, tgt_b = batch_info

                        def _t_loss_fn(params, traj_b, gae, targets):
                            _, pi, value = teacher_network.apply(params, traj_b.obs)
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
                            loss_actor = -jnp.minimum(
                                loss_actor1, loss_actor2
                            ).mean()
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
                update_state = (t_state, traj, advantages, targets, epoch_rng)
                update_state, t_loss_info = jax.lax.scan(
                    _t_update_epoch, update_state, None, teacher_update_epochs
                )
                t_state = update_state[0]
                new_buffer = jax.tree_util.tree_map(jnp.zeros_like, t_buffer)
                new_count = jnp.zeros_like(t_count)
                metrics = (
                    t_loss_info[0].mean(),
                    t_loss_info[1][0].mean(),
                    t_loss_info[1][1].mean(),
                    t_loss_info[1][2].mean(),
                    jnp.array(1.0, dtype=jnp.float32),
                )
                return t_state, new_buffer, new_count, t_rng, metrics

            def _skip_teacher_update(operands):
                t_state, t_buffer, t_count, t_rng, last_val = operands
                z = jnp.array(0.0, dtype=jnp.float32)
                return t_state, t_buffer, t_count, t_rng, (z, z, z, z, z)

            teacher_should_update = (
                (teacher_buffer_count >= teacher_batch_size)
                & jnp.logical_not(jnp.asarray(teacher_uniform_random))
            )
            (
                teacher_train_state,
                teacher_buffer,
                teacher_buffer_count,
                rng,
                teacher_metrics,
            ) = jax.lax.cond(
                teacher_should_update,
                _run_teacher_update,
                _skip_teacher_update,
                (
                    teacher_train_state,
                    teacher_buffer,
                    teacher_buffer_count,
                    rng,
                    jnp.zeros((), dtype=jnp.float32),
                ),
            )
            (
                teacher_total_loss,
                teacher_value_loss,
                teacher_actor_loss,
                teacher_entropy,
                teacher_did_update,
            ) = teacher_metrics

            # ----- CL FUTURE-STATE BUFFER UPDATE -----
            raw_obs_chunk = traj_batch.obs[..., :3].astype(cl_buffer.obs.dtype)
            goal_map_chunk = traj_batch.obs[..., 3].astype(cl_buffer.goal.dtype)
            done_chunk = traj_batch.done.astype(bool)
            cl_buffer = EpisodeStore(
                obs=jax.lax.dynamic_update_slice_in_dim(
                    cl_buffer.obs, raw_obs_chunk, cl_buffer_ptr, axis=0
                ),
                goal=jax.lax.dynamic_update_slice_in_dim(
                    cl_buffer.goal, goal_map_chunk, cl_buffer_ptr, axis=0
                ),
                action=jax.lax.dynamic_update_slice_in_dim(
                    cl_buffer.action,
                    traj_batch.action.astype(cl_buffer.action.dtype),
                    cl_buffer_ptr,
                    axis=0,
                ),
                done=jax.lax.dynamic_update_slice_in_dim(
                    cl_buffer.done, done_chunk, cl_buffer_ptr, axis=0
                ),
                future_obs=cl_buffer.future_obs,
            )
            new_cl_buffer_ptr = cl_buffer_ptr + config["NUM_STEPS"]
            cl_window_full = new_cl_buffer_ptr >= cl_buffer_size
            rng, cl_sample_rng = jax.random.split(rng)

            def _flush_cl_buffer(operands):
                current_buffer, sample_rng = operands
                sampled_future_obs = sample_future_states(
                    sample_rng, current_buffer.obs, current_buffer.done, gamma_cl
                )
                completed = EpisodeStore(
                    obs=current_buffer.obs,
                    goal=current_buffer.goal,
                    action=current_buffer.action,
                    done=current_buffer.done,
                    future_obs=sampled_future_obs,
                )
                reset_buffer = jax.tree_util.tree_map(jnp.zeros_like, current_buffer)
                return reset_buffer, jnp.array(0, dtype=jnp.int32), completed

            def _keep_cl_buffer(operands):
                current_buffer, _ = operands
                return current_buffer, new_cl_buffer_ptr, completed_episode

            cl_buffer, cl_buffer_ptr, completed_episode = jax.lax.cond(
                cl_window_full,
                _flush_cl_buffer,
                _keep_cl_buffer,
                (cl_buffer, cl_sample_rng),
            )

            def _run_empowerment_update(operands):
                e_state, episode_data, e_rng = operands
                batch_size = cl_buffer_size * config["NUM_ENVS"]
                obs_batch = episode_data.obs.reshape((batch_size,) + episode_data.obs.shape[2:])
                goal_batch = episode_data.goal.reshape((batch_size,) + episode_data.goal.shape[2:])
                action_batch = episode_data.action.reshape((batch_size,))
                future_obs_batch = episode_data.future_obs.reshape(
                    (batch_size,) + episode_data.future_obs.shape[2:]
                )

                def _emp_loss_fn(params):
                    repr_1, repr_2, repr_3 = empowerment_network.apply(
                        params,
                        obs_batch,
                        goal_batch,
                        action_batch,
                        future_obs_batch,
                    )
                    logits_13 = energy_fn(
                        empowerment_energy_name,
                        repr_1[:, None, :],
                        repr_3[None, :, :],
                    )
                    logits_23 = energy_fn(
                        empowerment_energy_name,
                        repr_2[:, None, :],
                        repr_3[None, :, :],
                    )
                    loss_13 = contrastive_loss_fn(empowerment_loss_name, logits_13)
                    loss_23 = contrastive_loss_fn(empowerment_loss_name, logits_23)
                    total_emp_loss = loss_13 + loss_23
                    return total_emp_loss, (loss_13, loss_23)

                (total_emp_loss, (loss_13, loss_23)), grads = jax.value_and_grad(
                    _emp_loss_fn, has_aux=True
                )(e_state.params)
                e_state = e_state.apply_gradients(grads=grads)
                metrics = (
                    total_emp_loss,
                    loss_13,
                    loss_23,
                    jnp.array(1.0, dtype=jnp.float32),
                )
                return e_state, e_rng, metrics

            def _skip_empowerment_update(operands):
                e_state, _episode_data, e_rng = operands
                z = jnp.array(0.0, dtype=jnp.float32)
                return e_state, e_rng, (z, z, z, z)

            empowerment_should_update = teacher_should_update
            (
                empowerment_train_state,
                rng,
                empowerment_metrics,
            ) = jax.lax.cond(
                empowerment_should_update,
                _run_empowerment_update,
                _skip_empowerment_update,
                (empowerment_train_state, completed_episode, rng),
            )
            (
                empowerment_total_loss,
                empowerment_loss_13,
                empowerment_loss_23,
                empowerment_did_update,
            ) = empowerment_metrics

            student_current_lr = (
                linear_schedule(train_state.step) if config["ANNEAL_LR"] else config["LR"]
            )
            if teacher_uniform_random:
                teacher_current_lr = jnp.array(0.0, dtype=jnp.float32)
            else:
                teacher_current_lr = (
                    teacher_linear_schedule(teacher_train_state.step)
                    if config["ANNEAL_LR"]
                    else teacher_lr
                )

            # Debugging mode
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric)

            if config.get("WANDB_MODE", "disabled") != "disabled":
                def wandb_callback(args):
                    (
                        info,
                        total_loss,
                        value_loss,
                        actor_loss,
                        entropy,
                        goal_success_rate,
                        teacher_total_loss,
                        teacher_value_loss,
                        teacher_actor_loss,
                        teacher_entropy,
                        teacher_did_update,
                        empowerment_total_loss,
                        empowerment_loss_13,
                        empowerment_loss_23,
                        empowerment_did_update,
                        student_current_lr,
                        teacher_current_lr,
                        empowerment_reward_traj,
                        event_mask_traj,
                    ) = args
                    payload = {
                        "total_loss": float(total_loss),
                        "value_loss": float(value_loss),
                        "actor_loss": float(actor_loss),
                        "entropy": float(entropy),
                        "goal_success_rate": float(goal_success_rate),
                        "student/learning_rate": float(student_current_lr),
                        "teacher/learning_rate": float(teacher_current_lr),
                        "teacher/did_update": float(teacher_did_update),
                        "empowerment/did_update": float(empowerment_did_update),
                    }
                    if float(teacher_did_update) > 0.0:
                        payload.update(
                            {
                                "teacher/total_loss": float(teacher_total_loss),
                                "teacher/value_loss": float(teacher_value_loss),
                                "teacher/actor_loss": float(teacher_actor_loss),
                                "teacher/entropy": float(teacher_entropy),
                            }
                        )
                    if float(empowerment_did_update) > 0.0:
                        payload.update(
                            {
                                "empowerment/contrastive_loss_total": float(empowerment_total_loss),
                                "empowerment/contrastive_loss_13": float(empowerment_loss_13),
                                "empowerment/contrastive_loss_23": float(empowerment_loss_23),
                            }
                        )
                    # Keep all metrics on a consistent global step for stable wandb curves.
                    step = int(info["timestep"].max() * config["NUM_ENVS"])
                    returned_episode = info["returned_episode"]
                    return_values = info["returned_episode_returns"][returned_episode]
                    episode_lengths = info["returned_episode_lengths"][returned_episode]
                    if len(return_values) > 0:
                        payload.update(
                            {
                                "episodic_return": float(return_values.mean()),
                                "episodic_length": float(episode_lengths.mean()),
                            }
                        )
                    # Log the teacher's empowerment reward over the proposal-end
                    # events (done OR goal reached) collected this rollout.
                    event_bool = np.asarray(event_mask_traj).astype(bool)
                    emp_vals = np.asarray(empowerment_reward_traj)[event_bool]
                    if emp_vals.size > 0:
                        payload.update(
                            {
                                "teacher/empowerment_reward": float(emp_vals.mean()),
                            }
                        )
                    wandb.log(payload, step=step)

                jax.debug.callback(
                    wandb_callback,
                    (
                        metric,
                        total_loss,
                        value_loss,
                        actor_loss,
                        entropy,
                        goal_success_rate,
                        teacher_total_loss,
                        teacher_value_loss,
                        teacher_actor_loss,
                        teacher_entropy,
                        teacher_did_update,
                        empowerment_total_loss,
                        empowerment_loss_13,
                        empowerment_loss_23,
                        empowerment_did_update,
                        student_current_lr,
                        teacher_current_lr,
                        empowerment_reward_traj,
                        event_mask_traj,
                    ),
                )

            # Mid-training video renders: fires render_eval_episode at
            # RENDER_NUM_VIDEOS_DURING_TRAINING evenly-spaced update checkpoints.
            # Uses jax.debug.callback so it works inside the compiled scan loop
            # without breaking JIT — the Python-level interval check filters
            # which steps actually trigger a render.
            if config.get("RENDER_ENABLED", True) and config.get("RENDER_NUM_VIDEOS_DURING_TRAINING", 0) > 0:
                _num_renders = int(config["RENDER_NUM_VIDEOS_DURING_TRAINING"])
                _render_every_n = max(1, config["NUM_UPDATES"] // _num_renders)
                _updates_per_outer = config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]

                def _render_training_cb(optax_step, student_params, teacher_params):
                    # Optax step counts gradient updates; divide to get the outer
                    # (env-collection + PPO) update number.
                    outer_update = int(optax_step) // _updates_per_outer
                    if outer_update > 0 and (outer_update % _render_every_n) == 0:
                        global_step = (
                            outer_update * config["NUM_STEPS"] * config["NUM_ENVS"]
                        )
                        base = config.get(
                            "RENDER_OUT_PATH", "artifacts/minigrid_eval.mp4"
                        )
                        stem, ext = os.path.splitext(base)
                        out = f"{stem}_step{global_step:010d}{ext}"
                        vid = render_eval_episode(
                            student_params,
                            jax.random.PRNGKey(outer_update),
                            teacher_params,
                            out_path=out,
                        )
                        if config.get("WANDB_MODE", "disabled") != "disabled":
                            fmt = "gif" if vid.endswith(".gif") else "mp4"
                            wandb.log(
                                {"eval/video": wandb.Video(vid, format=fmt)},
                                step=global_step,
                            )

                jax.debug.callback(
                    _render_training_cb,
                    train_state.step,
                    train_state.params,
                    teacher_train_state.params,
                )

            # Mid-training teacher softmax snapshots at a fixed outer-update
            # frequency. This uses the fixed reset observation so snapshots are
            # comparable over time.
            if config.get("TEACHER_HEATMAP_EVERY_N_UPDATES", 0) > 0:
                _heatmap_every_n = int(config["TEACHER_HEATMAP_EVERY_N_UPDATES"])
                _updates_per_outer = config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
                _heatmap_base = config.get(
                    "TEACHER_HEATMAP_OUT_PATH", "artifacts/teacher_softmax.png"
                )

                def _teacher_heatmap_training_cb(optax_step, probs_snapshot):
                    outer_update = int(optax_step) // _updates_per_outer
                    if outer_update > 0 and (outer_update % _heatmap_every_n) == 0:
                        global_step = (
                            outer_update * config["NUM_STEPS"] * config["NUM_ENVS"]
                        )
                        probs = np.asarray(probs_snapshot)
                        stem, ext = os.path.splitext(_heatmap_base)
                        if not ext:
                            ext = ".png"
                        out = f"{stem}_step{global_step:010d}{ext}"
                        save_teacher_softmax_snapshot(
                            probs,
                            config["TEACHER_GRID_H"],
                            config["TEACHER_GRID_W"],
                            out,
                            title=f"Teacher goal-cell distribution @ step {global_step}",
                            wandb_step=(
                                global_step
                                if config.get("WANDB_MODE", "disabled") != "disabled"
                                else None
                            ),
                            wandb_key="teacher/goal_softmax_during_training",
                        )

                # Compute teacher probs on-device before crossing into the host
                # callback. Avoids nested JAX dispatch inside callback, which can
                # stall when plotting runs during a compiled scan.
                if teacher_uniform_random:
                    probs_snapshot = jnp.full(
                        (teacher_num_cells,),
                        1.0 / float(teacher_num_cells),
                        dtype=jnp.float32,
                    )
                else:
                    _, teacher_pi_snapshot, _ = teacher_apply(
                        teacher_train_state.params, last_obs
                    )
                    probs_snapshot = teacher_pi_snapshot.probs[0]
                jax.debug.callback(
                    _teacher_heatmap_training_cb,
                    train_state.step,
                    probs_snapshot,
                )

            runner_state = (
                train_state,
                teacher_train_state,
                empowerment_train_state,
                env_state,
                last_obs,
                pending,
                goal_ref,
                success_since_boundary,
                teacher_buffer,
                teacher_buffer_count,
                cl_buffer,
                cl_buffer_ptr,
                completed_episode,
                rng,
            )
            return runner_state, metric

        # Sanity-check the teacher head once on the fixed reset obs so its init
        # and apply paths are exercised (shapes verified) before training.
        teacher_logits_map, _teacher_pi, _teacher_value = teacher_apply(
            teacher_train_state.params, obsv
        )
        (
            teacher_flat_idx,
            (teacher_row, teacher_col),
            _teacher_lp,
            teacher_value,
        ) = teacher_sample(teacher_train_state.params, obsv, deterministic=True)

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            teacher_train_state,
            empowerment_train_state,
            env_state,
            obsv,
            pending,
            goal_ref,
            jnp.zeros((config["NUM_ENVS"],), dtype=bool),
            teacher_buffer,
            teacher_buffer_count,
            cl_buffer,
            cl_buffer_ptr,
            completed_episode,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        # Final teacher distribution over the (H, W) grid for a representative env
        # (all envs share the same fixed reset), exported for heatmap viz.
        final_teacher_train_state = runner_state[1]
        if teacher_uniform_random:
            teacher_probs = jnp.full(
                (teacher_num_cells,),
                1.0 / float(teacher_num_cells),
                dtype=jnp.float32,
            )
        else:
            _, final_teacher_pi, _ = teacher_apply(
                final_teacher_train_state.params, obsv
            )
            teacher_probs = final_teacher_pi.probs[0]
        return {
            "runner_state": runner_state,
            "metrics": metric,
            "teacher_params": final_teacher_train_state.params,
            "teacher_logits_map_shape": teacher_logits_map.shape,
            "teacher_value_shape": teacher_value.shape,
            "teacher_sample": (teacher_flat_idx, teacher_row, teacher_col),
            "teacher_probs": teacher_probs,
            "completed_episode": runner_state[12],
        }

    return train, render_eval_episode


if __name__ == "__main__":
    # Parse CLI arguments with typed defaults, then keep the existing dict-based
    # training pipeline unchanged.
    config = asdict(tyro.cli(TrainConfig))
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["PPO", "MINIGRID", config["ENV_NAME"], f"jax_{jax.__version__}"],
        name=f'purejaxrl_ppo_minigrid_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    train_fn, render_eval_episode = make_train(config)
    train_jit = jax.jit(train_fn)
    out = train_jit(rng)

    final_step = int(config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"])

    if config.get("RENDER_ENABLED", True):
        final_params = out["runner_state"][0].params
        video_path = render_eval_episode(
            final_params, jax.random.PRNGKey(0), out["teacher_params"]
        )
        if config.get("WANDB_MODE", "disabled") != "disabled":
            video_format = "gif" if video_path.endswith(".gif") else "mp4"
            wandb.log(
                {"eval/video": wandb.Video(video_path, format=video_format)},
                step=final_step,
            )

    # Teacher categorical heatmap over the (H, W) goal grid.
    teacher_probs = np.asarray(jax.device_get(out["teacher_probs"]))
    heatmap_path = config.get("TEACHER_HEATMAP_OUT_PATH", "artifacts/teacher_softmax.png")
    save_teacher_softmax_snapshot(
        teacher_probs,
        config["TEACHER_GRID_H"],
        config["TEACHER_GRID_W"],
        heatmap_path,
        title="Teacher goal-cell distribution",
        wandb_step=(final_step if config.get("WANDB_MODE", "disabled") != "disabled" else None),
        wandb_key="teacher/goal_softmax",
    )
