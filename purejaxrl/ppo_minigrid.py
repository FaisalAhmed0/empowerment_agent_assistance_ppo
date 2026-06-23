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
    TOTAL_TIMESTEPS: int = 25_000_000
    UPDATE_EPOCHS: int = 1
    NUM_MINIBATCHES: int = 8
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.3
    ENT_COEF: float = 0.001
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTIVATION: str = "relu"
    ENV_NAME: str = "Navix-DoorKey-16x16-v0"
    OBS_EMBED_DIM: int = 16
    OBS_TAG_VOCAB_SIZE: int = 11
    OBS_COLOR_VOCAB_SIZE: int = 6
    OBS_STATE_VOCAB_SIZE: int = 4
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
    TEACHER_REWARD_TYPE: str = "competence_lp"
    TEACHER_LP_ABSOLUTE: bool = False
    TEACHER_SUCCESS_BONUS: float = 1.0
    TEACHER_EVAL_EPISODES: int = 1
    TEACHER_EVAL_NUM_ENVS: int = 10
    TEACHER_EPISODE_LENGTH: int = 5
    TEACHER_LR: float = 2.5e-4
    TEACHER_GAMMA: float = 1.0
    TEACHER_GAE_LAMBDA: float = 0.999
    TEACHER_CLIP_EPS: float = 0.3
    TEACHER_ENT_COEF: float = 0.01
    TEACHER_VF_COEF: float = 0.5
    TEACHER_MAX_GRAD_NORM: float = 0.5
    TEACHER_UPDATE_EPOCHS: int = 4
    TEACHER_NUM_MINIBATCHES: int = 4


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
    reward: jnp.ndarray  # competence learning progress on that goal
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # raw symbolic obs the teacher conditioned on


class PendingTeacher(NamedTuple):
    """Per-env goal proposal currently active for one student update step."""

    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    # Student success rate on this goal measured when it was proposed
    # (the "before" term of the competence-LP teacher reward).
    competence_before: jnp.ndarray
    # The proposed flat goal-cell index, kept so competence can be re-evaluated
    # on the SAME goal after the student update (the "after" term).
    goal: jnp.ndarray


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
    print(f"Number of updates: {config['NUM_UPDATES']}")
    # Teacher PPO + competence learning-progress (LP) reward settings. The
    # teacher proposes a goal cell; its reward is the change in the student's
    # success rate on that SAME goal measured at the synchronized episode
    # boundary (before vs. after the student's experience for that episode).
    config.setdefault("TEACHER_REWARD_TYPE", "competence_lp")
    config.setdefault("TEACHER_LP_ABSOLUTE", False)
    config.setdefault("TEACHER_SUCCESS_BONUS", 1.0)
    config.setdefault("TEACHER_EVAL_EPISODES", 1)
    config.setdefault("TEACHER_EVAL_NUM_ENVS", 4)
    config.setdefault("TEACHER_EPISODE_LENGTH", 4)
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
    # envs end together every H = 4 * grid_h * grid_w steps. This makes the
    # episode boundary synchronized (all done at the same step), which lets
    # the competence LP eval fire reliably via lax.cond(any(done)).
    config["FIXED_EPISODE_LENGTH"] = 4 * teacher_grid_h * teacher_grid_w
    # Default eval horizon to one full fixed episode; can be overridden in config.
    config.setdefault("TEACHER_EVAL_HORIZON", config["FIXED_EPISODE_LENGTH"])
    config["TEACHER_BATCH_SIZE"] = (
        config["TEACHER_EPISODE_LENGTH"] * config["NUM_ENVS"]
    )
    assert (
        config["TEACHER_BATCH_SIZE"] % config["TEACHER_NUM_MINIBATCHES"] == 0
    ), "teacher batch size must be divisible by TEACHER_NUM_MINIBATCHES"
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
        # The teacher updates once every ``TEACHER_EPISODE_LENGTH`` student
        # updates, so anneal over the number of teacher (not student) updates.
        n = max(
            int(config["NUM_UPDATES"]) // config["TEACHER_EPISODE_LENGTH"], 1
        )
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
        # is trained with its own PPO loop. Its reward is the student's
        # competence learning progress on the proposed goal: the student's
        # success rate on that goal measured after vs. before each student update.
        grid_h = config["TEACHER_GRID_H"]
        grid_w = config["TEACHER_GRID_W"]
        teacher_gamma = config["TEACHER_GAMMA"]
        teacher_gae_lambda = config["TEACHER_GAE_LAMBDA"]
        teacher_clip_eps = config["TEACHER_CLIP_EPS"]
        teacher_vf_coef = config["TEACHER_VF_COEF"]
        teacher_ent_coef = config["TEACHER_ENT_COEF"]
        teacher_episode_length = config["TEACHER_EPISODE_LENGTH"]
        teacher_num_minibatches = config["TEACHER_NUM_MINIBATCHES"]
        teacher_update_epochs = config["TEACHER_UPDATE_EPOCHS"]
        teacher_batch_size = config["TEACHER_BATCH_SIZE"]
        teacher_lr = config["TEACHER_LR"]
        teacher_eval_horizon = config["TEACHER_EVAL_HORIZON"]
        teacher_eval_episodes = config["TEACHER_EVAL_EPISODES"]
        teacher_eval_num_envs = config["TEACHER_EVAL_NUM_ENVS"]
        teacher_lp_absolute = config["TEACHER_LP_ABSOLUTE"]
        teacher_success_bonus = config["TEACHER_SUCCESS_BONUS"]
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

        def teacher_apply(params, obs):
            """Apply the teacher CNN, returning ``(logits_map, pi, value)``."""
            return teacher_network.apply(params, obs)

        def teacher_sample(params, obs, rng=None, deterministic=False):
            """Sample a goal cell from the teacher for the given observation(s).

            Returns ``(flat_index, (row, col), log_prob, value)`` over the
            (H, W) grid, where ``value`` is the teacher critic's estimate.
            """
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

        # COMPETENCE EVAL (for the teacher's learning-progress reward).
        # Roll out the student conditioned on each goal cell from the fixed start
        # state and measure the fraction of independent (stochastic) rollouts that
        # reach the goal within ``TEACHER_EVAL_HORIZON`` steps. A goal cell counts
        # as "reached" when its symbolic content changes vs. the start snapshot --
        # the same criterion used during training. The student is sampled
        # stochastically so the per-goal success rate varies in ``[0, 1]`` even
        # though every rollout starts from the identical fixed reset.
        _eval_reset_obs0 = jax.tree_util.tree_map(lambda x: x[0], fixed_reset_obsv)
        _eval_reset_state0 = jax.tree_util.tree_map(
            lambda x: x[0], fixed_reset_env_state
        )

        def _eval_goal_competence(student_params, goal_indices, rng):
            n_goals = goal_indices.shape[0]
            reps = teacher_eval_episodes * teacher_eval_num_envs
            batch = n_goals * reps
            # [n_goals, reps] -> [batch] (goal index varies slowest).
            goals_b = jnp.broadcast_to(
                goal_indices[:, None], (n_goals, reps)
            ).reshape(batch)
            obs0 = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x[None], (batch,) + x.shape),
                _eval_reset_obs0,
            )
            state0 = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x[None], (batch,) + x.shape),
                _eval_reset_state0,
            )
            goal_ref0 = gather_goal_cell(obs0, goals_b, grid_h, grid_w)

            def _step(carry, unused):
                obs, state, reached, rng = carry
                cond_obs = condition(obs, goals_b)
                pi, _ = network.apply(student_params, cond_obs)
                rng, a_rng, s_rng = jax.random.split(rng, 3)
                action = pi.sample(seed=a_rng)
                step_rngs = jax.random.split(s_rng, batch)
                obs, state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_rngs, state, action, env_params)
                # Detect reach on the raw post-step obs BEFORE the fixed reset is
                # applied, so a reach that coincides with episode end still counts.
                current_cell = gather_goal_cell(obs, goals_b, grid_h, grid_w)
                reached = jnp.logical_or(
                    reached, jnp.any(jnp.not_equal(current_cell, goal_ref0), axis=-1)
                )

                def _reset_done(reset_leaf, leaf):
                    mask = jnp.reshape(
                        done, (done.shape[0],) + (1,) * (leaf.ndim - 1)
                    )
                    return jnp.where(mask, reset_leaf, leaf)

                # Keep the eval on the fixed-start distribution (mirror training).
                obs = jax.tree_util.tree_map(_reset_done, obs0, obs)
                state = jax.tree_util.tree_map(_reset_done, state0, state)
                return (obs, state, reached, rng), None

            (_, _, reached, _), _ = jax.lax.scan(
                _step,
                (obs0, state0, jnp.zeros((batch,), dtype=bool), rng),
                None,
                length=teacher_eval_horizon,
            )
            success = reached.astype(jnp.float32).reshape(n_goals, reps)
            return jnp.mean(success, axis=-1)

        # INIT TEACHER PROPOSAL + BUFFER
        # Sample one goal cell per env from the teacher on the fixed reset obs,
        # snapshot the goal cell content (for reach detection), and measure the
        # student's "before" competence on each proposed goal. One teacher
        # transition is collected per student update step; ``TEACHER_EPISODE_LENGTH``
        # of them form one teacher PPO episode.
        rng, _goal_rng = jax.random.split(rng)
        init_goal_idx, _, init_log_prob, init_value = teacher_sample(
            teacher_train_state.params, obsv, rng=_goal_rng
        )
        goal_ref = gather_goal_cell(obsv, init_goal_idx, grid_h, grid_w)
        rng, _eval_rng = jax.random.split(rng)
        init_competence_before = _eval_goal_competence(
            train_state.params, init_goal_idx, _eval_rng
        )
        pending = PendingTeacher(
            obs=obsv,
            action=init_goal_idx,
            value=init_value,
            log_prob=init_log_prob,
            competence_before=init_competence_before,
            goal=init_goal_idx,
        )
        teacher_obs_shape = env.observation_space(env_params).shape
        # Allocate teacher_episode_length + 1 rows so the padding-row pattern
        # can safely absorb out-of-range writes when the buffer is full.
        _tbuf_rows = teacher_episode_length + 1
        teacher_buffer = TeacherTransition(
            done=jnp.zeros((_tbuf_rows, config["NUM_ENVS"])),
            action=jnp.zeros(
                (_tbuf_rows, config["NUM_ENVS"]), dtype=jnp.int32
            ),
            value=jnp.zeros((_tbuf_rows, config["NUM_ENVS"])),
            reward=jnp.zeros((_tbuf_rows, config["NUM_ENVS"])),
            log_prob=jnp.zeros((_tbuf_rows, config["NUM_ENVS"])),
            obs=jnp.zeros((_tbuf_rows, config["NUM_ENVS"]) + teacher_obs_shape),
        )
        teacher_buffer_count = jnp.array(0, dtype=jnp.int32)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            (
                train_state,
                teacher_train_state,
                env_state,
                last_obs,
                pending,
                goal_ref,
                success_since_boundary,
                teacher_buffer,
                teacher_buffer_count,
                rng,
            ) = runner_state

            # COLLECT TRAJECTORIES
            # goal_idx resamples on episode-done OR teacher-goal-reached so the
            # student always has an active goal to chase. pending.goal is the
            # teacher's macro-goal for LP tracking; it only updates at the
            # synchronized episode boundary (done is all-True once per H steps).
            def _env_step(inner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    goal_idx,
                    goal_ref,
                    pending,
                    success_since_boundary,
                    teacher_buffer,
                    teacher_buffer_count,
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
                # Store the goal-conditioned observation so the PPO update reruns
                # the network on exactly the same inputs used to act.
                # transition = Transition(
                #     done, action, value, reward, log_prob, cond_obs, info
                # )

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
                goal_reached = jnp.any(
                    jnp.not_equal(current_cell, goal_ref), axis=-1
                )
                # Add the goal reached signal to the reward.
                reward = env_reward + goal_reached
                transition = Transition(
                    done, action, value, reward, log_prob, cond_obs, info
                )

                # SAMPLE NEXT TEACHER PROPOSAL from the current (frozen) teacher.
                # Used for student resample on done|reach and for updating the
                # pending macro-goal at the episode boundary.
                rng, _t_sample_rng = jax.random.split(rng)
                new_t_goal_idx, _, new_t_log_prob, new_t_value = teacher_sample(
                    teacher_train_state.params, obsv, rng=_t_sample_rng
                )

                # RESAMPLE STUDENT GOAL on episode end or teacher-goal reached.
                resample_mask = jnp.logical_or(done, goal_reached)
                goal_idx = jnp.where(resample_mask, new_t_goal_idx, goal_idx)
                new_ref = gather_goal_cell(obsv, goal_idx, grid_h, grid_w)
                goal_ref = jnp.where(resample_mask[:, None], new_ref, goal_ref)

                # COMPETENCE-LP AT SYNCHRONIZED EPISODE BOUNDARY
                # Episodes end only by truncation (fixed horizon), so done is
                # all-True or all-False for all envs simultaneously.
                done_mask = done.astype(bool)
                any_done = jnp.any(done_mask)

                def _do_eval(operands):
                    old_goals, new_goals, sp, e_rng = operands
                    ra, rb = jax.random.split(e_rng)
                    c_after = _eval_goal_competence(sp, old_goals, ra)
                    c_before = _eval_goal_competence(sp, new_goals, rb)
                    return c_after, c_before

                def _skip_eval(operands):
                    zeros = jnp.zeros(
                        (config["NUM_ENVS"],), dtype=jnp.float32
                    )
                    return zeros, zeros

                rng, _eval_rng = jax.random.split(rng)
                comp_after, comp_before_new = jax.lax.cond(
                    any_done,
                    _do_eval,
                    _skip_eval,
                    (
                        pending.goal,
                        new_t_goal_idx,
                        train_state.params,
                        _eval_rng,
                    ),
                )

                # TEACHER REWARD: LP = success_after - success_before_proposal.
                # Non-zero only at the episode boundary (done steps).
                comp_before_old = pending.competence_before
                lp = comp_after - comp_before_old
                if teacher_lp_absolute:
                    lp = jnp.abs(lp)
                success_bonus = (
                    teacher_success_bonus
                    * success_since_boundary.astype(lp.dtype)
                )
                teacher_reward = jnp.where(
                    done_mask, lp + success_bonus, jnp.zeros_like(lp)
                )
                success_since_boundary = jnp.where(
                    done_mask,
                    jnp.zeros_like(success_since_boundary),
                    success_since_boundary,
                )

                # BUFFER WRITE: one row per synchronized episode, writing all
                # NUM_ENVS envs at once. safe_write_row clamps to the padding row
                # (teacher_episode_length) when the buffer is already full.
                can_record = any_done & (
                    teacher_buffer_count < teacher_episode_length
                )
                safe_write_row = jnp.where(
                    can_record, teacher_buffer_count, teacher_episode_length
                )
                teacher_buffer = TeacherTransition(
                    done=teacher_buffer.done.at[safe_write_row].set(
                        jnp.zeros_like(teacher_reward)
                    ),
                    action=teacher_buffer.action.at[safe_write_row].set(
                        pending.action
                    ),
                    value=teacher_buffer.value.at[safe_write_row].set(
                        pending.value
                    ),
                    reward=teacher_buffer.reward.at[safe_write_row].set(
                        teacher_reward
                    ),
                    log_prob=teacher_buffer.log_prob.at[safe_write_row].set(
                        pending.log_prob
                    ),
                    obs=teacher_buffer.obs.at[safe_write_row].set(pending.obs),
                )
                teacher_buffer_count = teacher_buffer_count + any_done.astype(
                    jnp.int32
                )

                # UPDATE PENDING on done only; pending.goal is the macro-goal for
                # LP; the rest are the teacher PPO transition fields.
                pending = PendingTeacher(
                    obs=jnp.where(any_done, obsv, pending.obs),
                    action=jnp.where(done_mask, new_t_goal_idx, pending.action),
                    value=jnp.where(done_mask, new_t_value, pending.value),
                    log_prob=jnp.where(
                        done_mask, new_t_log_prob, pending.log_prob
                    ),
                    competence_before=jnp.where(
                        done_mask, comp_before_new, pending.competence_before
                    ),
                    goal=jnp.where(done_mask, new_t_goal_idx, pending.goal),
                )

                inner_state = (
                    train_state,
                    env_state,
                    obsv,
                    goal_idx,
                    goal_ref,
                    pending,
                    success_since_boundary,
                    teacher_buffer,
                    teacher_buffer_count,
                    rng,
                )
                return inner_state, (
                    transition,
                    goal_reached,
                    teacher_reward,
                    comp_after,
                    comp_before_old,
                )

            inner_state = (
                train_state,
                env_state,
                last_obs,
                pending.goal,
                goal_ref,
                pending,
                success_since_boundary,
                teacher_buffer,
                teacher_buffer_count,
                rng,
            )
            inner_state, (
                traj_batch,
                goal_reached_traj,
                teacher_reward_traj,
                comp_after_traj,
                comp_before_traj,
            ) = jax.lax.scan(
                _env_step, inner_state, None, config["NUM_STEPS"]
            )
            # Fraction of steps where the teacher's goal cell changed = goal reached.
            goal_success_rate = goal_reached_traj.mean()

            # Unpack teacher state updated during the rollout.
            (
                train_state,
                env_state,
                last_obs,
                goal_idx,
                goal_ref,
                pending,
                success_since_boundary,
                teacher_buffer,
                teacher_buffer_count,
                rng,
            ) = inner_state

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
            # The teacher buffer was filled inside the rollout scan (at each
            # synchronized episode boundary). Fires once TEACHER_EPISODE_LENGTH
            # completed episodes have been collected; buffer/count reset after.
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
                # Slice valid rows only (drop the padding row at index teacher_episode_length).
                traj = jax.tree_util.tree_map(
                    lambda x: x[:teacher_episode_length], t_buffer
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

            teacher_should_update = teacher_buffer_count >= teacher_episode_length
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
                    pending.value,
                ),
            )
            (
                teacher_total_loss,
                teacher_value_loss,
                teacher_actor_loss,
                teacher_entropy,
                teacher_did_update,
            ) = teacher_metrics
            student_current_lr = (
                linear_schedule(train_state.step) if config["ANNEAL_LR"] else config["LR"]
            )
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
                        student_current_lr,
                        teacher_current_lr,
                        teacher_reward_traj,
                        comp_after_traj,
                        comp_before_traj,
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
                    # Log LP and competence only at episode boundaries (done steps).
                    # teacher_reward_traj is zero at non-done steps.
                    boundary_mask = returned_episode  # (NUM_STEPS, NUM_ENVS) bool
                    lp_vals = teacher_reward_traj[boundary_mask]
                    ca_vals = comp_after_traj[boundary_mask]
                    cb_vals = comp_before_traj[boundary_mask]
                    if len(lp_vals) > 0:
                        payload.update(
                            {
                                "teacher_learning_progress": float(lp_vals.mean()),
                                "teacher/competence_after": float(ca_vals.mean()),
                                "teacher/competence_before": float(cb_vals.mean()),
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
                        student_current_lr,
                        teacher_current_lr,
                        teacher_reward_traj,
                        comp_after_traj,
                        comp_before_traj,
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
                _, teacher_pi_snapshot, _ = teacher_apply(
                    teacher_train_state.params, last_obs
                )
                jax.debug.callback(
                    _teacher_heatmap_training_cb,
                    train_state.step,
                    teacher_pi_snapshot.probs[0],
                )

            runner_state = (
                train_state,
                teacher_train_state,
                env_state,
                last_obs,
                pending,
                goal_ref,
                success_since_boundary,
                teacher_buffer,
                teacher_buffer_count,
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
            env_state,
            obsv,
            pending,
            goal_ref,
            jnp.zeros((config["NUM_ENVS"],), dtype=bool),
            teacher_buffer,
            teacher_buffer_count,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        # Final teacher distribution over the (H, W) grid for a representative env
        # (all envs share the same fixed reset), exported for heatmap viz.
        final_teacher_train_state = runner_state[1]
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
