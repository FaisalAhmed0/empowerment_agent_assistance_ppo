import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from dataclasses import asdict, dataclass
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
import navix as nx
import tyro
from wrappers import LogWrapper, NavixGymnaxWrapper

WANDB_RUN = None


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

        symbolic = x.astype(jnp.int32)
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
        embedding = jnp.concatenate((tag_embed, color_embed, state_embed), axis=-1)

        embedding = nn.Conv(
            32,
            kernel_size=(2, 2),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = activation(embedding)
        embedding = nn.Conv(
            64,
            kernel_size=(2, 2),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = activation(embedding)
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@dataclass
class Config:
    lr: float = 2.5e-4
    num_envs: int = 128
    num_steps: int = 128
    total_timesteps: int = int(50e6)
    update_epochs: int = 1
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    activation: str = "relu"
    env_name: str = "Navix-DoorKey-16x16-v0"
    obs_embed_dim: int = 16
    obs_tag_vocab_size: int = 11
    obs_color_vocab_size: int = 6
    obs_state_vocab_size: int = 4
    anneal_lr: bool = True
    debug: bool = True
    render_enabled: bool = True
    render_num_steps: int = 500
    render_fps: int = 10
    render_out_path: str = "artifacts/minigrid_eval.mp4"
    render_deterministic: bool = False
    wandb_enabled: bool = True
    wandb_project: str = "minigrid_no_teacher"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_mode: str = "online"
    seed: int = 30


def _config_to_legacy_dict(args: Config) -> dict[str, Any]:
    config = asdict(args)
    return {k.upper(): v for k, v in config.items()}


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    _probe_env, env_params = NavixGymnaxWrapper(config["ENV_NAME"]), None
    _probe_obsv, _ = _probe_env.reset(jax.random.PRNGKey(0), env_params)
    teacher_grid_h, teacher_grid_w = int(_probe_obsv.shape[0]), int(_probe_obsv.shape[1])
    config["TEACHER_GRID_H"] = teacher_grid_h
    config["TEACHER_GRID_W"] = teacher_grid_w
    config["TEACHER_NUM_CELLS"] = teacher_grid_h * teacher_grid_w
    config["FIXED_EPISODE_LENGTH"] = 4 * teacher_grid_h * teacher_grid_w
    env, env_params = NavixGymnaxWrapper(config["ENV_NAME"], disable_termination=True, max_steps=config["FIXED_EPISODE_LENGTH"]), None
    obsv, env_state = env.reset(jax.random.PRNGKey(0), env_params)
    print("shape of observation before flattening")
    jax.debug.print("obsv.shape: {obsv}", obsv=obsv.shape)
    env = LogWrapper(env)

    def render_eval_episode(
        params, rng, out_path=None, num_steps=None, fps=None, deterministic=None
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

        def _rollout(params, rng):
            rng, reset_rng = jax.random.split(rng)
            obs, state = env.reset(reset_rng, env_params)

            def _step(carry, _):
                obs, state, rng = carry
                pi, _ = network.apply(params, obs)
                rng, sample_rng, step_rng = jax.random.split(rng, 3)
                if deterministic:
                    action = pi.mode()
                else:
                    action = pi.sample(seed=sample_rng)
                obs, state, reward, done, info = env.step(
                    step_rng, state, action, env_params
                )
                return (obs, state, rng), state

            _, states = jax.lax.scan(_step, (obs, state, rng), None, length=num_steps)
            # `states` is the LogEnvState stacked over time; the wrapped Navix
            # timestep (and its renderable state) lives in `env_state`.
            frames = jax.vmap(nx.observations.rgb)(states.env_state.state)
            return frames

        frames = jax.jit(_rollout)(params, rng)
        frames = np.asarray(jax.device_get(frames)).astype(np.uint8)

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

        print(
            f"[render_eval_episode] saved agent video to {out_path} ({len(frames)} frames)"
        )
        return out_path

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

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
        init_x = jnp.zeros(env.observation_space(env_params).shape)
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                # jax.debug.print("obsv.shape: {obsv}", obsv=obsv.shape)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
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
            
            # Debug and host-side logging (stdout / wandb).
            if config.get("DEBUG") or config.get("WANDB_ENABLED"):
                def callback(info):
                    global WANDB_RUN
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        step = int(timesteps[t])
                        episodic_return = float(return_values[t])
                        if config.get("DEBUG"):
                            print(
                                f"global step={step}, episodic return={episodic_return}"
                            )
                        if config.get("WANDB_ENABLED") and WANDB_RUN is not None:
                            WANDB_RUN.log(
                                {"episodic_return": episodic_return, "global_step": step},
                                step=step,
                            )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train, render_eval_episode


if __name__ == "__main__":
    args = tyro.cli(Config)
    config = _config_to_legacy_dict(args)
    if config["WANDB_ENABLED"]:
        try:
            import wandb
        except ImportError as err:
            raise ImportError(
                "wandb is not installed. Install with `pip install wandb` "
                "or run with `--wandb-enabled False`."
            ) from err
        WANDB_RUN = wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            name=config["WANDB_NAME"],
            mode=config["WANDB_MODE"],
            config=config,
        )

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn, render_eval_episode = make_train(config)
    train_jit = jax.jit(train_fn)
    out = train_jit(rng)

    if config.get("RENDER_ENABLED", True):
        final_params = out["runner_state"][0].params
        render_eval_episode(final_params, jax.random.PRNGKey(0))

    if config.get("WANDB_ENABLED") and WANDB_RUN is not None:
        WANDB_RUN.finish()