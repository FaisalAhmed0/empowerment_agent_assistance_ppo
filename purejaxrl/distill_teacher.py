"""Distill a teacher from saved teacher interactions with a supervised KL objective.

Expects interactions written by ``ppo_lp_teacher.py --SAVE_TEACHER_INTERACTIONS``:
  EXP_DIR/
    config.json
    teacher_interactions/update_XXXXXXX.npz   # one file per teacher PPO update

Each file holds, per teacher transition: ``base_obs`` (normalized obs fed to the
teacher), ``competence``, ``goal_idx``, ``raw_goal``, ``logits`` (sampling-time
teacher logits), ``teacher_obs`` and ``update_idx``.

The distilled model minimizes KL(p_teacher || p_distilled) over the goal grid.

Writes:
  EXP_DIR/distilled_teacher/
    checkpoint/            # Orbax TrainState
    distill_config.json    # input mode and network dims (see load_distilled_teacher)

Example:
  python purejaxrl/distill_teacher.py --EXP_DIR $SCRATCH/purejaxrl_simple_teachers/<name>_<id>
  python purejaxrl/distill_teacher.py --EXP_DIR ... --ONLY_COMPETENCE_INPUT --RECENCY_WEIGHTING
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from flax.training.train_state import TrainState

try:
    from purejaxrl.ppo_lp_teacher import (
        TeacherActorCritic,
        load_checkpoint,
        save_checkpoint,
    )
    from purejaxrl.load_teacher_model import load_experiment_config
except ImportError:
    from ppo_lp_teacher import TeacherActorCritic, load_checkpoint, save_checkpoint
    from load_teacher_model import load_experiment_config


INTERACTION_KEYS = ("base_obs", "competence", "goal_idx", "logits", "update_idx")
DISTILL_CONFIG_NAME = "distill_config.json"
CHECKPOINT_SUBDIR = "checkpoint"


@dataclass
class DistillConfig:
    EXP_DIR: str
    INTERACTIONS_SUBDIR: str = "teacher_interactions"
    OUTPUT_SUBDIR: str = "distilled_teacher"
    # Feed only the competence vector to the distilled teacher (no state).
    ONLY_COMPETENCE_INPUT: bool = False
    # Exponentially up-weight recent transitions in the loss; the most recent
    # update has weight 1 and weights halve every
    # RECENCY_HALF_LIFE_FRAC * (max_update_idx - min_update_idx) updates.
    RECENCY_WEIGHTING: bool = False
    RECENCY_HALF_LIFE_FRAC: float = 0.25
    LR: float = 3e-4
    MAX_GRAD_NORM: float = 1.0
    NUM_EPOCHS: int = 50
    BATCH_SIZE: int = 1024
    VAL_FRAC: float = 0.1
    # None => inherit TEACHER_HIDDEN_DIM / TEACHER_ACTIVATION / TEACHER_USE_ENCODERS
    # from the source experiment config.
    HIDDEN_DIM: int | None = None
    ACTIVATION: str | None = None
    USE_ENCODERS: bool | None = None
    SEED: int = 0
    WANDB_MODE: str = "disabled"
    ENTITY: str = ""
    PROJECT: str = "purejaxrl_distill"


def load_teacher_interactions(
    exp_dir: str, subdir: str = "teacher_interactions"
) -> dict[str, np.ndarray]:
    """Concatenate all saved teacher batches, ordered by update index."""
    paths = sorted(glob.glob(os.path.join(exp_dir, subdir, "update_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No teacher interaction files in {os.path.join(exp_dir, subdir)}; "
            "train with --SAVE_TEACHER_INTERACTIONS."
        )
    chunks = {k: [] for k in INTERACTION_KEYS}
    for path in paths:
        with np.load(path) as data:
            for k in INTERACTION_KEYS:
                chunks[k].append(np.asarray(data[k]))
    data = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    order = np.argsort(data["update_idx"], kind="stable")
    return {k: v[order] for k, v in data.items()}


def recency_weights(update_idx: np.ndarray, half_life_frac: float) -> np.ndarray:
    u = update_idx.astype(np.float64)
    span = float(u.max() - u.min())
    if span <= 0.0:
        return np.ones_like(u, dtype=np.float32)
    half_life = max(half_life_frac * span, 1e-8)
    return np.power(2.0, (u - u.max()) / half_life).astype(np.float32)


def build_distill_inputs(
    base_obs: np.ndarray | jnp.ndarray,
    competence: np.ndarray | jnp.ndarray,
    only_competence_input: bool,
):
    if only_competence_input:
        return competence
    xp = jnp if isinstance(base_obs, jnp.ndarray) else np
    return xp.concatenate([base_obs, competence], axis=-1)


def make_distill_network(distill_cfg: dict[str, Any]) -> TeacherActorCritic:
    only_competence = bool(distill_cfg["only_competence_input"])
    return TeacherActorCritic(
        num_actions=int(distill_cfg["num_goals"]),
        obs_dim=0 if only_competence else int(distill_cfg["base_obs_dim"]),
        competence_dim=int(distill_cfg["num_competence"]),
        activation=distill_cfg["activation"],
        hidden_dim=int(distill_cfg["hidden_dim"]),
        use_encoders=bool(distill_cfg["use_encoders"]),
    )


def _input_dim(distill_cfg: dict[str, Any]) -> int:
    if distill_cfg["only_competence_input"]:
        return int(distill_cfg["num_competence"])
    return int(distill_cfg["base_obs_dim"]) + int(distill_cfg["num_competence"])


def _make_tx(distill_cfg: dict[str, Any]):
    return optax.chain(
        optax.clip_by_global_norm(float(distill_cfg["max_grad_norm"])),
        optax.adam(learning_rate=float(distill_cfg["lr"]), eps=1e-5),
    )


def create_distill_train_state(distill_cfg: dict[str, Any], rng) -> TrainState:
    network = make_distill_network(distill_cfg)
    params = network.init(rng, jnp.zeros((_input_dim(distill_cfg),)))
    return TrainState.create(
        apply_fn=network.apply, params=params, tx=_make_tx(distill_cfg)
    )


def load_distilled_teacher(
    exp_dir: str, subdir: str = "distilled_teacher"
) -> tuple[TrainState, dict[str, Any]]:
    """Rebuild the distilled TeacherActorCritic and restore its TrainState.

    Returns ``(train_state, distill_cfg)``; ``distill_cfg`` includes
    ``only_competence_input``, ``base_obs_dim``, ``num_competence`` and
    ``num_goals``. Inputs are ``competence`` or ``concat(base_obs, competence)``
    (see ``build_distill_inputs``).
    """
    out_dir = os.path.join(exp_dir, subdir)
    with open(os.path.join(out_dir, DISTILL_CONFIG_NAME), "r") as f:
        distill_cfg = json.load(f)
    template = create_distill_train_state(distill_cfg, jax.random.PRNGKey(0))
    ckpt_dir = os.path.abspath(os.path.join(out_dir, CHECKPOINT_SUBDIR))
    return load_checkpoint(template, ckpt_dir), distill_cfg


def kl_to_teacher(teacher_logits, student_logits):
    """Per-sample KL(p_teacher || p_student) over the goal grid."""
    log_p_t = jax.nn.log_softmax(teacher_logits, axis=-1)
    log_p_s = jax.nn.log_softmax(student_logits, axis=-1)
    return jnp.sum(jnp.exp(log_p_t) * (log_p_t - log_p_s), axis=-1)


def main():
    args = tyro.cli(DistillConfig)
    source_config = load_experiment_config(args.EXP_DIR)
    data = load_teacher_interactions(args.EXP_DIR, args.INTERACTIONS_SUBDIR)
    num_rows = int(data["goal_idx"].shape[0])
    update_idx = data["update_idx"]
    print(
        f"[distill] loaded {num_rows} teacher transitions from "
        f"{len(np.unique(update_idx))} updates (update_idx {update_idx.min()}..{update_idx.max()})"
    )

    distill_cfg = {
        "source_exp_dir": args.EXP_DIR,
        "only_competence_input": bool(args.ONLY_COMPETENCE_INPUT),
        "recency_weighting": bool(args.RECENCY_WEIGHTING),
        "recency_half_life_frac": float(args.RECENCY_HALF_LIFE_FRAC),
        "base_obs_dim": int(data["base_obs"].shape[-1]),
        "num_competence": int(data["competence"].shape[-1]),
        "num_goals": int(data["logits"].shape[-1]),
        "hidden_dim": int(
            args.HIDDEN_DIM
            if args.HIDDEN_DIM is not None
            else source_config.get("TEACHER_HIDDEN_DIM", 256)
        ),
        "activation": (
            args.ACTIVATION
            if args.ACTIVATION is not None
            else source_config.get("TEACHER_ACTIVATION", "tanh")
        ),
        "use_encoders": bool(
            args.USE_ENCODERS
            if args.USE_ENCODERS is not None
            else source_config.get("TEACHER_USE_ENCODERS", True)
        ),
        "lr": float(args.LR),
        "max_grad_norm": float(args.MAX_GRAD_NORM),
        "num_rows": num_rows,
    }

    inputs = build_distill_inputs(
        data["base_obs"].astype(np.float32),
        data["competence"].astype(np.float32),
        distill_cfg["only_competence_input"],
    )
    targets = data["logits"].astype(np.float32)
    if args.RECENCY_WEIGHTING:
        weights = recency_weights(update_idx, args.RECENCY_HALF_LIFE_FRAC)
    else:
        weights = np.ones((num_rows,), dtype=np.float32)

    np_rng = np.random.default_rng(args.SEED)
    perm = np_rng.permutation(num_rows)
    num_val = int(round(args.VAL_FRAC * num_rows))
    if num_val >= num_rows:
        num_val = 0
    val_idx, train_idx = perm[:num_val], perm[num_val:]
    num_train = int(train_idx.shape[0])
    batch_size = min(int(args.BATCH_SIZE), num_train)
    num_batches = max(num_train // batch_size, 1)
    print(
        f"[distill] train={num_train} val={num_val} batch_size={batch_size} "
        f"input_dim={inputs.shape[-1]} only_competence={distill_cfg['only_competence_input']} "
        f"recency_weighting={distill_cfg['recency_weighting']}"
    )

    x_train = jnp.asarray(inputs[train_idx])
    y_train = jnp.asarray(targets[train_idx])
    w_train = jnp.asarray(weights[train_idx])
    x_val = jnp.asarray(inputs[val_idx])
    y_val = jnp.asarray(targets[val_idx])
    w_val = jnp.asarray(weights[val_idx])

    train_state = create_distill_train_state(distill_cfg, jax.random.PRNGKey(args.SEED))

    def _weighted_loss(params, x, y, w):
        pi, _ = train_state.apply_fn(params, x)
        kl = kl_to_teacher(y, pi.logits)
        return jnp.sum(w * kl) / jnp.maximum(jnp.sum(w), 1e-8)

    @jax.jit
    def train_epoch(state, perm, x, y, w):
        batches = perm[: num_batches * batch_size].reshape(num_batches, batch_size)

        def _step(state, idx):
            loss, grads = jax.value_and_grad(_weighted_loss)(
                state.params, x[idx], y[idx], w[idx]
            )
            return state.apply_gradients(grads=grads), loss

        state, losses = jax.lax.scan(_step, state, batches)
        return state, losses.mean()

    @jax.jit
    def evaluate(params, x, y, w):
        pi, _ = train_state.apply_fn(params, x)
        kl = kl_to_teacher(y, pi.logits)
        weighted_kl = jnp.sum(w * kl) / jnp.maximum(jnp.sum(w), 1e-8)
        top1 = jnp.mean(jnp.argmax(pi.logits, -1) == jnp.argmax(y, -1))
        return kl.mean(), weighted_kl, top1

    wandb.init(
        entity=args.ENTITY or None,
        project=args.PROJECT,
        name=f"distill_{os.path.basename(os.path.normpath(args.EXP_DIR))}",
        config={**asdict(args), **distill_cfg},
        mode=args.WANDB_MODE,
    )

    rng = jax.random.PRNGKey(args.SEED + 1)
    for epoch in range(int(args.NUM_EPOCHS)):
        rng, perm_rng = jax.random.split(rng)
        train_state, train_loss = train_epoch(
            train_state,
            jax.random.permutation(perm_rng, num_train),
            x_train,
            y_train,
            w_train,
        )
        log = {"epoch": epoch, "train/weighted_kl": float(train_loss)}
        if num_val > 0:
            val_kl, val_weighted_kl, val_top1 = evaluate(
                train_state.params, x_val, y_val, w_val
            )
            log.update(
                {
                    "val/kl": float(val_kl),
                    "val/weighted_kl": float(val_weighted_kl),
                    "val/top1_agreement": float(val_top1),
                }
            )
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in log.items()))
        wandb.log(log, step=epoch)

    out_dir = os.path.join(args.EXP_DIR, args.OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    save_checkpoint(train_state, os.path.abspath(os.path.join(out_dir, CHECKPOINT_SUBDIR)))
    with open(os.path.join(out_dir, DISTILL_CONFIG_NAME), "w") as f:
        json.dump(distill_cfg, f, indent=2)
    print(f"[distill] saved distilled teacher to {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
