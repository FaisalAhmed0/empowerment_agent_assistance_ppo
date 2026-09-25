"""Visualize teacher goal distributions across competence contexts.

Loads a teacher checkpoint from an experiment directory and renders subplot
grids of goal-distribution heatmaps (softmax probs and empirical sample
frequencies).

Competence sweep: observation slots match training episode reset (env.reset
obs; when ``TEACHER_OBS_GOAL_ONLY``, only the last two obs entries / target
goal are fed to the teacher); only the competence context varies across panels.

Action sweep (when the teacher conditions on action): fix reset obs and one
competence vector, vary uniformly sampled actions in ``[-1, 1]`` (bootstrap
action included as panel 0).

When the experiment was trained with ``TEACHER_CONDITION_ONLY_ON_COMPETENCE``,
inputs are competence-only (no student bootstrap / action sweep). When trained
with ``CONDITION_TEACHER_ON_ACTION=False`` (v2 default), the student action is
omitted from teacher inputs, the student checkpoint is not loaded, and the
action sweep is skipped. Use
``--use_teacher_ema`` to load ``checkpoints/teacher_ema``, or
``--use_teacher_avg`` to load ``checkpoints/teacher_avg`` (uniform average
over teacher updates, also written whenever max-log-sum competence is
saved). ``--use_max_log_sum_competence`` loads the paired
``teacher_max_log_sum_competence`` / ``student_max_log_sum_competence``
checkpoints (usable mid-training once a best has been saved).

Also saves a line plot of ``KL(P_c || P_{c=0})`` vs progressive competence
frontier ``t``, using the all-zero competence vector as the reference.

Use ``--use_one_hot_competence`` to sweep path-ordered one-hot competence
vectors (one panel per competence component) instead of the progressive
frontier; ``num_contexts`` is ignored in that mode.

Use ``--use_random_competence`` to sweep ``num_contexts`` competence vectors
with entries sampled i.i.d. from ``Uniform[0, 1]`` (mutually exclusive with
``--use_one_hot_competence``).

Use ``--sweep_eval_goals`` (default on) to also render a competence-sweep
figure pair for each maze evaluation goal from ``all_possible_goals()``,
highlighting that goal and conditioning the student bootstrap on it when
``CONDITION_ON_GOAL`` is set. Pass ``--no-sweep_eval_goals`` to skip.

Use ``--competence_vector_scale`` to override competence magnitude at
inference (CLI > experiment ``COMPETENCE_VECTOR_SCALE`` > default ``1.0``).

Example:
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id>
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> --use_teacher_ema
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> --use_teacher_avg
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> --num_samples 1000
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> \\
      --num_action_samples 8 --competence_idx 4
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> \\
      --use_one_hot_competence
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> \\
      --use_random_competence
  python purejaxrl/visualize_teacher_context_sweep.py \\
      --exp_dir $SCRATCH/purejaxrl_simple_teachers/<name>_<id> \\
      --no-sweep_eval_goals
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tyro

try:
    from envs.ant_maze import U_MAZE_PATH_CELLS, all_possible_goals
except ImportError:
    from purejaxrl.envs.ant_maze import U_MAZE_PATH_CELLS, all_possible_goals

try:
    from purejaxrl.load_teacher_model import (
        _make_env,
        load_experiment_config,
        load_obs_norm_stats,
        load_student_model,
        load_teacher_model,
    )
except ImportError:
    from load_teacher_model import (
        _make_env,
        load_experiment_config,
        load_obs_norm_stats,
        load_student_model,
        load_teacher_model,
    )


@dataclass
class ContextSweepConfig:
    exp_dir: str
    checkpoint_dir: str = "checkpoints"
    num_contexts: int = 8
    seed: int = 30
    output_dir: Optional[str] = None
    size_scaling: float = 4.0
    use_teacher_ema: bool = False
    use_teacher_avg: bool = False
    use_max_log_sum_competence: bool = False
    num_samples: int = 1500
    num_action_samples: int = 8
    competence_idx: Optional[int] = None
    use_one_hot_competence: bool = False
    use_random_competence: bool = False
    sweep_eval_goals: bool = True
    # None => use config COMPETENCE_VECTOR_SCALE if present, else 1.0.
    competence_vector_scale: Optional[float] = None


def _resolve_competence_vector_scale(
    args: ContextSweepConfig,
    config: dict[str, Any],
) -> float:
    """CLI override > experiment config > training-consistent default (1.0)."""
    if args.competence_vector_scale is not None:
        return float(args.competence_vector_scale)
    if "COMPETENCE_VECTOR_SCALE" in config:
        return float(config["COMPETENCE_VECTOR_SCALE"])
    return 1.0


def _build_goal_grid(config: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Rebuild the teacher discrete goal grid used during training."""
    num_points = int(config["TEACHER_NUM_GOAL_POINTS"])
    xs = np.linspace(
        float(config["TEACHER_GOAL_X_MIN"]),
        float(config["TEACHER_GOAL_X_MAX"]),
        num_points,
    )
    ys = np.linspace(
        float(config["TEACHER_GOAL_Y_MIN"]),
        float(config["TEACHER_GOAL_Y_MAX"]),
        num_points,
    )
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    goal_grid = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
    custom_goal = np.array([12.0, 8.0], dtype=np.float32)
    replace_idx = int(np.argmin(np.sum((goal_grid - custom_goal) ** 2, axis=-1)))
    goal_grid[replace_idx] = custom_goal
    return goal_grid, num_points


def _path_ordered_competence_ranks(
    competence_goals: np.ndarray,
    size_scaling: float = 4.0,
) -> np.ndarray:
    """Map each competence goal to its rank along the U-corridor path."""
    path_xy = np.asarray(U_MAZE_PATH_CELLS, dtype=np.float32) * float(size_scaling)
    ranks = np.full(competence_goals.shape[0], -1, dtype=np.int32)
    used = set()
    for rank, xy in enumerate(path_xy):
        dists = np.sum((competence_goals - xy) ** 2, axis=-1)
        order = np.argsort(dists)
        for idx in order:
            idx = int(idx)
            if idx not in used:
                ranks[idx] = rank
                used.add(idx)
                break
    if np.any(ranks < 0):
        raise ValueError("Failed to assign path ranks to all competence goals")
    return ranks


def build_progressive_contexts(
    num_competence: int,
    ranks: np.ndarray,
    num_contexts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Progressive mastery: near goals saturate first, far goals last.

    For frontier ``t`` in ``linspace(0, 1, num_contexts)``::
        competence[i] = clip(t * N - rank[i], 0, 1)
    """
    n = float(num_competence)
    ts = np.linspace(0.0, 1.0, num_contexts, dtype=np.float32)
    contexts = np.zeros((num_contexts, num_competence), dtype=np.float32)
    for i, t in enumerate(ts):
        contexts[i] = np.clip(t * n - ranks.astype(np.float32), 0.0, 1.0)
    return contexts, ts


def build_one_hot_contexts(
    num_competence: int,
    ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One-hot competence vectors ordered by path rank.

    Context ``i`` has a 1.0 at the competence goal whose path rank is ``i``.
    ``ts`` is the integer path index ``0 .. N-1``.
    """
    n = int(num_competence)
    ranks = np.asarray(ranks, dtype=np.int32).reshape(-1)
    if ranks.shape[0] != n:
        raise ValueError(
            f"ranks length {ranks.shape[0]} != num_competence {n}"
        )
    contexts = np.zeros((n, n), dtype=np.float32)
    for goal_idx, rank in enumerate(ranks):
        contexts[int(rank), goal_idx] = 1.0
    ts = np.arange(n, dtype=np.float32)
    return contexts, ts


def build_random_contexts(
    num_competence: int,
    num_contexts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random competence vectors with i.i.d. Uniform[0, 1] entries.

    ``ts`` is the sample index ``0 .. num_contexts-1`` (for KL x-axis).
    """
    rng = np.random.default_rng(seed)
    contexts = rng.uniform(
        0.0, 1.0, size=(int(num_contexts), int(num_competence))
    ).astype(np.float32)
    ts = np.arange(int(num_contexts), dtype=np.float32)
    return contexts, ts


def _build_teacher_input(
    obs: jnp.ndarray,
    competence: jnp.ndarray,
    action: jnp.ndarray,
    *,
    condition_on_competence: bool,
    only_on_competence: bool = False,
    condition_on_action: bool = True,
    goal_only: bool = False,
) -> jnp.ndarray:
    if only_on_competence:
        return competence
    if goal_only:
        obs = jnp.asarray(obs)[..., -2:]
    parts = [obs]
    if condition_on_competence:
        parts.append(competence)
    if condition_on_action:
        parts.append(action)
    return jnp.concatenate(parts, axis=-1)


def _normalize_obs(
    obs: jnp.ndarray,
    obs_norm: tuple[np.ndarray, np.ndarray] | None,
) -> jnp.ndarray:
    """Apply saved running-mean/var normalization, matching training."""
    if obs_norm is None:
        return obs
    mean, var = obs_norm
    return (obs - jnp.asarray(mean, dtype=obs.dtype)) / jnp.sqrt(
        jnp.asarray(var, dtype=obs.dtype) + 1e-8
    )


def _reset_and_normalize_obs(
    env,
    env_params,
    config: dict[str, Any],
    *,
    obs_norm: tuple[np.ndarray, np.ndarray] | None,
    reset_rng,
) -> tuple[jnp.ndarray, np.ndarray]:
    """Reset env with ``reset_rng`` and optionally normalize obs."""
    normalize_env = bool(config.get("NORMALIZE_ENV", False))
    raw_obs, _env_state = env.reset(reset_rng, env_params)
    raw_obs = jnp.asarray(raw_obs, dtype=jnp.float32).reshape(-1)
    start_xy = np.asarray(jax.device_get(raw_obs[:2]), dtype=np.float32)

    if normalize_env:
        if obs_norm is None:
            raise FileNotFoundError(
                "NORMALIZE_ENV is True but obs_norm_stats.npz was not found; "
                "needed to match training reset observations."
            )
        norm_obs = _normalize_obs(raw_obs, obs_norm)
    else:
        norm_obs = raw_obs
    return norm_obs, start_xy


def sample_reset_obs(
    env,
    env_params,
    config: dict[str, Any],
    *,
    obs_norm: tuple[np.ndarray, np.ndarray] | None,
    seed: int,
) -> tuple[jnp.ndarray, np.ndarray]:
    """Reset the env and optionally normalize obs (no student needed).

    Returns ``(norm_obs, start_xy)`` where ``start_xy`` is the denormalized/raw
    agent xy for plotting.
    """
    rng = jax.random.PRNGKey(seed)
    _, reset_rng = jax.random.split(rng)
    return _reset_and_normalize_obs(
        env, env_params, config, obs_norm=obs_norm, reset_rng=reset_rng
    )


def sample_reset_teacher_inputs(
    env,
    env_params,
    student_train_state,
    config: dict[str, Any],
    *,
    obs_norm: tuple[np.ndarray, np.ndarray] | None,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
    """Match training episode-start teacher inputs.

    Mirrors ``make_train`` reset bootstrap:
      1. ``env.reset``
      2. normalize obs if ``NORMALIZE_ENV``
      3. student samples an action with a zero goal appended (if conditioned)
      4. teacher sees ``(goal_xy or base_obs, competence[, bootstrap_action])``

    Returns ``(norm_obs, bootstrap_action, start_xy)`` where ``start_xy`` is the
    denormalized/raw agent xy for plotting.
    """
    condition_on_goal = bool(config.get("CONDITION_ON_GOAL", False))
    goal_dim = 2

    rng = jax.random.PRNGKey(seed)
    _, reset_rng, action_rng = jax.random.split(rng, 3)
    norm_obs, start_xy = _reset_and_normalize_obs(
        env, env_params, config, obs_norm=obs_norm, reset_rng=reset_rng
    )

    policy_obs = norm_obs
    if condition_on_goal:
        policy_obs = jnp.concatenate(
            [policy_obs, jnp.zeros((goal_dim,), dtype=policy_obs.dtype)],
            axis=-1,
        )
    pi, _ = student_train_state.apply_fn(student_train_state.params, policy_obs)
    bootstrap_action = pi.sample(seed=action_rng)
    bootstrap_action = jnp.asarray(bootstrap_action, dtype=jnp.float32).reshape(-1)
    return norm_obs, bootstrap_action, start_xy


def _normalize_xy_goal(
    goal: jnp.ndarray,
    obs_norm: tuple[np.ndarray, np.ndarray] | None,
) -> jnp.ndarray:
    """Normalize a raw xy goal with the first two obs-norm channels."""
    goal = jnp.asarray(goal, dtype=jnp.float32).reshape(-1)[:2]
    if obs_norm is None:
        return goal
    mean, var = obs_norm
    mean_xy = jnp.asarray(mean, dtype=goal.dtype).reshape(-1)[:2]
    var_xy = jnp.asarray(var, dtype=goal.dtype).reshape(-1)[:2]
    return (goal - mean_xy) / jnp.sqrt(var_xy + 1e-8)


def sample_bootstrap_action_for_goal(
    student_train_state,
    config: dict[str, Any],
    norm_obs: jnp.ndarray,
    raw_goal_xy: np.ndarray | jnp.ndarray,
    *,
    obs_norm: tuple[np.ndarray, np.ndarray] | None,
    seed: int,
) -> jnp.ndarray:
    """Sample a student bootstrap action conditioned on ``raw_goal_xy``.

    Reuses a fixed reset ``norm_obs``. When ``CONDITION_ON_GOAL`` is set, the
    raw goal is (optionally) normalized like training and appended to the
    policy observation; otherwise this matches the zero-goal bootstrap path.
    """
    condition_on_goal = bool(config.get("CONDITION_ON_GOAL", False))
    normalize_env = bool(config.get("NORMALIZE_ENV", False))
    action_rng = jax.random.PRNGKey(seed)
    policy_obs = jnp.asarray(norm_obs, dtype=jnp.float32).reshape(-1)
    if condition_on_goal:
        if normalize_env:
            policy_goal = _normalize_xy_goal(raw_goal_xy, obs_norm)
        else:
            policy_goal = jnp.asarray(raw_goal_xy, dtype=jnp.float32).reshape(-1)[:2]
        policy_obs = jnp.concatenate([policy_obs, policy_goal], axis=-1)
    pi, _ = student_train_state.apply_fn(student_train_state.params, policy_obs)
    bootstrap_action = pi.sample(seed=action_rng)
    return jnp.asarray(bootstrap_action, dtype=jnp.float32).reshape(-1)


def compute_teacher_probs(
    teacher_train_state,
    config: dict[str, Any],
    contexts: np.ndarray,
    *,
    obs: jnp.ndarray | None = None,
    action: jnp.ndarray | None = None,
    competence_vector_scale: float = 1.0,
) -> np.ndarray:
    """Return teacher P(goal) for each context; shape (num_contexts, num_goals)."""
    only_on_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    condition_on_competence = bool(
        config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
    ) or only_on_competence
    condition_on_action = bool(config.get("CONDITION_TEACHER_ON_ACTION", True))
    goal_only = bool(config.get("TEACHER_OBS_GOAL_ONLY", False))
    params = teacher_train_state.params
    apply_fn = teacher_train_state.apply_fn
    scale = float(competence_vector_scale)

    probs_list = []
    for competence in contexts:
        teacher_obs = _build_teacher_input(
            obs if obs is not None else jnp.zeros((0,), dtype=jnp.float32),
            jnp.asarray(competence, dtype=jnp.float32) * scale,
            action if action is not None else jnp.zeros((0,), dtype=jnp.float32),
            condition_on_competence=condition_on_competence,
            only_on_competence=only_on_competence,
            condition_on_action=condition_on_action,
            goal_only=goal_only,
        )
        pi, _ = apply_fn(params, teacher_obs)
        probs_list.append(np.asarray(jax.device_get(pi.probs)).reshape(-1))
    return np.stack(probs_list, axis=0)


def sample_teacher_goal_counts(
    teacher_train_state,
    config: dict[str, Any],
    contexts: np.ndarray,
    *,
    num_samples: int,
    seed: int,
    obs: jnp.ndarray | None = None,
    action: jnp.ndarray | None = None,
    competence_vector_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample goals from the teacher and bin counts per context.

    Returns ``(counts, freqs)`` each of shape ``(num_contexts, num_goals)``,
    where ``freqs = counts / num_samples``.
    """
    only_on_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    condition_on_competence = bool(
        config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
    ) or only_on_competence
    condition_on_action = bool(config.get("CONDITION_TEACHER_ON_ACTION", True))
    goal_only = bool(config.get("TEACHER_OBS_GOAL_ONLY", False))
    params = teacher_train_state.params
    apply_fn = teacher_train_state.apply_fn
    scale = float(competence_vector_scale)
    rng = jax.random.PRNGKey(seed)

    counts_list = []
    for competence in contexts:
        teacher_obs = _build_teacher_input(
            obs if obs is not None else jnp.zeros((0,), dtype=jnp.float32),
            jnp.asarray(competence, dtype=jnp.float32) * scale,
            action if action is not None else jnp.zeros((0,), dtype=jnp.float32),
            condition_on_competence=condition_on_competence,
            only_on_competence=only_on_competence,
            condition_on_action=condition_on_action,
            goal_only=goal_only,
        )
        pi, _ = apply_fn(params, teacher_obs)
        rng, sample_rng = jax.random.split(rng)
        goal_idxs = pi.sample(seed=sample_rng, sample_shape=(num_samples,))
        goal_idxs_np = np.asarray(jax.device_get(goal_idxs)).reshape(-1).astype(np.int32)
        num_goals = int(np.asarray(jax.device_get(pi.probs)).reshape(-1).shape[0])
        counts = np.bincount(goal_idxs_np, minlength=num_goals).astype(np.float32)
        counts_list.append(counts[:num_goals])
    counts_arr = np.stack(counts_list, axis=0)
    freqs = counts_arr / float(max(num_samples, 1))
    return counts_arr, freqs


def sample_uniform_actions(
    action_dim: int,
    num_samples: int,
    seed: int,
    *,
    bootstrap_action: jnp.ndarray | np.ndarray | None = None,
) -> np.ndarray:
    """Sample actions uniformly in ``[-1, 1]^{action_dim}``.

    If ``bootstrap_action`` is provided it is prepended as row 0.
    Returns shape ``(num_samples [+ 1], action_dim)``.
    """
    rng = np.random.default_rng(seed)
    actions = rng.uniform(
        low=-1.0, high=1.0, size=(num_samples, action_dim)
    ).astype(np.float32)
    if bootstrap_action is not None:
        boot = np.asarray(jax.device_get(bootstrap_action), dtype=np.float32).reshape(-1)
        if boot.shape[0] != action_dim:
            raise ValueError(
                f"bootstrap_action dim {boot.shape[0]} != action_dim {action_dim}"
            )
        actions = np.concatenate([boot[None, :], actions], axis=0)
    return actions


def compute_teacher_probs_for_actions(
    teacher_train_state,
    config: dict[str, Any],
    competence: np.ndarray,
    actions: np.ndarray,
    *,
    obs: jnp.ndarray,
    competence_vector_scale: float = 1.0,
) -> np.ndarray:
    """Return teacher P(goal) for each action; shape (num_actions, num_goals)."""
    only_on_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    condition_on_action = bool(config.get("CONDITION_TEACHER_ON_ACTION", True))
    if only_on_competence or not condition_on_action:
        raise ValueError(
            "compute_teacher_probs_for_actions requires a teacher that "
            "conditions on action"
        )
    condition_on_competence = bool(
        config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
    )
    goal_only = bool(config.get("TEACHER_OBS_GOAL_ONLY", False))
    params = teacher_train_state.params
    apply_fn = teacher_train_state.apply_fn
    competence_j = jnp.asarray(competence, dtype=jnp.float32).reshape(-1)
    scale = float(competence_vector_scale)

    probs_list = []
    for action in actions:
        teacher_obs = _build_teacher_input(
            obs,
            competence_j * scale,
            jnp.asarray(action, dtype=jnp.float32).reshape(-1),
            condition_on_competence=condition_on_competence,
            only_on_competence=False,
            condition_on_action=True,
            goal_only=goal_only,
        )
        pi, _ = apply_fn(params, teacher_obs)
        probs_list.append(np.asarray(jax.device_get(pi.probs)).reshape(-1))
    return np.stack(probs_list, axis=0)


def sample_teacher_goal_counts_for_actions(
    teacher_train_state,
    config: dict[str, Any],
    competence: np.ndarray,
    actions: np.ndarray,
    *,
    num_samples: int,
    seed: int,
    obs: jnp.ndarray,
    competence_vector_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample goals from the teacher and bin counts per action.

    Returns ``(counts, freqs)`` each of shape ``(num_actions, num_goals)``.
    """
    only_on_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    condition_on_action = bool(config.get("CONDITION_TEACHER_ON_ACTION", True))
    if only_on_competence or not condition_on_action:
        raise ValueError(
            "sample_teacher_goal_counts_for_actions requires a teacher that "
            "conditions on action"
        )
    condition_on_competence = bool(
        config.get("CONDITION_TEACHER_ON_COMPETENCE", True)
    )
    goal_only = bool(config.get("TEACHER_OBS_GOAL_ONLY", False))
    params = teacher_train_state.params
    apply_fn = teacher_train_state.apply_fn
    competence_j = jnp.asarray(competence, dtype=jnp.float32).reshape(-1)
    scale = float(competence_vector_scale)
    rng = jax.random.PRNGKey(seed)

    counts_list = []
    for action in actions:
        teacher_obs = _build_teacher_input(
            obs,
            competence_j * scale,
            jnp.asarray(action, dtype=jnp.float32).reshape(-1),
            condition_on_competence=condition_on_competence,
            only_on_competence=False,
            condition_on_action=True,
            goal_only=goal_only,
        )
        pi, _ = apply_fn(params, teacher_obs)
        rng, sample_rng = jax.random.split(rng)
        goal_idxs = pi.sample(seed=sample_rng, sample_shape=(num_samples,))
        goal_idxs_np = np.asarray(jax.device_get(goal_idxs)).reshape(-1).astype(np.int32)
        num_goals = int(np.asarray(jax.device_get(pi.probs)).reshape(-1).shape[0])
        counts = np.bincount(goal_idxs_np, minlength=num_goals).astype(np.float32)
        counts_list.append(counts[:num_goals])
    counts_arr = np.stack(counts_list, axis=0)
    freqs = counts_arr / float(max(num_samples, 1))
    return counts_arr, freqs


def _format_competence_title(competence: np.ndarray) -> str:
    """Format competence as a compact row-vector title string."""
    vals = np.asarray(competence).reshape(-1)
    body = ", ".join(f"{v:.2f}" for v in vals)
    return f"c = [{body}]"


def _kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """KL(p || q) for discrete distributions, with clipping for stability."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def kl_vs_zero_competence(
    probs: np.ndarray,
    zero_probs: np.ndarray,
) -> np.ndarray:
    """Per-context KL(P_c || P_{c=0}); shape ``(num_contexts,)``."""
    zero_probs = np.asarray(zero_probs).reshape(-1)
    return np.asarray(
        [_kl_divergence(probs[i], zero_probs) for i in range(probs.shape[0])],
        dtype=np.float64,
    )


def render_kl_vs_zero_figure(
    ts: np.ndarray,
    kl_values: np.ndarray,
    *,
    save_path: Optional[str] = None,
    title: str = r"KL$(P_c \| P_{c=0})$ vs progressive competence",
    xlabel: str = r"competence frontier $t$",
):
    """Line plot of KL divergence to the all-zero competence reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(ts, kl_values, marker="o", linewidth=2.0, markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"KL$(P_c \| P_{c=0})$")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(float(np.min(ts)) - 0.02, float(np.max(ts)) + 0.02)
    y_max = float(np.max(kl_values)) if kl_values.size else 0.0
    ax.set_ylim(bottom=0.0, top=max(y_max * 1.1, 1e-6))
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def _format_action_title(action: np.ndarray, *, index: int, is_bootstrap: bool) -> str:
    """Format an action vector as a compact panel title."""
    vals = np.asarray(action).reshape(-1)
    # Show first 4 dims; truncate if longer.
    preview = ", ".join(f"{v:.2f}" for v in vals[:4])
    suffix = ", ..." if vals.shape[0] > 4 else ""
    label = "bootstrap" if is_bootstrap else f"a[{index}]"
    return f"{label}: [{preview}{suffix}]"


def _plot_probs_on_ax(
    ax,
    goal_grid_xy: np.ndarray,
    probs: np.ndarray,
    num_points: int,
    *,
    cmap,
    vmin: float,
    vmax: float,
    start_xy=None,
    competence_goals=None,
    competence=None,
    highlight_goal_xy=None,
):
    """Draw a single teacher goal-distribution heatmap on ``ax``."""
    goal_grid_xy = np.asarray(goal_grid_xy)
    probs = np.asarray(probs).reshape(-1)
    gx = goal_grid_xy[:, 0].reshape(num_points, num_points)
    gy = goal_grid_xy[:, 1].reshape(num_points, num_points)
    vgrid = probs.reshape(num_points, num_points)
    mesh = ax.pcolormesh(
        gx, gy, vgrid, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax
    )
    if start_xy is not None:
        start_xy = np.asarray(start_xy).reshape(-1)
        ax.scatter(
            start_xy[0],
            start_xy[1],
            s=160,
            marker="*",
            c="black",
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
            label="start",
        )
    if competence_goals is not None:
        competence_goals = np.asarray(competence_goals)
        # Inactive competence goals: thin open circles.
        ax.scatter(
            competence_goals[:, 0],
            competence_goals[:, 1],
            s=36,
            marker="o",
            facecolors="none",
            edgecolors="0.35",
            linewidths=0.9,
            zorder=3,
            label="competence goals",
        )
        if competence is not None:
            competence = np.asarray(competence).reshape(-1)
            active = competence > 1e-6
            if np.any(active):
                # Highlight goals with non-zero competence; marker size and
                # fill alpha scale with the competence entry.
                active_xy = competence_goals[active]
                active_c = competence[active]
                sizes = 80.0 + 140.0 * active_c
                ax.scatter(
                    active_xy[:, 0],
                    active_xy[:, 1],
                    s=sizes,
                    marker="o",
                    c=active_c,
                    cmap="YlOrRd",
                    vmin=0.0,
                    vmax=1.0,
                    edgecolors="black",
                    linewidths=1.4,
                    zorder=5,
                    label="active competence",
                )
    if highlight_goal_xy is not None:
        highlight_goal_xy = np.asarray(highlight_goal_xy).reshape(-1)
        ax.scatter(
            highlight_goal_xy[0],
            highlight_goal_xy[1],
            s=220,
            marker="s",
            c="cyan",
            edgecolors="black",
            linewidths=1.6,
            zorder=6,
            label="eval goal",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return mesh


def render_context_sweep_figure(
    goal_grid_xy: np.ndarray,
    values: np.ndarray,
    contexts: np.ndarray,
    ts: np.ndarray,
    num_points: int,
    *,
    competence_goals: np.ndarray,
    start_xy=None,
    highlight_goal_xy=None,
    save_path: Optional[str] = None,
    colorbar_label: str = "P(goal)",
    suptitle: str = "Teacher goal distribution vs progressive competence",
):
    """Subplot grid of per-goal values; each panel has its own color scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_contexts = int(values.shape[0])
    ncols = int(math.ceil(math.sqrt(num_contexts)))
    nrows = int(math.ceil(num_contexts / ncols))
    cmap = "Blues"

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 6.0 * nrows),
        squeeze=False,
    )
    for i in range(num_contexts):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        panel_vals = np.asarray(values[i])
        vmax = float(np.max(panel_vals)) if panel_vals.size else 1.0
        if vmax <= 0.0:
            vmax = 1.0
        mesh = _plot_probs_on_ax(
            ax,
            goal_grid_xy,
            values[i],
            num_points,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            start_xy=start_xy,
            competence_goals=competence_goals,
            competence=contexts[i],
            highlight_goal_xy=highlight_goal_xy,
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
        ax.set_title(_format_competence_title(contexts[i]), fontsize=10)

    for j in range(num_contexts, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].axis("off")

    fig.suptitle(
        suptitle,
        fontsize=16,
        y=1.01,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def render_action_sweep_figure(
    goal_grid_xy: np.ndarray,
    values: np.ndarray,
    actions: np.ndarray,
    num_points: int,
    *,
    competence: np.ndarray,
    competence_goals: np.ndarray,
    start_xy=None,
    save_path: Optional[str] = None,
    colorbar_label: str = "P(goal)",
    suptitle: str = "Teacher goal distribution vs student action",
    bootstrap_first: bool = True,
):
    """Subplot grid of teacher distributions for fixed competence, varying actions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_actions = int(values.shape[0])
    ncols = int(math.ceil(math.sqrt(num_actions)))
    nrows = int(math.ceil(num_actions / ncols))
    vmax = float(np.max(values)) if values.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    cmap = "Blues"

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 6.0 * nrows),
        squeeze=False,
    )
    for i in range(num_actions):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        mesh = _plot_probs_on_ax(
            ax,
            goal_grid_xy,
            values[i],
            num_points,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            start_xy=start_xy,
            competence_goals=competence_goals,
            competence=competence,
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
        is_bootstrap = bool(bootstrap_first and i == 0)
        title_idx = 0 if is_bootstrap else (i - 1 if bootstrap_first else i)
        ax.set_title(
            _format_action_title(actions[i], index=title_idx, is_bootstrap=is_bootstrap),
            fontsize=10,
        )

    for j in range(num_actions, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].axis("off")

    fig.suptitle(suptitle, fontsize=16, y=1.01)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def run_context_sweep(args: ContextSweepConfig) -> dict[str, Any]:
    config = load_experiment_config(args.exp_dir)
    competence_vector_scale = _resolve_competence_vector_scale(args, config)
    print(
        f"[context_sweep] competence_vector_scale={competence_vector_scale:g}"
    )
    teacher_train_state = load_teacher_model(
        exp_dir=args.exp_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        use_ema=args.use_teacher_ema,
        use_avg=args.use_teacher_avg,
        use_max_log_sum_competence=args.use_max_log_sum_competence,
    )
    only_on_competence = bool(
        config.get("TEACHER_CONDITION_ONLY_ON_COMPETENCE", False)
    )
    condition_on_action = bool(config.get("CONDITION_TEACHER_ON_ACTION", True))

    norm_obs = None
    bootstrap_action = None
    start_xy = None
    student_train_state = None
    obs_norm = None
    if not only_on_competence:
        obs_norm = load_obs_norm_stats(args.exp_dir, args.checkpoint_dir)
        env, env_params = _make_env(config)
        if condition_on_action:
            student_train_state = load_student_model(
                exp_dir=args.exp_dir,
                checkpoint_dir=args.checkpoint_dir,
                seed=args.seed,
                use_max_log_sum_competence=args.use_max_log_sum_competence,
            )
            norm_obs, bootstrap_action, start_xy = sample_reset_teacher_inputs(
                env,
                env_params,
                student_train_state,
                config,
                obs_norm=obs_norm,
                seed=args.seed,
            )
        else:
            norm_obs, start_xy = sample_reset_obs(
                env,
                env_params,
                config,
                obs_norm=obs_norm,
                seed=args.seed,
            )

    competence_goals = np.asarray(all_possible_goals(), dtype=np.float32)
    num_competence = int(competence_goals.shape[0])
    ranks = _path_ordered_competence_ranks(
        competence_goals, size_scaling=args.size_scaling
    )
    if args.use_one_hot_competence and args.use_random_competence:
        raise ValueError(
            "use_one_hot_competence and use_random_competence are mutually "
            "exclusive; set at most one"
        )
    if args.use_one_hot_competence:
        contexts, ts = build_one_hot_contexts(num_competence, ranks)
        competence_mode_label = "one-hot competence"
    elif args.use_random_competence:
        contexts, ts = build_random_contexts(
            num_competence, args.num_contexts, args.seed
        )
        competence_mode_label = "random competence"
    else:
        contexts, ts = build_progressive_contexts(
            num_competence, ranks, args.num_contexts
        )
        competence_mode_label = "progressive competence"

    goal_grid, num_points = _build_goal_grid(config)
    sweep_action = bootstrap_action if condition_on_action else None
    probs = compute_teacher_probs(
        teacher_train_state,
        config,
        contexts,
        obs=norm_obs,
        action=sweep_action,
        competence_vector_scale=competence_vector_scale,
    )
    zero_competence = np.zeros((1, num_competence), dtype=np.float32)
    zero_probs = compute_teacher_probs(
        teacher_train_state,
        config,
        zero_competence,
        obs=norm_obs,
        action=sweep_action,
        competence_vector_scale=competence_vector_scale,
    )[0]
    kl_vs_zero = kl_vs_zero_competence(probs, zero_probs)
    sample_counts, sample_freqs = sample_teacher_goal_counts(
        teacher_train_state,
        config,
        contexts,
        num_samples=args.num_samples,
        seed=args.seed + 1,
        obs=norm_obs,
        action=sweep_action,
        competence_vector_scale=competence_vector_scale,
    )

    output_dir = args.output_dir or os.path.join(
        args.exp_dir, "teacher_context_sweep"
    )
    os.makedirs(output_dir, exist_ok=True)
    if args.use_max_log_sum_competence:
        ckpt_tag = "max_log_sum_competence"
    elif args.use_teacher_avg:
        ckpt_tag = "avg"
    elif args.use_teacher_ema:
        ckpt_tag = "ema"
    else:
        ckpt_tag = "final"
    if args.use_one_hot_competence:
        ckpt_tag = f"{ckpt_tag}_onehot"
    elif args.use_random_competence:
        ckpt_tag = f"{ckpt_tag}_random"
    png_path = os.path.join(
        output_dir, f"teacher_goal_distribution_contexts_{ckpt_tag}.png"
    )
    samples_png_path = os.path.join(
        output_dir, f"teacher_goal_samples_contexts_{ckpt_tag}.png"
    )
    kl_png_path = os.path.join(
        output_dir, f"teacher_kl_vs_zero_competence_{ckpt_tag}.png"
    )
    npz_path = os.path.join(
        output_dir, f"teacher_goal_distribution_contexts_{ckpt_tag}.npz"
    )

    fig = render_context_sweep_figure(
        goal_grid,
        probs,
        contexts,
        ts,
        num_points,
        competence_goals=competence_goals,
        start_xy=start_xy,
        save_path=png_path,
        colorbar_label="P(goal)",
        suptitle=f"Teacher goal distribution vs {competence_mode_label}",
    )
    import matplotlib.pyplot as plt

    plt.close(fig)

    samples_fig = render_context_sweep_figure(
        goal_grid,
        sample_freqs,
        contexts,
        ts,
        num_points,
        competence_goals=competence_goals,
        start_xy=start_xy,
        save_path=samples_png_path,
        colorbar_label="sample freq",
        suptitle=(
            f"Teacher goal samples vs {competence_mode_label} "
            f"(N={args.num_samples})"
        ),
    )
    plt.close(samples_fig)

    if args.use_one_hot_competence:
        kl_title = r"KL$(P_c \| P_{c=0})$ vs one-hot competence"
        kl_xlabel = r"path-ordered competence index"
    elif args.use_random_competence:
        kl_title = r"KL$(P_c \| P_{c=0})$ vs random competence"
        kl_xlabel = r"random sample index"
    else:
        kl_title = r"KL$(P_c \| P_{c=0})$ vs progressive competence"
        kl_xlabel = r"competence frontier $t$"
    kl_fig = render_kl_vs_zero_figure(
        ts,
        kl_vs_zero,
        save_path=kl_png_path,
        title=kl_title,
        xlabel=kl_xlabel,
    )
    plt.close(kl_fig)

    np.savez(
        npz_path,
        probs=probs,
        sample_counts=sample_counts,
        sample_freqs=sample_freqs,
        num_samples=np.asarray(args.num_samples),
        contexts=contexts,
        ts=ts,
        goal_grid=goal_grid,
        competence_goals=competence_goals,
        ranks=ranks,
        zero_probs=zero_probs,
        kl_vs_zero_competence=kl_vs_zero,
        reset_obs=(
            np.asarray(jax.device_get(norm_obs)) if norm_obs is not None else np.array([])
        ),
        bootstrap_action=(
            np.asarray(jax.device_get(bootstrap_action))
            if bootstrap_action is not None
            else np.array([])
        ),
        start_xy=start_xy if start_xy is not None else np.array([]),
        use_teacher_ema=np.asarray(args.use_teacher_ema),
        use_teacher_avg=np.asarray(args.use_teacher_avg),
        use_max_log_sum_competence=np.asarray(args.use_max_log_sum_competence),
        use_one_hot_competence=np.asarray(args.use_one_hot_competence),
        use_random_competence=np.asarray(args.use_random_competence),
        only_on_competence=np.asarray(only_on_competence),
        condition_on_action=np.asarray(condition_on_action),
        competence_vector_scale=np.asarray(competence_vector_scale),
    )
    print(f"[context_sweep] saved figure to {png_path}")
    print(f"[context_sweep] saved samples figure to {samples_png_path}")
    print(f"[context_sweep] saved KL figure to {kl_png_path}")
    print(f"[context_sweep] saved arrays to {npz_path}")
    print(
        f"[context_sweep] KL vs c=0: "
        f"min={float(np.min(kl_vs_zero)):.4g} "
        f"max={float(np.max(kl_vs_zero)):.4g} "
        f"mean={float(np.mean(kl_vs_zero)):.4g}"
    )
    print(f"[context_sweep] competence mode: {competence_mode_label}")
    if only_on_competence:
        print("[context_sweep] teacher conditioned only on competence vector")
    else:
        action_msg = (
            f"action_norm={float(np.linalg.norm(jax.device_get(bootstrap_action))):.4f}"
            if condition_on_action and bootstrap_action is not None
            else "action not in teacher input"
        )
        print(
            f"[context_sweep] reset start_xy={start_xy.tolist()} "
            f"{action_msg}"
        )
    if args.use_max_log_sum_competence:
        ckpt_label = "max_log_sum_competence"
    elif args.use_teacher_avg:
        ckpt_label = "AVG"
    elif args.use_teacher_ema:
        ckpt_label = "EMA"
    else:
        ckpt_label = "final"
    print(f"[context_sweep] using {ckpt_label} teacher checkpoint")

    result: dict[str, Any] = {
        "png_path": png_path,
        "samples_png_path": samples_png_path,
        "kl_png_path": kl_png_path,
        "npz_path": npz_path,
        "probs": probs,
        "sample_counts": sample_counts,
        "sample_freqs": sample_freqs,
        "zero_probs": zero_probs,
        "kl_vs_zero_competence": kl_vs_zero,
        "contexts": contexts,
    }

    if args.sweep_eval_goals:
        condition_on_goal = bool(config.get("CONDITION_ON_GOAL", False))
        eval_goal_paths: list[dict[str, str]] = []
        for goal_idx, goal_xy in enumerate(competence_goals):
            goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
            goal_dir = os.path.join(
                output_dir,
                f"eval_goal_{goal_idx:02d}_x{goal_xy[0]:g}_y{goal_xy[1]:g}",
            )
            os.makedirs(goal_dir, exist_ok=True)

            goal_bootstrap = bootstrap_action
            if (
                not only_on_competence
                and condition_on_action
                and condition_on_goal
                and student_train_state is not None
                and norm_obs is not None
            ):
                goal_bootstrap = sample_bootstrap_action_for_goal(
                    student_train_state,
                    config,
                    norm_obs,
                    goal_xy,
                    obs_norm=obs_norm,
                    seed=args.seed + 100 + goal_idx,
                )
            goal_sweep_action = (
                goal_bootstrap if condition_on_action else None
            )

            goal_probs = compute_teacher_probs(
                teacher_train_state,
                config,
                contexts,
                obs=norm_obs,
                action=goal_sweep_action,
                competence_vector_scale=competence_vector_scale,
            )
            goal_sample_counts, goal_sample_freqs = sample_teacher_goal_counts(
                teacher_train_state,
                config,
                contexts,
                num_samples=args.num_samples,
                seed=args.seed + 200 + goal_idx,
                obs=norm_obs,
                action=goal_sweep_action,
                competence_vector_scale=competence_vector_scale,
            )

            goal_png = os.path.join(
                goal_dir, f"teacher_goal_distribution_contexts_{ckpt_tag}.png"
            )
            goal_samples_png = os.path.join(
                goal_dir, f"teacher_goal_samples_contexts_{ckpt_tag}.png"
            )
            goal_npz = os.path.join(
                goal_dir, f"teacher_goal_distribution_contexts_{ckpt_tag}.npz"
            )
            goal_label = (
                f"eval goal[{goal_idx}]=({goal_xy[0]:.1f}, {goal_xy[1]:.1f})"
            )
            goal_fig = render_context_sweep_figure(
                goal_grid,
                goal_probs,
                contexts,
                ts,
                num_points,
                competence_goals=competence_goals,
                start_xy=start_xy,
                highlight_goal_xy=goal_xy,
                save_path=goal_png,
                colorbar_label="P(goal)",
                suptitle=(
                    f"Teacher goal distribution vs {competence_mode_label} "
                    f"({goal_label})"
                ),
            )
            plt.close(goal_fig)

            goal_samples_fig = render_context_sweep_figure(
                goal_grid,
                goal_sample_freqs,
                contexts,
                ts,
                num_points,
                competence_goals=competence_goals,
                start_xy=start_xy,
                highlight_goal_xy=goal_xy,
                save_path=goal_samples_png,
                colorbar_label="sample freq",
                suptitle=(
                    f"Teacher goal samples vs {competence_mode_label} "
                    f"({goal_label}, N={args.num_samples})"
                ),
            )
            plt.close(goal_samples_fig)

            np.savez(
                goal_npz,
                probs=goal_probs,
                sample_counts=goal_sample_counts,
                sample_freqs=goal_sample_freqs,
                num_samples=np.asarray(args.num_samples),
                contexts=contexts,
                ts=ts,
                goal_grid=goal_grid,
                competence_goals=competence_goals,
                ranks=ranks,
                eval_goal_idx=np.asarray(goal_idx),
                eval_goal_xy=goal_xy,
                bootstrap_action=(
                    np.asarray(jax.device_get(goal_bootstrap))
                    if goal_bootstrap is not None
                    else np.array([])
                ),
                start_xy=start_xy if start_xy is not None else np.array([]),
            )
            eval_goal_paths.append(
                {
                    "png_path": goal_png,
                    "samples_png_path": goal_samples_png,
                    "npz_path": goal_npz,
                }
            )
            print(
                f"[eval_goal_sweep] goal[{goal_idx}] "
                f"xy=({goal_xy[0]:.1f}, {goal_xy[1]:.1f}) "
                f"saved to {goal_dir}"
            )
        result["eval_goal_paths"] = eval_goal_paths
        print(
            f"[eval_goal_sweep] wrote {len(eval_goal_paths)} "
            f"eval-goal competence sweeps"
        )
    else:
        print("[eval_goal_sweep] skipped (--no-sweep_eval_goals)")

    if only_on_competence or not condition_on_action:
        reason = (
            "teacher conditioned only on competence "
            "(action not in teacher input)"
            if only_on_competence
            else "CONDITION_TEACHER_ON_ACTION=False "
            "(action not in teacher input)"
        )
        print(f"[action_sweep] skipped: {reason}")
        return result

    # Fixed competence for the action sweep (default: mid context panel).
    if args.competence_idx is None:
        competence_idx = int(contexts.shape[0]) // 2
    else:
        competence_idx = int(args.competence_idx)
    if competence_idx < 0 or competence_idx >= contexts.shape[0]:
        raise ValueError(
            f"competence_idx={competence_idx} out of range "
            f"[0, {contexts.shape[0]})"
        )
    fixed_competence = contexts[competence_idx]
    fixed_t = float(ts[competence_idx])

    action_dim = int(np.asarray(jax.device_get(bootstrap_action)).reshape(-1).shape[0])
    actions = sample_uniform_actions(
        action_dim,
        args.num_action_samples,
        seed=args.seed + 2,
        bootstrap_action=bootstrap_action,
    )
    action_probs = compute_teacher_probs_for_actions(
        teacher_train_state,
        config,
        fixed_competence,
        actions,
        obs=norm_obs,
        competence_vector_scale=competence_vector_scale,
    )
    action_sample_counts, action_sample_freqs = sample_teacher_goal_counts_for_actions(
        teacher_train_state,
        config,
        fixed_competence,
        actions,
        num_samples=args.num_samples,
        seed=args.seed + 3,
        obs=norm_obs,
        competence_vector_scale=competence_vector_scale,
    )

    action_png_path = os.path.join(
        output_dir, f"teacher_goal_distribution_actions_{ckpt_tag}.png"
    )
    action_samples_png_path = os.path.join(
        output_dir, f"teacher_goal_samples_actions_{ckpt_tag}.png"
    )
    action_npz_path = os.path.join(
        output_dir, f"teacher_goal_distribution_actions_{ckpt_tag}.npz"
    )

    competence_title = _format_competence_title(fixed_competence)
    if args.use_one_hot_competence:
        fixed_context_label = f"idx={int(fixed_t)}"
    elif args.use_random_competence:
        fixed_context_label = f"sample={int(fixed_t)}"
    else:
        fixed_context_label = f"t={fixed_t:.2f}"
    action_fig = render_action_sweep_figure(
        goal_grid,
        action_probs,
        actions,
        num_points,
        competence=fixed_competence,
        competence_goals=competence_goals,
        start_xy=start_xy,
        save_path=action_png_path,
        colorbar_label="P(goal)",
        suptitle=(
            f"Teacher goal distribution vs student action "
            f"(fixed {competence_title}, {fixed_context_label})"
        ),
        bootstrap_first=True,
    )
    plt.close(action_fig)

    action_samples_fig = render_action_sweep_figure(
        goal_grid,
        action_sample_freqs,
        actions,
        num_points,
        competence=fixed_competence,
        competence_goals=competence_goals,
        start_xy=start_xy,
        save_path=action_samples_png_path,
        colorbar_label="sample freq",
        suptitle=(
            f"Teacher goal samples vs student action "
            f"(fixed {competence_title}, {fixed_context_label}, "
            f"N={args.num_samples})"
        ),
        bootstrap_first=True,
    )
    plt.close(action_samples_fig)

    np.savez(
        action_npz_path,
        probs=action_probs,
        sample_counts=action_sample_counts,
        sample_freqs=action_sample_freqs,
        num_samples=np.asarray(args.num_samples),
        actions=actions,
        competence=fixed_competence,
        competence_idx=np.asarray(competence_idx),
        t=np.asarray(fixed_t),
        goal_grid=goal_grid,
        competence_goals=competence_goals,
        reset_obs=np.asarray(jax.device_get(norm_obs)),
        bootstrap_action=np.asarray(jax.device_get(bootstrap_action)),
        start_xy=start_xy if start_xy is not None else np.array([]),
        use_teacher_ema=np.asarray(args.use_teacher_ema),
        use_teacher_avg=np.asarray(args.use_teacher_avg),
        use_max_log_sum_competence=np.asarray(args.use_max_log_sum_competence),
        use_one_hot_competence=np.asarray(args.use_one_hot_competence),
        use_random_competence=np.asarray(args.use_random_competence),
    )
    print(f"[action_sweep] saved figure to {action_png_path}")
    print(f"[action_sweep] saved samples figure to {action_samples_png_path}")
    print(f"[action_sweep] saved arrays to {action_npz_path}")
    print(
        f"[action_sweep] fixed competence_idx={competence_idx} "
        f"{fixed_context_label} num_actions={actions.shape[0]} "
        f"(bootstrap + {args.num_action_samples} uniform)"
    )

    result.update(
        {
            "action_png_path": action_png_path,
            "action_samples_png_path": action_samples_png_path,
            "action_npz_path": action_npz_path,
            "action_probs": action_probs,
            "action_sample_counts": action_sample_counts,
            "action_sample_freqs": action_sample_freqs,
            "actions": actions,
            "fixed_competence": fixed_competence,
            "competence_idx": competence_idx,
        }
    )
    return result


def main():
    args = tyro.cli(ContextSweepConfig)
    run_context_sweep(args)


if __name__ == "__main__":
    main()
