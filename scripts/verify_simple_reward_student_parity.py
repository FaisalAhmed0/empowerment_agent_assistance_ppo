"""Compare student PPO tensors between simple_reward and no_teacher_reward."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = REPO / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    simple = _load_module(
        "simple_reward_mod",
        "purejaxrl/ppo_continuous_action_custom_brax_with_teacher_simple_reward.py",
    )
    no_teacher = _load_module(
        "no_teacher_mod",
        "purejaxrl/ppo_continuous_action_custom_brax_with_teacher_no_teacher_reward.py",
    )

    shared = {
        "SEED": 42,
        "NUM_ENVS": 8,
        "NUM_STEPS": 4,
        "TOTAL_TIMESTEPS": 8 * 4 * 2,
        "NUM_MINIBATCHES": 2,
        "WANDB_MODE": "disabled",
        "DEBUG": False,
        "SAVE_MODEL": False,
        "EVAL_RENDER_LOG_WANDB_HTML": False,
        "NORMALIZE_ENV": False,
        "CONDITION_TEACHER_ON_COMPETENCE": False,
        "USE_AVERAGE_COMPETENCE_REWARD": False,
        "TEACHER_SOFTMAX_VIZ_LOG_EVERY_UPDATES": 0,
    }
    simple_cfg = asdict(simple.TrainConfig())
    simple_cfg.update(shared)
    no_teacher_cfg = asdict(no_teacher.TrainConfig())
    no_teacher_cfg.update({k: shared[k] for k in shared})

    simple_train = simple.make_train(simple_cfg)
    no_teacher_train = no_teacher.make_train(no_teacher_cfg)
    rng = jax.random.PRNGKey(42)
    simple_out = jax.jit(simple_train[0])(rng)
    no_teacher_out = jax.jit(no_teacher_train[0])(rng)

    simple_metric = jax.device_get(simple_out["metrics"])
    no_teacher_metric = jax.device_get(no_teacher_out["metrics"])

    simple_return = float(np.mean(simple_metric["returned_episode_returns"]))
    no_teacher_return = float(np.mean(no_teacher_metric["returned_episode_returns"]))
    simple_task = float(np.mean(simple_metric.get("task_reward", simple_metric["returned_episode_returns"])))
    # task_reward may only be in debug logs; compare returns from info
    diff = abs(simple_return - no_teacher_return)
    print(f"simple_reward mean return: {simple_return:.6f}")
    print(f"no_teacher mean return:    {no_teacher_return:.6f}")
    print(f"absolute difference:       {diff:.6f}")

    # Teacher sentinel check: student log_prob must differ from teacher goal log_prob
    # when shadowing bug is present they would be identical in distribution to teacher.
    # Static check: ensure env step no longer reassigns student names.
    src = (REPO / "purejaxrl/ppo_continuous_action_custom_brax_with_teacher_simple_reward.py").read_text()
    assert "new_raw_goals, goal_idx, log_prob, value, teacher_obs = _teacher_act" not in src
    assert "episode_done = done[0]" not in src
    assert "teacher_log_prob" in src and "teacher_value" in src
    print("static shadowing checks: OK")

    if diff > 1e-3:
        raise SystemExit(
            f"Student rollout metrics diverge more than expected (diff={diff})"
        )
    print("parity verification passed")


if __name__ == "__main__":
    main()
