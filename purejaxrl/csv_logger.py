"""Periodic CSV logging of hyperparameters and train/eval metrics.

Layout per run directory (``EXP_DIR``)::

    hparams.csv   # one row with all hyperparameters
    metrics.csv   # append-only rows: update, env_steps, metrics + hparams

Hyperparameters are duplicated on every metrics row so a single
``pd.concat`` over ``metrics.csv`` files is enough for mean±std
learning curves across seeds.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

# Columns that identify a run / path and should not be used as group keys
# when aggregating across seeds.
RUN_ID_KEYS = ("SEED", "run_id", "EXP_DIR", "GPU_NAME")

# Metric keys written by training scripts (safe CSV headers, no slashes).
TRAIN_METRIC_KEYS: tuple[str, ...] = (
    "episodic_return",
    "total_loss",
    "value_loss",
    "actor_loss",
    "entropy",
    "task_reward_mean",
    "task_reward_std",
    "goal_reward_mean",
    "goal_reward_std",
    "task_reward_running_mean",
    "task_reward_running_std",
    "goal_reward_running_mean",
    "goal_reward_running_std",
    "learning_rate",
    "obs_norm_mean",
    "obs_norm_var",
    "train_goal_success_rate",
    "rnd_reward_mean",
    "rnd_reward_std",
    "rnd_raw_intrinsic_mean",
    "rnd_rew_running_std",
    "rnd_predictor_loss",
    "icm_reward_mean",
    "icm_reward_std",
    "icm_raw_intrinsic_mean",
    "icm_rew_running_std",
    "icm_loss",
)

EVAL_METRIC_KEYS: tuple[str, ...] = (
    "eval_success_rate",
    "eval_episodic_return",
    "eval_oracle_return",
)

CORE_METRIC_KEYS: tuple[str, ...] = ("update", "env_steps") + TRAIN_METRIC_KEYS + EVAL_METRIC_KEYS

_lock = threading.Lock()


def _safe_header(key: str) -> str:
    return str(key).replace("/", "_")


def flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a config dict into CSV-friendly scalar/string values."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        col = _safe_header(key)
        if isinstance(value, (dict, list, tuple)):
            flat[col] = json.dumps(value, sort_keys=True, default=str)
        elif value is None:
            flat[col] = ""
        elif isinstance(value, bool):
            flat[col] = int(value)
        elif isinstance(value, (int, float, str)):
            flat[col] = value
        else:
            flat[col] = str(value)
    return flat


def _write_dict_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def init_run_csvs(
    exp_dir: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    metric_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create ``hparams.csv`` and an empty ``metrics.csv`` with a fixed header.

    Returns the frozen flattened hyperparameters that should be merged into
    every subsequent metrics row (layout C).
    """
    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    run_id = exp_path.name
    # Avoid nesting the frozen hparams dict into itself if re-initialized.
    config_for_flat = {
        k: v for k, v in config.items() if k not in ("CSV_HPARAMS",)
    }
    hparams = flatten_config(config_for_flat)
    hparams["run_id"] = run_id
    hparams["EXP_DIR"] = str(exp_path)

    hparam_keys = list(hparams.keys())
    _write_dict_csv(exp_path / "hparams.csv", hparam_keys, [hparams])

    keys = list(metric_keys) if metric_keys is not None else list(CORE_METRIC_KEYS)
    # Ensure core keys first, then any extra metric keys, then hparams.
    seen: set[str] = set()
    fieldnames: list[str] = []
    for k in list(keys) + hparam_keys:
        if k not in seen:
            fieldnames.append(k)
            seen.add(k)

    metrics_path = exp_path / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Stash header next to the file for appends.
    header_path = exp_path / ".metrics_header.json"
    header_path.write_text(json.dumps(fieldnames))

    return hparams


def append_metrics_row(
    exp_dir: str | os.PathLike[str],
    row: Mapping[str, Any],
    *,
    hparams: Mapping[str, Any] | None = None,
) -> None:
    """Append one metrics row (merged with frozen hparams) and flush."""
    exp_path = Path(exp_dir)
    metrics_path = exp_path / "metrics.csv"
    header_path = exp_path / ".metrics_header.json"

    merged: MutableMapping[str, Any] = {}
    if hparams is not None:
        merged.update(hparams)
    for k, v in row.items():
        merged[_safe_header(k)] = "" if v is None else v

    with _lock:
        if header_path.exists():
            fieldnames = json.loads(header_path.read_text())
        elif metrics_path.exists():
            with metrics_path.open("r", newline="") as f:
                reader = csv.reader(f)
                fieldnames = next(reader)
        else:
            fieldnames = list(merged.keys())
            with metrics_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            header_path.write_text(json.dumps(fieldnames))

        # Drop unknown keys; fill missing with empty string.
        out = {k: merged.get(k, "") for k in fieldnames}
        with metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(out)
            f.flush()
            os.fsync(f.fileno())


def should_log_csv(config: Mapping[str, Any], update_idx: int) -> bool:
    """Return True if this update should write a CSV metrics row."""
    if not bool(config.get("LOG_CSV", True)):
        return False
    freq = int(config.get("CSV_LOG_FREQ", 1))
    if freq <= 0:
        return False
    return int(update_idx) % freq == 0
