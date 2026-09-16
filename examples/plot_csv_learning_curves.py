#!/usr/bin/env python3
"""Load CSV experiment logs and plot mean ± std learning curves across seeds.

Example::

    python examples/plot_csv_learning_curves.py \\
      --root $SCRATCH/purejaxrl \\
      --metric eval_episodic_return \\
      --x env_steps \\
      --out figures/eval_return.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keys that identify a single run; exclude when grouping across seeds.
EXCLUDE_GROUP_KEYS = {
    "SEED",
    "run_id",
    "EXP_DIR",
    "GPU_NAME",
    "update",
    "env_steps",
    # Metric columns should never be group keys.
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
    "eval_success_rate",
    "eval_episodic_return",
    "eval_oracle_return",
}

MISSING_HPARAM = "<missing>"
LABEL_PRIORITY = (
    "COMMENT",
    "USE_RND",
    "RND_COEF",
    "USE_ICM",
    "ICM_COEF",
    "LR",
    "ENV_NAME",
    "NUM_ENVS",
    "TOTAL_TIMESTEPS",
)
MAX_LABEL_FIELDS = 4


def load_metrics(
    root: str | os.PathLike[str],
    csv_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    if csv_path is not None:
        path = Path(csv_path).expanduser()
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} row(s) from {path}")
        return df

    root_path = Path(root).expanduser()
    paths = sorted(root_path.rglob("metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv found under {root_path}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df.to_csv("merged_metrics.csv", index=False)
    print(f"Loaded {len(paths)} run(s), {len(df)} row(s) from {root_path}")
    
    return df


def parse_filters(filter_args: Sequence[str] | None) -> dict[str, str]:
    filters: dict[str, str] = {}
    if not filter_args:
        return filters
    for item in filter_args:
        if "=" not in item:
            raise ValueError(f"Filter must be key=value, got: {item}")
        key, value = item.split("=", 1)
        filters[key] = value
    return filters


def apply_filters(df: pd.DataFrame, filters: dict[str, str], env: str | None) -> pd.DataFrame:
    out = df
    if env is not None:
        if "ENV_NAME" not in out.columns:
            raise KeyError("ENV_NAME column missing; cannot filter by --env")
        out = out[out["ENV_NAME"].astype(str) == env]
    for key, value in filters.items():
        if key not in out.columns:
            raise KeyError(f"Filter column {key!r} not in dataframe")
        out = out[out[key].astype(str) == value]
    return out


def hparam_group_columns(df: pd.DataFrame, x_col: str, y_col: str) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_GROUP_KEYS or c in (x_col, y_col):
            continue
        # Skip unnamed / empty
        if c is None or str(c).startswith("Unnamed"):
            continue
        # Constant hyperparameters do not identify different curves. Excluding
        # them also avoids pandas dropping all rows when a constant is NaN.
        if df[c].nunique(dropna=False) > 1:
            cols.append(c)
    return cols


def differing_hparam_label(row: pd.Series, group_cols: Sequence[str], df: pd.DataFrame) -> str:
    """Build a short legend label from hyperparams that actually vary."""
    varying = [c for c in group_cols if df[c].nunique(dropna=False) > 1]
    if not varying:
        return "all runs"

    # Prefer experiment-description and algorithm fields. Long labels can make
    # tight_layout shrink the plotting axes until the figure appears empty.
    ordered = [c for c in LABEL_PRIORITY if c in varying]
    ordered.extend(c for c in varying if c not in ordered)
    selected = ordered[:MAX_LABEL_FIELDS]
    parts = []
    for col in selected:
        value = row[col]
        if value == MISSING_HPARAM:
            value = "default"
        parts.append(f"{col}={value}")
    return ", ".join(parts)


def aggregate_curves(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    if x_col not in df.columns:
        raise KeyError(f"x column {x_col!r} not found")
    if y_col not in df.columns:
        raise KeyError(f"metric {y_col!r} not found")

    work = df.dropna(subset=[y_col, x_col]).copy()
    if work.empty:
        raise ValueError(f"No non-NaN rows for metric {y_col!r}")

    group_cols = hparam_group_columns(work, x_col, y_col)
    # Older pandas versions do not support groupby(dropna=False) and silently
    # discard rows with NaN group keys. A stable sentinel preserves those rows.
    for col in group_cols:
        work[col] = work[col].where(work[col].notna(), MISSING_HPARAM)

    # First combine duplicate runs for the same seed, then compute uncertainty
    # across seeds. This prevents repeated runs of one seed from overweighting
    # the learning curve.
    seed_cols = group_cols + ["SEED", x_col] if "SEED" in work.columns else group_cols + [x_col]
    per_seed = (
        work.groupby(seed_cols)[y_col]
        .mean()
        .reset_index()
    )
    grouped = (
        per_seed.groupby(group_cols + [x_col])[y_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped, group_cols


def plot_curves(
    curve_df: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    x_col: str,
    y_col: str,
    title: str | None = None,
    out: str | os.PathLike[str] | None = None,
    show: bool = False,
    min_seeds: int = 2,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    if group_cols:
        # One series per unique hyperparam combination.
        for keys, sub in curve_df.groupby(list(group_cols)):
            seed_count = int(sub["n"].max())
            if seed_count < min_seeds:
                continue
            if not isinstance(keys, tuple):
                keys = (keys,)
            label_row = pd.Series(dict(zip(group_cols, keys)))
            label = (
                f"{differing_hparam_label(label_row, group_cols, curve_df)} "
                f"(n={seed_count} seeds)"
            )
            sub = sub.sort_values(x_col)
            x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
            mean = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
            std = pd.to_numeric(sub["std"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(mean) & np.isfinite(std)
            ax.plot(x[valid], mean[valid], label=label)
            ax.fill_between(
                x[valid],
                mean[valid] - std[valid],
                mean[valid] + std[valid],
                alpha=0.2,
            )
            plotted += 1
    else:
        sub = curve_df.sort_values(x_col)
        seed_count = int(sub["n"].max())
        if seed_count < min_seeds:
            raise ValueError(
                f"Only {seed_count} seed(s) are available; "
                f"--min-seeds is {min_seeds}"
            )
        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        mean = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
        std = pd.to_numeric(sub["std"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(mean) & np.isfinite(std)
        ax.plot(x[valid], mean[valid], label="mean")
        ax.fill_between(
            x[valid],
            mean[valid] - std[valid],
            mean[valid] + std[valid],
            alpha=0.2,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        max_seeds = int(curve_df["n"].max())
        raise ValueError(
            f"No hyperparameter setting has at least {min_seeds} seeds "
            f"(maximum available: {max_seeds}). Run more seeds or lower "
            "--min-seeds to 1 to plot unreplicated runs."
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} (mean ± std across seeds)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> None:
    default_root = os.path.join(os.environ.get("SCRATCH", "."), "purejaxrl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default=default_root, help="Root dir to search for metrics.csv")
    parser.add_argument("--csv", type=str, default=None, help="Load one merged metrics CSV instead of --root")
    parser.add_argument("--metric", type=str, default="episodic_return", help="Y metric column")
    parser.add_argument("--x", type=str, default="env_steps", help="X axis column")
    parser.add_argument("--env", type=str, default=None, help="Filter ENV_NAME")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Extra filter key=value (repeatable)",
    )
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Display the plot")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=2,
        help="Only plot settings with at least this many aligned seeds (default: 2)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    df = load_metrics(args.root, args.csv)
    filters = parse_filters(args.filter)
    df = apply_filters(df, filters, args.env)
    if df.empty:
        raise SystemExit("No rows left after filtering")

    curve_df, group_cols = aggregate_curves(df, x_col=args.x, y_col=args.metric)
    print(
        f"Aggregated {curve_df['n'].sum()} points into {len(curve_df)} "
        f"(hparam × step) curve points; {curve_df['n'].max()} max seeds/step"
    )
    plot_curves(
        curve_df,
        group_cols,
        x_col=args.x,
        y_col=args.metric,
        title=args.title,
        out=args.out,
        show=args.show,
        min_seeds=args.min_seeds,
    )


if __name__ == "__main__":
    main()
