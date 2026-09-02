"""Render experiment visualization MP4s from saved checkpoint artifacts.

Usage:
    python scripts/render_experiment_visuals.py discretion_1613285
    python scripts/render_experiment_visuals.py wrap_640159401 --start 1 --end 10000
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

VisualKind = Literal["agent-positions", "teacher-goals", "empowerment"]
ALL_VISUALS: tuple[VisualKind, ...] = (
    "agent-positions",
    "teacher-goals",
    "empowerment",
)


def load_agent_positions(save_dir: str, start_index: int, last_index: int) -> dict:
    """Load agent positions saved as `{index}_agent_positions.npz`.

    Returns a dict with keys `agent_xy` (concatenated numpy array) and `update_idx`.
    """
    files = []
    for name in os.listdir(save_dir):
        match = re.match(r"^(\d+)_agent_positions\.npz$", name)
        if match:
            files.append((int(match.group(1)), os.path.join(save_dir, name)))

    if not files:
        raise FileNotFoundError(f"No '*_agent_positions.npz' files found in {save_dir}")

    files.sort(key=lambda x: x[0])
    candidates = files[start_index:last_index]

    if not candidates:
        raise FileNotFoundError("No candidate files available")

    agent_xy_list = []
    update_indices = []

    for selected_idx, fname in candidates:
        with np.load(fname) as data:
            agent_xy_list.append(data["agent_xy"].copy())
            update_indices.append(
                int(data["update_idx"]) if "update_idx" in data else selected_idx
            )

    agent_xy = np.concatenate(agent_xy_list, axis=0)

    return {"agent_xy": agent_xy, "update_idx": update_indices}


def _frames_to_mp4(frames, output_path: str, duration: float = 0.5) -> None:
    """Write RGB frames to an mp4 file via imageio/ffmpeg."""
    fps = 1.0 / duration if duration and duration > 0 else 2.0
    target_h, target_w = frames[-1].shape[:2]

    rgb_frames = []
    for img in frames:
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[0] != target_h or img.shape[1] != target_w:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img = np.asarray(pil_img)
        rgb_frames.append(img)
    imageio.mimsave(output_path, rgb_frames, fps=fps)


def _checkpoint_color(uidx, step_min, step_max, cmap) -> np.ndarray:
    if step_max > step_min:
        t = float((uidx - step_min) / (step_max - step_min))
    else:
        t = 0.0
    return np.asarray(cmap(t), dtype=np.float32)


def _rasterize_checkpoint_layer(
    xy,
    color_rgba,
    xlim,
    ylim,
    size,
    point_alpha: float = 0.1,
) -> np.ndarray:
    width, height = size
    layer = np.zeros((height, width, 4), dtype=np.float32)
    if xy.shape[0] == 0:
        return layer

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    if x_span <= 0 or y_span <= 0:
        return layer

    cols = np.clip(
        ((xy[:, 0] - xlim[0]) / x_span * (width - 1)).astype(np.int32),
        0,
        width - 1,
    )
    rows = np.clip(
        ((ylim[1] - xy[:, 1]) / y_span * (height - 1)).astype(np.int32),
        0,
        height - 1,
    )

    r, g, b, _ = color_rgba
    np.add.at(layer[..., 0], (rows, cols), r * point_alpha)
    np.add.at(layer[..., 1], (rows, cols), g * point_alpha)
    np.add.at(layer[..., 2], (rows, cols), b * point_alpha)
    np.add.at(layer[..., 3], (rows, cols), point_alpha)
    np.minimum(layer[..., 3], 1.0, out=layer[..., 3])
    return layer


def _alpha_over(dst, src) -> np.ndarray:
    src_a = np.clip(src[..., 3:4], 0.0, 1.0)
    dst_a = np.clip(dst[..., 3:4], 0.0, 1.0)
    out_a = src_a + dst_a * (1.0 - src_a)
    out_a = np.clip(out_a, 0.0, 1.0)
    out_rgb = np.zeros_like(dst[..., :3])
    mask = out_a[..., 0] > 0
    out_rgb[mask] = (
        src[..., :3][mask] * src_a[mask]
        + dst[..., :3][mask] * dst_a[mask] * (1.0 - src_a[mask])
    ) / out_a[mask]
    result = np.zeros_like(dst)
    result[..., :3] = out_rgb
    result[..., 3:4] = out_a
    return result


def _composite_layers(layers) -> np.ndarray:
    canvas = np.zeros(layers[0].shape, dtype=np.float32)
    for layer in layers:
        canvas = _alpha_over(canvas, layer)
    return canvas


def _layer_frame_to_rgb(frame_rgba) -> np.ndarray:
    alpha = np.clip(frame_rgba[..., 3:4], 0.0, 1.0)
    rgb = frame_rgba[..., :3] * alpha + (1.0 - alpha)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _add_frame_title(rgb_frame, title: str) -> np.ndarray:
    img = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), title, fill=(0, 0, 0))
    return np.asarray(img)


def save_static_agent_scatter(
    save_dir: str,
    start_index: int,
    last_index: int,
    output_path: Optional[str] = None,
) -> str:
    """Save a static scatter PNG of concatenated agent positions."""
    output = load_agent_positions(save_dir, start_index, last_index)
    xy = output["agent_xy"].reshape(-1, 2)

    if output_path is None:
        output_path = os.path.join(save_dir, "agent_trajectory_static.png")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.1, linewidths=0)
    ax.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved static agent scatter to {output_path}")
    return output_path


def load_agent_positions_visuals(
    save_dir: str,
    start_index: int,
    last_index: int,
    output_path: Optional[str] = None,
    duration: float = 0.5,
    render_mode: str = "layers",
    max_points_per_checkpoint: Optional[int] = None,
    image_size: tuple[int, int] = (800, 800),
    dpi: int = 150,
    store_frames: bool = True,
) -> dict:
    """Load agent position checkpoints and create a cumulative scatter MP4 animation."""
    files = []
    for name in os.listdir(save_dir):
        match = re.match(r"^(\d+)_agent_positions\.npz$", name)
        if match:
            files.append((int(match.group(1)), os.path.join(save_dir, name)))

    if not files:
        raise FileNotFoundError(f"No '*_agent_positions.npz' files found in {save_dir}")

    files.sort(key=lambda x: x[0])
    candidates = files[start_index:last_index]

    if not candidates:
        raise FileNotFoundError("No candidate files available")

    checkpoints = []
    update_indices = []
    for selected_idx, fname in candidates:
        with np.load(fname) as data:
            xy = data["agent_xy"].copy().reshape(-1, 2)
            uidx = int(data["update_idx"]) if "update_idx" in data else selected_idx
            if (
                max_points_per_checkpoint is not None
                and xy.shape[0] > max_points_per_checkpoint
            ):
                pick = np.random.choice(
                    xy.shape[0], int(max_points_per_checkpoint), replace=False
                )
                xy = xy[pick]
            checkpoints.append((xy, uidx))
            update_indices.append(uidx)

    all_xy = np.concatenate([xy for xy, _ in checkpoints], axis=0)
    if all_xy.shape[0] == 0:
        raise ValueError("Need at least one trajectory point to plot.")

    padding = 0.05
    x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    x_pad = (x_max - x_min) * padding if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * padding if y_max > y_min else 1.0
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    step_min = float(min(uidx for _, uidx in checkpoints))
    step_max = float(max(uidx for _, uidx in checkpoints))

    cmap = plt.get_cmap("Blues")
    mp4_frames = []
    returned_frames: list[np.ndarray] = [] if store_frames else []

    if render_mode == "layers":
        width, height = image_size
        canvas = np.zeros((height, width, 4), dtype=np.float32)
        for xy, uidx in checkpoints:
            color = _checkpoint_color(uidx, step_min, step_max, cmap)
            layer = _rasterize_checkpoint_layer(
                xy, color, xlim, ylim, (width, height), point_alpha=0.1
            )
            canvas = _alpha_over(canvas, layer)
            rgb = _add_frame_title(
                _layer_frame_to_rgb(canvas), f"Agent Positions (update {uidx})"
            )
            mp4_frames.append(rgb)
            if store_frames:
                returned_frames.append(rgb)
    elif render_mode == "scatter":
        cumulative_xy = []
        cumulative_steps = []
        for xy, uidx in checkpoints:
            cumulative_xy.append(xy)
            cumulative_steps.append(np.full(xy.shape[0], uidx, dtype=np.float64))

            xy_cat = np.concatenate(cumulative_xy, axis=0)
            steps_cat = np.concatenate(cumulative_steps, axis=0)

            if step_max > step_min:
                color_values = (steps_cat - step_min) / (step_max - step_min)
            else:
                color_values = np.zeros_like(steps_cat, dtype=np.float64)

            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(
                xy_cat[:, 0],
                xy_cat[:, 1],
                c=cmap(color_values),
                s=1,
                alpha=0.1,
                linewidths=0,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(f"Agent Positions (update {uidx})")
            ax.axis("off")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            rgb = imageio.imread(buf)
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            mp4_frames.append(rgb)
            if store_frames:
                returned_frames.append(rgb)
    else:
        raise ValueError("render_mode must be 'layers' or 'scatter'")

    if output_path is None:
        output_path = os.path.join(
            save_dir, f"agent_positions_visuals_{start_index}_{last_index}.mp4"
        )

    _frames_to_mp4(mp4_frames, output_path, duration=duration)
    print(f"Saved agent positions visuals to {output_path}")

    return {
        "video_path": output_path,
        "update_idx": update_indices,
        "frames": returned_frames,
    }


def load_teacher_goal_visuals(
    save_dir: str,
    start_index: int,
    last_index: int,
    output_path: Optional[str] = None,
    duration: float = 0.5,
) -> dict:
    """Load teacher goal count images and create an MP4 animation."""
    files = []
    for name in os.listdir(save_dir):
        match = re.match(
            r"^purejaxrl_ppo_brax_ant_u_maze_single_goal_teacher_goal_counts_(\d+)\.png$",
            name,
        )
        if match:
            files.append((int(match.group(1)), os.path.join(save_dir, name)))

    if not files:
        raise FileNotFoundError(
            f"No '*_teacher_goal_counts_*.png' files found in {save_dir}"
        )

    files.sort(key=lambda x: x[0])
    candidates = files[start_index:last_index]

    if not candidates:
        raise FileNotFoundError("No candidate files available")

    update_indices = []
    frames = []
    for selected_idx, fname in candidates:
        frames.append(imageio.imread(fname))
        update_indices.append(selected_idx)

    if output_path is None:
        output_path = os.path.join(
            save_dir, f"teacher_goal_visuals_{start_index}_{last_index}.mp4"
        )

    _frames_to_mp4(frames, output_path, duration=duration)
    print(f"Saved teacher goal visuals to {output_path}")

    return {
        "video_path": output_path,
        "update_idx": update_indices,
        "frames": frames,
    }


def load_empowerment_visuals(
    save_dir: str,
    start_index: int,
    last_index: int,
    output_path: Optional[str] = None,
    duration: float = 0.5,
) -> dict:
    """Load teacher empowerment images and create an MP4 animation."""
    files = []
    for name in os.listdir(save_dir):
        match = re.match(
            r"^purejaxrl_ppo_brax_ant_u_maze_single_goal_teacher_empowerment_grid_(\d+)\.png$",
            name,
        )
        if match:
            files.append((int(match.group(1)), os.path.join(save_dir, name)))

    if not files:
        raise FileNotFoundError(
            f"No '*_teacher_empowerment_grid_*.png' files found in {save_dir}"
        )

    files.sort(key=lambda x: x[0])
    candidates = files[start_index:last_index]

    if not candidates:
        raise FileNotFoundError("No candidate files available")

    update_indices = []
    frames = []
    for selected_idx, fname in candidates:
        frames.append(imageio.imread(fname))
        update_indices.append(selected_idx)

    if output_path is None:
        output_path = os.path.join(
            save_dir, f"teacher_empowerment_visuals_{start_index}_{last_index}.mp4"
        )

    _frames_to_mp4(frames, output_path, duration=duration)
    print(f"Saved teacher empowerment visuals to {output_path}")

    return {
        "video_path": output_path,
        "update_idx": update_indices,
        "frames": frames,
    }


@dataclass(frozen=True)
class ExperimentPaths:
    exp_path: Path
    agent_positions: Path
    teacher_goal_visuals: Path
    teacher_empowerment_visuals: Path


def resolve_experiment_paths(exp_path: Path) -> ExperimentPaths:
    exp_path = exp_path.resolve()
    return ExperimentPaths(
        exp_path=exp_path,
        agent_positions=exp_path / "agent_positions",
        teacher_goal_visuals=exp_path / "teacher_goal_visuals",
        teacher_empowerment_visuals=exp_path / "teacher_empowerment_visuals",
    )


def _handle_visual_error(
    label: str,
    path: Path,
    err: Exception,
    skip_missing: bool,
) -> bool:
    if skip_missing:
        print(f"[skip] {label}: {err}", file=sys.stderr)
        return False
    raise err


def render_all(
    exp_path: Path,
    *,
    start_index: int = 1,
    last_index: int = 10_000_000,
    duration: float = 0.25,
    render_mode: str = "layers",
    max_points_per_checkpoint: Optional[int] = None,
    image_size: tuple[int, int] = (800, 800),
    only: Optional[list[VisualKind]] = None,
    skip_missing: bool = True,
    static_scatter: bool = False,
) -> int:
    """Render selected experiment visuals. Returns number of outputs generated."""
    paths = resolve_experiment_paths(exp_path)
    selected = set(only) if only else set(ALL_VISUALS)
    generated = 0

    if "agent-positions" in selected:
        agent_dir = paths.agent_positions
        try:
            if not agent_dir.is_dir():
                raise FileNotFoundError(f"Directory not found: {agent_dir}")
            load_agent_positions_visuals(
                str(agent_dir),
                start_index,
                last_index,
                duration=duration,
                render_mode=render_mode,
                max_points_per_checkpoint=max_points_per_checkpoint,
                image_size=image_size,
                store_frames=False,
            )
            generated += 1
            if static_scatter:
                save_static_agent_scatter(str(agent_dir), start_index, last_index)
                generated += 1
        except (FileNotFoundError, ValueError) as err:
            if not _handle_visual_error("agent-positions", agent_dir, err, skip_missing):
                pass

    if "teacher-goals" in selected:
        goal_dir = paths.teacher_goal_visuals
        try:
            if not goal_dir.is_dir():
                raise FileNotFoundError(f"Directory not found: {goal_dir}")
            load_teacher_goal_visuals(
                str(goal_dir),
                start_index,
                last_index,
                duration=duration,
            )
            generated += 1
        except (FileNotFoundError, ValueError) as err:
            if not _handle_visual_error("teacher-goals", goal_dir, err, skip_missing):
                pass

    if "empowerment" in selected:
        emp_dir = paths.teacher_empowerment_visuals
        try:
            if not emp_dir.is_dir():
                raise FileNotFoundError(f"Directory not found: {emp_dir}")
            load_empowerment_visuals(
                str(emp_dir),
                start_index,
                last_index,
                duration=duration,
            )
            generated += 1
        except (FileNotFoundError, ValueError) as err:
            if not _handle_visual_error("empowerment", emp_dir, err, skip_missing):
                pass

    return generated


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render experiment visualization MP4s from saved checkpoint artifacts.",
    )
    parser.add_argument(
        "exp_path",
        type=Path,
        help="Path to experiment directory (e.g. discretion_1613285)",
    )
    parser.add_argument("--start", type=int, default=1, help="Slice start index")
    parser.add_argument(
        "--end",
        type=int,
        default=10_000_000,
        help="Slice end index into sorted checkpoint list",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.25,
        help="Seconds per frame in output MP4",
    )
    parser.add_argument(
        "--render-mode",
        choices=("layers", "scatter"),
        default="layers",
        help="Agent positions render mode",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Max points per agent-positions checkpoint (random subsample)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=800,
        help="Square raster size for layer render mode",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=ALL_VISUALS,
        dest="only",
        help="Restrict to specific visual types (repeatable)",
    )
    parser.add_argument(
        "--no-skip-missing",
        action="store_true",
        help="Fail instead of skipping missing visual directories/files",
    )
    parser.add_argument(
        "--static-scatter",
        action="store_true",
        help="Also save static agent trajectory PNG",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if not args.exp_path.exists():
        print(f"Experiment path not found: {args.exp_path}", file=sys.stderr)
        return 1

    generated = render_all(
        args.exp_path,
        start_index=args.start,
        last_index=args.end,
        duration=args.duration,
        render_mode=args.render_mode,
        max_points_per_checkpoint=args.max_points,
        image_size=(args.image_size, args.image_size),
        only=args.only,
        skip_missing=not args.no_skip_missing,
        static_scatter=args.static_scatter,
    )

    if generated == 0:
        print("No visuals were generated.", file=sys.stderr)
        return 1

    print(f"Generated {generated} visual(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
