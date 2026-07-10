import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_agent_trajectory_xy(path):
    data = np.load(path)
    xy = data["agent_xy"]          # (T, 2)
    steps = data["global_step"]    # (T,)
    return xy, steps


def plot_agent_trajectory_xy(xy, steps, save_path, *, title="Agent Trajectory"):
    """Scatter plot of agent (x, y) positions colored by training progress."""
    xy = np.asarray(xy).reshape(-1, 2)
    steps = np.asarray(steps).reshape(-1)
    if len(xy) == 0:
        raise ValueError("Need at least one trajectory point to plot.")

    cmap = plt.get_cmap("Blues")
    step_min = float(steps.min())
    step_max = float(steps.max())
    if step_max > step_min:
        color_values = (steps - step_min) / (step_max - step_min)
    else:
        color_values = np.zeros_like(steps, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color_values,
        cmap=cmap,
        s=4,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("off")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("normalized training steps")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, save_path


if __name__ == "__main__":
    path = "/home/mila/f/faisal.mohamed/git/empowerment_agent_assistance_ppo/purejaxrl/wandb/run-20260710_102938-rz4kbxop/files/purejaxrl_ppo_brax_ant_u_maze_single_goal_agent_trajectory_xy.npz"
    xy, steps = load_agent_trajectory_xy(path)
    print(xy.shape)
    print(steps.shape)
    plot_agent_trajectory_xy(xy, steps, "agent_trajectory.png")
