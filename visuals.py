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


def _load_brax_html_system(html_path):
    """Extracts the system/trajectory JSON embedded in a brax `html.render` file."""
    import base64
    import json
    import re
    import zlib

    with open(html_path) as f:
        text = f.read()
    match = re.search(r'var system = "([^"]+)"', text)
    if match is None:
        raise ValueError(f"No embedded brax system found in {html_path}")
    return json.loads(zlib.decompress(base64.b64decode(match.group(1))))


def _brax_json_to_mjcf(system, width=1920, height=1080):
    """Builds an MJCF scene with one mocap body per link, holding that link's geoms."""
    fmt = lambda v: " ".join(f"{float(x):.6g}" for x in v)
    geom_types = {"Plane": "plane", "Box": "box", "Sphere": "sphere",
                  "Capsule": "capsule", "Cylinder": "cylinder", "Ellipsoid": "ellipsoid"}

    def geom_xml(g):
        gtype = geom_types.get(g["name"])
        if gtype is None:  # meshes etc. are not supported
            return ""
        size = g["size"]
        material = ""
        if gtype == "plane":
            size = [size[0], size[1], 0.1]
            material = ' material="grid"'
        return (f'<geom type="{gtype}" size="{fmt(size)}" pos="{fmt(g["pos"])}" '
                f'quat="{fmt(g["rot"])}" rgba="{fmt(g["rgba"])}"{material}/>')

    link_names = system["link_names"]
    world = "".join(geom_xml(g) for g in system["geoms"].get("world", []))
    bodies = []
    for i, name in enumerate(link_names):
        name = name or f"link {i}"
        geoms = "".join(geom_xml(g) for g in system["geoms"].get(name, []))
        bodies.append(f'<body name="link{i}" mocap="true">{geoms}</body>')

    return f"""
<mujoco>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
    <global offwidth="{max(width, 1920)}" offheight="{max(height, 1080)}"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.6 0.75 0.9" rgb2="0.2 0.3 0.45" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.42 0.35 0.28" rgb2="0.36 0.30 0.24" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="100 100" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 20" dir="0 0 -1" directional="true" castshadow="true"/>
    {world}
    {"".join(bodies)}
  </worldbody>
</mujoco>"""


def brax_html_to_mp4(
    html_path,
    mp4_path=None,
    *,
    width=640,
    height=480,
    quality=8,
    fps=None,
    camera="track",
    track_link=0,
    distance=5.0,
    azimuth=90.0,
    elevation=-35.0,
    lookat=None,
):
    """Converts a brax `html.render` visualization file into an mp4 video.

    Args:
        html_path: path to the brax html file.
        mp4_path: output path; defaults to `html_path` with a `.mp4` extension.
        width, height: video resolution in pixels.
        quality: encoder quality from 0 (smallest file) to 10 (best).
        fps: frames per second; defaults to real time (1 / env timestep).
        camera: "track" follows `track_link` (the torso by default), "fixed"
            looks at `lookat` (defaults to the centre of the world geoms).
        distance, azimuth, elevation: camera placement (degrees for angles).

    Returns:
        The path of the written mp4 file.
    """
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio
    import mujoco

    system = _load_brax_html_system(html_path)
    states = system["states"]["x"]
    if not states:
        raise ValueError("The html file contains no states.")

    model = mujoco.MjModel.from_xml_string(_brax_json_to_mjcf(system, width, height))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    if camera == "fixed":
        if lookat is None:
            world_pos = [g["pos"] for g in system["geoms"].get("world", []) if g["name"] != "Plane"]
            lookat = np.mean(world_pos, axis=0) if world_pos else np.zeros(3)
        cam.lookat[:] = lookat
    elif camera != "track":
        raise ValueError(f"Unknown camera mode: {camera!r}")

    if mp4_path is None:
        mp4_path = os.path.splitext(html_path)[0] + ".mp4"
    if fps is None:
        fps = 1.0 / system.get("opt", {}).get("timestep", 1.0 / 30)

    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=quality,
                            macro_block_size=1) as writer:
        for x in states:
            data.mocap_pos[:] = np.asarray(x["pos"])
            data.mocap_quat[:] = np.asarray(x["rot"])
            mujoco.mj_forward(model, data)
            if camera == "track":
                cam.lookat[:] = data.mocap_pos[track_link]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    renderer.close()
    return mp4_path




if __name__ == "__main__":
    html_path = "/home/mila/f/faisal.mohamed/git/empowerment_agent_assistance_ppo/agent.html"
    mp4_path = "agent_video.mp4"
    brax_html_to_mp4(html_path, mp4_path, width=1280, height=720, quality=10)