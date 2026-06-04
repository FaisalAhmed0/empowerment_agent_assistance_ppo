"""Multi-ant maze (N ants, one shared physics target) for CRL + teacher training.

Observation layout (CRL contract):
  ``obs = concat_i block_i``
  where each ``block_i`` is ``[qpos_i, qvel_i, goal_i_xy]``.

Rewards use per-ant distance to that ant's local goal slice.
"""

from __future__ import annotations

import os
import warnings
import xml.etree.ElementTree as ET

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

from envs.ant_maze import (
    BIG_MAZE,
    BIG_MAZE_EVAL,
    BIG_MAZE_SINGLE_GOAL,
    BIG_MAZE_MANY_STARTS,
    GOAL,
    HARDEST_MAZE,
    HARDEST_MAZE_50_PERCENT,
    HARDEST_MAZE_HARD_GOALS,
    HARDEST_MAZE_SINGLE_GOAL,
    MAZE_HEIGHT,
    RESET,
    U_MAZE,
    U_MAZE_MANY_STARTS,
    U_MAZE_1,
    U_MAZE_2,
    U_MAZE_3,
    U_MAZE_4,
    U_MAZE_50_PERCENT,
    U_MAZE_EVAL,
    U_MAZE_SINGLE_GOAL,
    U_MAZE_SKILL_EVAL,
    find_floor,
    find_goals,
    find_starts,
    find_walls,
)
from envs.multi_ant_mjcf import build_multi_ant_maze_root, world_agent_conaffinity_mask


def make_multi_ant_maze_xml(maze_layout_name: str, maze_size_scaling: float, n_agents: int):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "u_maze_many_starts":
        maze_layout = U_MAZE_MANY_STARTS
    elif maze_layout_name == "u_maze_1":
        maze_layout = U_MAZE_1
    elif maze_layout_name == "u_maze_2":
        maze_layout = U_MAZE_2
    elif maze_layout_name == "u_maze_3":
        maze_layout = U_MAZE_3
    elif maze_layout_name == "u_maze_4":
        maze_layout = U_MAZE_4
    elif maze_layout_name == "u_maze_skill_eval":
        maze_layout = U_MAZE_SKILL_EVAL
    elif maze_layout_name == "u_maze_single_goal":
        maze_layout = U_MAZE_SINGLE_GOAL
    elif maze_layout_name == "u_maze_eval":
        maze_layout = U_MAZE_EVAL
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "big_maze_single_goal":
        maze_layout = BIG_MAZE_SINGLE_GOAL
    elif maze_layout_name == "big_maze_eval":
        maze_layout = BIG_MAZE_EVAL
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    elif maze_layout_name == "hardest_maze_50_percent":
        maze_layout = HARDEST_MAZE_50_PERCENT
    elif maze_layout_name == "hardest_maze_single_goal":
        maze_layout = HARDEST_MAZE_SINGLE_GOAL
    elif maze_layout_name == "hardest_maze_hard_goals":
        maze_layout = HARDEST_MAZE_HARD_GOALS
    elif maze_layout_name == "big_maze_many_starts":
        maze_layout = BIG_MAZE_MANY_STARTS
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    possible_starts = find_starts(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)
    _walls = find_walls(maze_layout, maze_size_scaling)
    _floor = find_floor(maze_layout, maze_size_scaling)

    base_root = build_multi_ant_maze_root(n_agents)
    wall_agent_conaff = world_agent_conaffinity_mask(n_agents)
    tree = ET.ElementTree(base_root)
    worldbody = tree.find(".//worldbody")
    if worldbody is None:
        raise RuntimeError("worldbody missing")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f"
                    % (
                        i * maze_size_scaling,
                        j * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f"
                    % (
                        0.5 * maze_size_scaling,
                        0.5 * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity=str(wall_agent_conaff),
                    rgba="0.7 0.5 0.3 1.0",
                )

    rows, cols = len(maze_layout), len(maze_layout[0])
    cx = 0.5 * (rows - 1) * maze_size_scaling
    cy = 0.5 * (cols - 1) * maze_size_scaling
    span_xy = max(rows, cols) * maze_size_scaling
    cam_z = max(22.0, 0.95 * span_xy)
    fovy = min(85.0, 42.0 + 0.5 * span_xy)
    for _cam in list(worldbody.findall("camera")):
        if _cam.get("name") == "maze_top":
            worldbody.remove(_cam)
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "maze_top",
            "mode": "fixed",
            "pos": f"{cx} {cy} {cam_z}",
            "xyaxes": "1 0 0 0 1 0",
            "fovy": str(fovy),
        },
    )

    tree = tree.getroot()
    return ET.tostring(tree), possible_starts, possible_goals


class MultiAgentAntMaze(PipelineEnv):
    """N ants in one maze; CRL obs is concatenated per-agent ``[base_i || goal_i]`` blocks."""

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        dense_reward: bool = False,
        n_agents: int = 2,
        agent_start_offsets=None,
        **kwargs,
    ):
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        self._n_agents = int(n_agents)
        self.num_agents = self._n_agents
        self._qpos_per_agent = 15
        self._qvel_per_agent = 14
        self._act_per_agent = 8
        if agent_start_offsets is None:
            agent_start_offsets = tuple((float(i * 1.5), 0.0) for i in range(n_agents))
        assert len(agent_start_offsets) == n_agents, (
            f"agent_start_offsets must have length n_agents={n_agents}"
        )
        self._agent_start_offsets = jnp.array(agent_start_offsets, dtype=jnp.float32)

        xml_string, possible_starts, possible_goals = make_multi_ant_maze_xml(
            maze_layout_name, maze_size_scaling, n_agents
        )

        self.maze_layout_name = maze_layout_name
        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        n_starts = int(self.possible_starts.shape[0])
        self._use_fixed_start_assignment = n_starts == self._n_agents
        if n_starts != self._n_agents:
            warnings.warn(
                f"MultiAgentAntMaze ({maze_layout_name}): {n_starts} possible start(s) but "
                f"n_agents={self._n_agents}. Start positions will be assigned randomly per reset.",
                stacklevel=2,
            )
        if "single_goal" in self.maze_layout_name:
            self.hard_goal = jnp.array([12, 4])
        if "u_maze" in self.maze_layout_name:
            self.max_x = 15
            self.min_x = 2.5
        if "big_maze" in self.maze_layout_name:
            self.max_x = 28
            self.min_x = 2.0
        if "hardest_maze" in self.maze_layout_name:
            self.max_x = 45
            self.min_x = 2.5

        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        if backend == "positional":
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear)))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self.dense_reward = dense_reward
        # import pdb; pdb.set_trace()

        per_agent_qpos_dim = self._qpos_per_agent - (
            2 if exclude_current_positions_from_observation else 0
        )
        self._per_agent_base_dim = per_agent_qpos_dim + self._qvel_per_agent
        self.per_agent_state_dim = int(self._per_agent_base_dim)
        self.state_dim = self._n_agents * self._per_agent_base_dim
        self._per_agent_obs_dim = self._per_agent_base_dim + 2
        self.goal_indices = jnp.array([0, 1]) # NOTE: this is the goal indicies per agent future observation.
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    @property
    def observation_size(self) -> int:
        """Flat CRL vector: concatenated per-agent ``[state_i, goal_i]`` blocks."""
        return int(self._n_agents * self._per_agent_obs_dim)

    def goal_xy_for_agent(self, obs: jnp.ndarray, agent_index: int) -> jnp.ndarray:
        """Slice ``(2,)`` goal for agent ``agent_index`` from full observation."""
        start = agent_index * self._per_agent_obs_dim + self._per_agent_base_dim
        return jax.lax.dynamic_slice_in_dim(obs, start, 2, axis=-1)

    def local_observation(self, obs: jnp.ndarray, agent_index: int) -> jnp.ndarray:
        """``[base_i || goal_i]`` for student ``agent_index`` (shape ``(..., per_agent_state_dim + 2)``)."""
        start = agent_index * self._per_agent_obs_dim
        return jax.lax.dynamic_slice_in_dim(obs, start, self._per_agent_obs_dim, axis=-1)

    def reset(self, rng: jax.Array) -> State:
        skill_idx = jnp.int32(0)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        return self._reset_impl(rng, rng1, rng2, rng3, skill_idx)

    def reset_with_skill(self, rng: jax.Array, skill_idx: jax.Array) -> State:
        skill_idx = jnp.int32(skill_idx)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        return self._reset_impl(rng, rng1, rng2, rng3, skill_idx)

    def _reset_impl(self, rng, rng1, rng2, rng3, skill_idx) -> State:
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        if self.possible_starts.shape[0] > 1:
            if self._use_fixed_start_assignment:
                # Agent i -> start index (n_agents - 1 - i), e.g. agent 0 at start 1, agent 1 at start 0.
                start_indices = jnp.arange(self._n_agents, dtype=jnp.int32)[::-1]
            else:
                start_indices = jax.random.permutation(rng2, self.possible_starts.shape[0])[
                    : self._n_agents
                ]
            for agent_i in range(self._n_agents):
                base = agent_i * self._qpos_per_agent
                start_i = self.possible_starts[start_indices[agent_i]]
                q = q.at[base : base + 2].set(start_i)
        else:
            start = self._random_start(rng2)
            for agent_i in range(self._n_agents):
                base = agent_i * self._qpos_per_agent
                q = q.at[base : base + 2].set(start)

        target = self._random_target(rng3, skill_idx)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state, skill_idx)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
            "current_step": zero,
            "skill_idx": jnp.float32(skill_idx),
        }
        for i in range(self._n_agents):
            metrics[f"success_agent_{i}"] = zero
            metrics[f"success_easy_agent_{i}"] = zero
            metrics[f"reward_agent_{i}"] = zero
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        q0 = pipeline_state0.q
        q1 = pipeline_state.q

        per_agent_xy0 = jnp.stack(
            [q0[i * self._qpos_per_agent : i * self._qpos_per_agent + 2] for i in range(self._n_agents)]
        )
        per_agent_xy1 = jnp.stack(
            [q1[i * self._qpos_per_agent : i * self._qpos_per_agent + 2] for i in range(self._n_agents)]
        )
        per_agent_z = jnp.stack(
            [q1[i * self._qpos_per_agent + 2] for i in range(self._n_agents)]
        )

        per_agent_vel = (per_agent_xy1 - per_agent_xy0) / self.dt
        forward_reward = jnp.mean(per_agent_vel[:, 0])

        min_z, max_z = self._healthy_z_range
        healthy_per_agent = jnp.where(per_agent_z < min_z, 0.0, 1.0)
        healthy_per_agent = jnp.where(per_agent_z > max_z, 0.0, healthy_per_agent)
        is_healthy = jnp.prod(healthy_per_agent)

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        skill_idx = state.metrics["skill_idx"].astype(jnp.int32)
        obs = self._get_obs(pipeline_state, skill_idx)

        # Goals are packed per agent as part of each local observation block.
        goals = self._goals_from_obs(obs)
        per_agent_dist0 = jnp.linalg.norm(per_agent_xy0 - goals, axis=-1)
        per_agent_dist1 = jnp.linalg.norm(per_agent_xy1 - goals, axis=-1)
        per_agent_success = (per_agent_dist1 < self.goal_reach_thresh).astype(jnp.float32)
        per_agent_success_easy = (per_agent_dist1 < 2.0).astype(jnp.float32)
        success = jnp.mean(per_agent_success)
        success_easy = jnp.mean(per_agent_success_easy)
        dist = jnp.mean(per_agent_dist1)
        vel_to_target = jnp.mean((per_agent_dist0 - per_agent_dist1) / self.dt)

        per_agent_sparse = per_agent_success
        per_agent_vel_to_target = (per_agent_dist0 - per_agent_dist1) / self.dt
        per_agent_actions = action.reshape(self._n_agents, self._act_per_agent)
        per_agent_ctrl = self._ctrl_cost_weight * jnp.sum(
            jnp.square(per_agent_actions), axis=-1
        )
        if self._terminate_when_unhealthy:
            per_agent_healthy_term = healthy_reward
        else:
            per_agent_healthy_term = self._healthy_reward * healthy_per_agent
        per_agent_dense = (
            10 * per_agent_vel_to_target
            + per_agent_healthy_term
            - per_agent_ctrl
            - contact_cost
        )
        per_agent_task_reward = jnp.where(
            self.dense_reward, per_agent_dense, per_agent_sparse
        )
        if self.dense_reward:
            reward = 10 * vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = jnp.mean(per_agent_sparse)

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        mean_xy1 = jnp.mean(per_agent_xy1, axis=0)
        mean_vel = jnp.mean(per_agent_vel, axis=0)
        per_agent_updates = {
            f"success_agent_{i}": per_agent_success[..., i]
            for i in range(self._n_agents)
        }
        per_agent_updates.update(
            {
                f"success_easy_agent_{i}": per_agent_success_easy[..., i]
                for i in range(self._n_agents)
            }
        )
        per_agent_updates.update(
            {
                f"reward_agent_{i}": per_agent_task_reward[..., i]
                for i in range(self._n_agents)
            }
        )
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=mean_xy1[0],
            y_position=mean_xy1[1],
            distance_from_origin=math.safe_norm(mean_xy1),
            x_velocity=mean_vel[0],
            y_velocity=mean_vel[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
            current_step=state.metrics["current_step"] + 1,
            **per_agent_updates,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State, _skill_index) -> jax.Array:
        qpos_all = pipeline_state.q[:-2]
        qvel_all = pipeline_state.qd[:-2]
        physics_target = pipeline_state.x.pos[-1][:2]

        blocks = []
        for i in range(self._n_agents):
            qpos_i = qpos_all[i * self._qpos_per_agent : (i + 1) * self._qpos_per_agent]
            qvel_i = qvel_all[i * self._qvel_per_agent : (i + 1) * self._qvel_per_agent]
            if self._exclude_current_positions_from_observation:
                qpos_i = qpos_i[2:]
            # Initialise each agent's local goal from the shared physics target.
            blocks.append(jnp.concatenate([qpos_i, qvel_i, physics_target]))

        return jnp.concatenate(blocks, axis=-1)

    def _goals_from_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Return goals as ``(..., n_agents, 2)`` from packed per-agent observations."""
        agent_blocks = obs.reshape(obs.shape[:-1] + (self._n_agents, self._per_agent_obs_dim))
        return agent_blocks[..., self._per_agent_base_dim : self._per_agent_base_dim + 2]

    def split_per_agent(self, obs: jnp.ndarray, action: jnp.ndarray):
        """Legacy helper left undefined for this packed per-agent observation layout."""
        raise NotImplementedError("split_per_agent is not defined for MultiAgentAntMaze CRL layout; use local_observation.")

    def _random_target(self, rng: jax.Array, _skill_idx) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]


# Backwards-compatible name used in notebooks
AntMaze = MultiAgentAntMaze
