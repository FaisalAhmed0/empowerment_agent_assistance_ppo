import os
import xml.etree.ElementTree as ET

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = "r"
GOAL = G = "g"


U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_MANY_STARTS = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, R, 1],
    [1, 1, 1, R, 1],
    [1, G, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_1 = [
    [1, 1, 1, 1, 1],
    [1, R, G, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]
U_MAZE_2 = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]
U_MAZE_3 = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]
U_MAZE_4 = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, 0, 0, G, 1],
    [1, 1, 1, 1, 1],
]


U_MAZE_SKILL_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]
# U_MAZE = [
#     [1, 1, 1, 1, 1],
#     [1, R, R, R, 1],
#     [1, 1, 1, R, 1],
#     [1, G, R, R, 1],
#     [1, 1, 1, 1, 1],
# ]

U_MAZE_NO_GOAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]


U_MAZE_50_PERCENT = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_SINGLE_GOAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, 0, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]


BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, 1, 1, G, G, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, 1, G, G, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_SINGLE_GOAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, G, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, R, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# Alias used by multi-agent maze naming (same grid as BIG_MAZE_SINGLE_GOAL in prototyping notebooks).
BIG_MAZE_MANY_STARTS = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, G, G, 1],
    [1, 0, 0, 1, 0, 0, G, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, R, 0, 0, 1, 0, R, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_HARD_GOALS = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, G, 1, G, 1],
    [1, 1, 1, 1, G, 1, G, 1],
    [1, 1, 1, 1, 1, 1, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, G, G, 1],
    [1, 0, 0, 1, 0, G, G, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, G, 0, 1, G, 1],
    [1, 0, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, 1, G, G, G, G, G, 1],
    [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, G, G, G, G, G, 1, G, G, G, 1],
    [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
    [1, G, G, 1, G, 1, G, G, G, G, G, 1],
    [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

HARDEST_MAZE_50_PERCENT = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, R, G, G, G, 1, 0, 0, 0, 0, 0, 1],
                        [1, G, 1, 1, G, 1, 0, 1, 0, 1, 0, 1],
                        [1, G, G, G, G, G, G, 1, 0, 0, 0, 1],
                        [1, G, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, G, G, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, G, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                        [1, G, G, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]]


HARDEST_MAZE_SINGLE_GOAL = \
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, G, G, G, 1], # goal coordinate is (28, 40)
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE_HARD_GOALS = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 1, 0, 0, 0, 0, G, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, G, 0, 1, 0, 0, 0, 1, 0, 0, G, 1], # goal coordinate is (28, 40)
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]



MAZE_HEIGHT = 0.5


def sample_goals(structure, rng, size_scaling=4):
    '''Sample goals given a maze structure'''
    possible_goals = find_goals(structure, size_scaling)
    idx = jax.random.randint(rng, (1,), 0, len(possible_goals))
    print(possible_goals[idx])
    jax.debug.print("possible_goals: {possible_goals}", possible_goals=possible_goals)
    return jnp.array(possible_goals[idx])[0]
    


def find_starts(structure, size_scaling):
    starts = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                starts.append([i * size_scaling, j * size_scaling])

    return jnp.array(starts)


def find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append([i * size_scaling, j * size_scaling])
    return jnp.array(goals)

def find_walls(structure, size_scaling):
    walls = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == 1:
                walls.append([i * size_scaling, j * size_scaling])
    return jnp.array(walls)

def find_floor(structure, size_scaling):
    floor = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == 0:
                floor.append([i * size_scaling, j * size_scaling])
    return jnp.array(floor)


# Create a xml with maze and a list of possible goal positions
def make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
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
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "ant_maze.xml")

    possible_starts = find_starts(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)
    walls = find_walls(maze_layout, maze_size_scaling)
    floor = find_floor(maze_layout, maze_size_scaling)

    # #TODO: reward to preven the assistant from generating goals inside the walls, this might help with groudning, test this reward function in a simple maze
    # def is_non_wall_location(goal, walls, tol=1e-6):
    #     """
    #     Checks if a goal (x, y) is not inside any wall (returns True if not a wall).
    #     Args:
    #         goal: jnp.ndarray with shape (2,) or list/tuple (x, y)
    #         walls: jnp.ndarray of shape (num_walls, 2)
    #         tol: float, tolerance for numerical precision

    #     Returns:
    #         bool: True if not inside a wall, False otherwise
    #     """
    #     # For each wall, check if goal matches within tolerance any wall position
    #     # `walls` may be empty
    #     if walls.shape[0] == 0:
    #         return True
    #     # Compute pairwise distance to all walls, return True if all dists > tol
    #     dists = jnp.linalg.norm(walls - jnp.asarray(goal), axis=1)
    #     import pdb;pdb.set_trace()
    #     return jnp.all(dists < tol)
    
    # import pdb;pdb.set_trace()
    # all_valid_location = jnp.concatenate([possible_starts, possible_goals, floor], axis=0)



    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

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
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)

    return xml_string, possible_starts, possible_goals


class AntMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=False,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        dense_reward: bool = False,
        skill_mazes: bool = False,
        skill_mazes_eval: bool = True,
        **kwargs,
    ):
        self.skill_mazes = skill_mazes
        self.skill_mazes_eval = skill_mazes_eval
        if self.skill_mazes:
            xml_string, possible_starts, _ = make_maze(maze_layout_name, maze_size_scaling)
            _, _, self.possible_goals_1 = make_maze("u_maze_1", maze_size_scaling)
            _, _, self.possible_goals_2 = make_maze("u_maze_2", maze_size_scaling)
            _, _, self.possible_goals_3 = make_maze("u_maze_3", maze_size_scaling)
            _, _, self.possible_goals_4 = make_maze("u_maze_4", maze_size_scaling)
            self.goals_skills = [self.possible_goals_1, self.possible_goals_2, self.possible_goals_3, self.possible_goals_4]
        elif self.skill_mazes_eval:
            xml_string, possible_starts, possible_goals = make_maze(maze_layout_name, maze_size_scaling)
            self.goals_skills = [possible_goals]
        else:
            xml_string, possible_starts, possible_goals = make_maze(maze_layout_name, maze_size_scaling)
            self.goals_skills = [possible_goals]
        # import pdb;pdb.set_trace()
        # print(possible_goals)
        self.maze_layout_name = maze_layout_name
        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = self.possible_goals_1 if self.skill_mazes else possible_goals
        if self.skill_mazes:
            pass
        else:
            self.maze_layout_name = maze_layout_name
            if "single_goal" in self.maze_layout_name:
                self.hard_goal = jnp.array([12, 4])
            if "u_maze" in self.maze_layout_name:
                self.max_x = 15
                self.min_x = 2.5
            if "big_maze" in self.maze_layout_name:
                self.max_x = 24
                self.min_x = 2.5
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
            # TODO: does the same actuator strength work as in spring
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
        self.state_dim = 29 + 4 if (self.skill_mazes or self.skill_mazes_eval) else 29
        self.goal_indices = jnp.array([0, 1])
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state with a randomly sampled skill."""
        skill_idx = jnp.int32(0)
        if self.skill_mazes:
            rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
            skill_idx = jax.random.randint(rng4, (1,), 0, 4)[0]
        else:
            rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        return self._reset_impl(rng, rng1, rng2, rng3, skill_idx)

    def reset_with_skill(self, rng: jax.Array, skill_idx: jax.Array) -> State:
        """Reset with a specific skill index (JIT-compatible).

        Unlike ``reset``, the skill is not sampled randomly but taken from
        ``skill_idx`` which can be a traced JAX value.  The RNG split pattern
        is kept identical to ``reset`` so downstream randomness is unchanged.
        """
        skill_idx = jnp.int32(skill_idx)
        if self.skill_mazes or self.skill_mazes_eval:
            rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
        else:
            rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        return self._reset_impl(rng, rng1, rng2, rng3, skill_idx)

    def _reset_impl(self, rng, rng1, rng2, rng3, skill_idx) -> State:
        """Shared reset body used by both ``reset`` and ``reset_with_skill``."""
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        start = self._random_start(rng2)
        q = q.at[:2].set(start)

        if self.skill_mazes or self.skill_mazes_eval:
            target = self._random_target_for_skill(rng3, skill_idx)
        else:
            target = self._random_target(rng3, skill_idx)
        q = q.at[-2:].set(target)

        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state, skill_idx)

        if "single_goal" in self.maze_layout_name:
            self.hard_goal = jnp.array([12, 4])

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
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        skill_idx = state.metrics["skill_idx"].astype(jnp.int32)
        old_obs = self._get_obs(pipeline_state0, skill_idx)
        old_dist = jnp.linalg.norm(old_obs[:2] - old_obs[-2:])
        obs = self._get_obs(pipeline_state, skill_idx)
        dist = jnp.linalg.norm(obs[:2] - obs[-2:])
        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0, dtype=float)

        if self.dense_reward:
            reward = 10 * vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = success

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
            current_step=state.metrics["current_step"]+1,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State, skill_index) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        if self.skill_mazes or self.skill_mazes_eval:
            skill_index_onehot = jax.nn.one_hot(jnp.int32(skill_index), 4)
            return jnp.concatenate([qpos] + [qvel] + [skill_index_onehot] + [target_pos] )
        else:
            return jnp.concatenate([qpos] + [qvel] + [target_pos])
        # return jnp.concatenate([qpos] + [qvel] + [target_pos])

    def _random_target(self, rng: jax.Array, skill_idx) -> jax.Array:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]

    def _random_target_for_skill(self, rng: jax.Array, skill_idx) -> jax.Array:
        """Returns a random target from the goal set corresponding to skill_idx (JIT-safe)."""
        branches = []
        for goals in self.goals_skills:
            goals_array = jnp.array(goals)
            def branch_fn(rng, g=goals_array):
                idx = jax.random.randint(rng, (1,), 0, g.shape[0])
                return g[idx[0]]
            branches.append(branch_fn)
        skills_target = jax.lax.switch(jnp.int32(skill_idx), branches, rng)
        # jax.debug.print("skill_idx: {x}", x=skill_idx)
        # jax.debug.print("skills_target: {x}", x=skills_target)
        return skills_target
        

    def _random_start(self, rng: jax.Array) -> jax.Array:
        """Returns a random start location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]
