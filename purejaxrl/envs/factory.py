import importlib
import inspect
from typing import Any

from envs.ant import Ant
from envs.ant_ball import AntBall
from envs.ant_ball_maze import AntBallMaze
from envs.ant_maze import AntMaze
from envs.multi_agent_ant_maze import MultiAgentAntMaze
from envs.ant_push import AntPush
from envs.half_cheetah import Halfcheetah
from envs.humanoid import Humanoid
from envs.humanoid_maze import HumanoidMaze
from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
from envs.manipulation.arm_binpick_hard import ArmBinpickHard
from envs.manipulation.arm_grasp import ArmGrasp
from envs.manipulation.arm_push_easy import ArmPushEasy
from envs.manipulation.arm_push_hard import ArmPushHard
from envs.manipulation.arm_reach import ArmReach
from envs.pusher import Pusher, PusherReacher
from envs.pusher2 import Pusher2
from envs.reacher import Reacher
from envs.simple_maze import SimpleMaze


_ENV_CLASS_MAP = {
    "ant": ("ant", "Ant"),
    "ant_push": ("ant_push", "AntPush"),
    "ant_ball": ("ant_ball", "AntBall"),
    "ant_maze": ("ant_maze", "AntMaze"),
    "ant_ball_maze": ("ant_ball_maze", "AntBallMaze"),
    "halfcheetah": ("half_cheetah", "Halfcheetah"),
    "half_cheetah": ("half_cheetah", "Halfcheetah"),
    "humanoid": ("humanoid", "Humanoid"),
    "humanoid_maze": ("humanoid_maze", "HumanoidMaze"),
    "simple_maze": ("simple_maze", "SimpleMaze"),
    "pusher": ("pusher", "Pusher"),
    "pusher_reacher": ("pusher", "PusherReacher"),
    "pusher2": ("pusher2", "Pusher2"),
    "reacher": ("reacher", "Reacher"),
    "arm_reach": ("manipulation.arm_reach", "ArmReach"),
    "arm_grasp": ("manipulation.arm_grasp", "ArmGrasp"),
    "arm_push_easy": ("manipulation.arm_push_easy", "ArmPushEasy"),
    "arm_push_hard": ("manipulation.arm_push_hard", "ArmPushHard"),
    "arm_binpick_easy": ("manipulation.arm_binpick_easy", "ArmBinpickEasy"),
    "arm_binpick_hard": ("manipulation.arm_binpick_hard", "ArmBinpickHard"),
    "arm_binpick_easy_eef": ("manipulation.arm_binpick_easy_EEF", "ArmBinpickEasyEEF"),
}


def list_custom_envs() -> list[str]:
    return sorted(_ENV_CLASS_MAP.keys())


def _filter_supported_kwargs(env_class: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(env_class.__init__)
    parameters = list(signature.parameters.values())[1:]  # skip `self`
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
    if accepts_var_kwargs:
        return kwargs

    allowed_names = {p.name for p in parameters}
    return {k: v for k, v in kwargs.items() if k in allowed_names}


def create_env(env_name: str, backend: str = None, **kwargs) -> object:
    """
    This function creates and returns an appropriate environment object based on the specified environment name and
    backend.

    Args:
        env_name (str): Name of the environment.
        backend (str): Backend to be used for the environment.

    Returns:
        object: The instantiated environment object.

    Raises:
        ValueError: If the specified environment name is unknown.
    """
    if env_name == "reacher":
        env = Reacher(backend=backend or "generalized")
    elif env_name == "ant":
        env = Ant(backend=backend or "spring", dense_reward=True)
    elif env_name == "ant_random_start":
        env = Ant(backend=backend or "spring", randomize_start=True)
    elif env_name == "ant_ball":
        env = AntBall(backend=backend or "spring")
    elif env_name == "ant_push":
        # This is stable only in mjx backend
        assert backend == "mjx" or backend is None
        env = AntPush(backend=backend or "mjx")
    elif "maze" in env_name:
        if "ant_ball" in env_name:
            env = AntBallMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        elif "ant" in env_name:
            if env_name.startswith("ant_multi_"):
                maze = env_name[len("ant_multi_") :]
                env = MultiAgentAntMaze(
                    backend=backend or "spring",
                    maze_layout_name=maze,
                    n_agents=int(kwargs.get("n_agents", 2)),
                    dense_reward=kwargs["dense_reward"]
                )
            else:
                # Possible env_name = {'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze'}
                env = AntMaze(
                    backend=backend or "spring",
                    maze_layout_name=env_name[4:],
                )
        elif "humanoid" in env_name:
            # Possible env_name = {'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze'}
            env = HumanoidMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        else:
            # Possible env_name = {'simple_u_maze', 'simple_big_maze', 'simple_hardest_maze'}
            env = SimpleMaze(backend=backend or "spring", maze_layout_name=env_name[7:])
    elif env_name == "cheetah":
        env = Halfcheetah()
    elif env_name == "pusher_easy":
        env = Pusher(backend=backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env = Pusher(backend=backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env = PusherReacher(backend=backend or "generalized")
    elif env_name == "pusher2":
        env = Pusher2(backend=backend or "generalized")
    elif env_name == "humanoid":
        env = Humanoid(backend=backend or "spring")
    elif env_name == "arm_reach":
        env = ArmReach(backend=backend or "mjx")
    elif env_name == "arm_grasp":
        env = ArmGrasp(backend=backend or "mjx")
    elif env_name == "arm_push_easy":
        env = ArmPushEasy(backend=backend or "mjx")
    elif env_name == "arm_push_hard":
        env = ArmPushHard(backend=backend or "mjx")
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=backend or "mjx")
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=backend or "mjx")
    else:
        print(f"Unknown environment: {env_name} revert to original Brax environments")
        return None
    return env

def make_custom_env(
    env_name: str,
    backend: str | None = None,
    env_kwargs: dict[str, Any] | None = None,
):
    env = create_env(env_name, backend, **env_kwargs)
    return env
