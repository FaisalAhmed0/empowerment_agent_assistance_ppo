import jax
from brax.envs import PipelineEnv, State, Wrapper
from jax import numpy as jnp


class TeacherGoalSampleWrapper(Wrapper):
    """Wrapper that at reset decides whether the goal for the episode comes from the
    teacher policy or from the environment.

    The teacher is a policy: in actor_step the agent uses it to produce goals. This
    wrapper only sets state.info["teacher_sample"] = 0 when the goal is from the
    environment and 1 when it should come from the teacher policy. The flag is
    preserved across steps so actor_step can use it. The env still produces the
    initial state/obs at reset; the agent overwrites the goal with the teacher
    policy's output in actor_step when teacher_sample == 1.
    """

    def __init__(self, env: PipelineEnv, use_teacher_goal_prob: float = 0.5):
        super().__init__(env)
        self.use_teacher_goal_prob = use_teacher_goal_prob

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        rng, subkey = jax.random.split(rng)
        # Decide per episode: use teacher policy's goal vs environment goal (batch-safe)
        use_teacher_goal = jax.random.uniform(subkey) < self.use_teacher_goal_prob
        teacher_sample = jnp.where(use_teacher_goal, 1, 0)
        if not hasattr(state, "info") or state.info is None:
            state = state.replace(info={})
        state.info["teacher_sample"] = teacher_sample
        return state

    def step(self, state: State, action: jax.Array) -> State:
        teacher_sample = state.info["teacher_sample"]
        state = self.env.step(state, action)
        state.info["teacher_sample"] = teacher_sample
        return state


class FixedSkillWrapper(Wrapper):
    """Wrapper that overrides ``reset`` to use a fixed skill index.

    Must wrap the base environment directly (i.e. the env must expose
    ``reset_with_skill(rng, skill_idx)``).  The skill index is stored as
    a concrete ``jnp.int32`` so it becomes a compile-time constant under
    ``jax.jit`` — no mutable state, fully JIT-safe.
    """

    def __init__(self, env: PipelineEnv, skill_idx: int):
        super().__init__(env)
        self.skill_idx = jnp.int32(skill_idx)

    def reset(self, rng: jax.Array) -> State:
        return self.env.reset_with_skill(rng, self.skill_idx)


class SkillSubsetWrapper(Wrapper):
    """Wrapper that restricts skill sampling to a given subset of indices.

    At each ``reset``, a skill index is drawn uniformly from ``skill_indices``
    and forwarded to ``env.reset_with_skill``.  Fully JIT-safe because the
    index array is a compile-time constant.
    """

    def __init__(self, env: PipelineEnv, skill_indices):
        super().__init__(env)
        self.skill_indices = jnp.array(skill_indices, dtype=jnp.int32)
        self._num_subset_skills = len(skill_indices)

    def reset(self, rng: jax.Array) -> State:
        rng, skill_rng = jax.random.split(rng)
        idx = jax.random.randint(skill_rng, (), 0, self._num_subset_skills)
        skill_idx = self.skill_indices[idx]
        return self.env.reset_with_skill(rng, skill_idx)


class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            traj_id = state.info["traj_id"] + jnp.where(state.info["steps"], 0, 1)
        else:
            traj_id = state.info["traj_id"]
        state = self.env.step(state, action)
        state.info["traj_id"] = traj_id
        return state
