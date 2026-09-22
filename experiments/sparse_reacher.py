"""Sparse-reward Reacher.

Identical to Brax's Reacher in physics, observations, episode length, action
space and termination.  The ONLY difference is the reward, which follows the
JaxGCRL convention:

    r_t = 1{ d_t < 0.05 },      d_t = || fingertip - target ||

so the episode return is the number of steps spent inside the goal radius
(0..episode_length).  Keeping everything else fixed means the sparse variant
differs from the dense Reacher used elsewhere in the paper by the reward
function alone, which minimises confounds with the algorithm, the environment
physics, the observations and the hyperparameter pipeline.

Importing this module registers two environments:

    env: brax/sparse_reacher        goal radius 0.05  (the JaxGCRL value)
    env: brax/sparse_reacher_hard   goal radius 0.01

The radius sets how sparse the task is.  Measured over a uniform-random policy
(100 episodes x 50 steps), the fraction of steps inside the goal is

    radius 0.05 -> 6.8%   (expected random return 3.4 / 50)
    radius 0.01 -> 0.44%  (expected random return 0.2 / 50)

so 0.05 still hands out a lot of reward by chance, while 0.01 is genuinely
sparse but not unreachable.
"""

import jax
import jax.numpy as jp
from brax import math
from brax.envs import register_environment
from brax.envs.base import State
from brax.envs.reacher import Reacher


class SparseReacher(Reacher):
    """Reacher whose reward is the goal indicator ``1{d_t < goal_radius}``."""

    goal_radius: float = 0.05

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)
        # Brax's Reacher puts the fingertip-to-target vector in the last three
        # entries of the observation, so d_t is its norm.  (The dense reward is
        # -d_t - ||a||^2; we discard it and use the goal indicator instead.)
        dist = math.safe_norm(state.obs[-3:])
        reward = jp.where(dist < self.goal_radius, 1.0, 0.0)
        # NOTE: we deliberately do not add a metrics key here -- Brax requires
        # the metrics pytree to have the same structure at reset and at step.
        return state.replace(reward=reward)


class SparseReacherHard(SparseReacher):
    goal_radius: float = 0.01


register_environment("sparse_reacher", SparseReacher)
register_environment("sparse_reacher_hard", SparseReacherHard)
