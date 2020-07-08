"""Sokoban environments."""

import random

import numpy as np
from gym_sokoban.envs import sokoban_env_fast

from alpacka.envs import base


class Sokoban(sokoban_env_fast.SokobanEnvFast, base.ModelEnv):
    """Sokoban with state clone/restore and returning a "solved" flag.

    Returns observations in one-hot encoding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Return observations as float32, so we don't have to cast them in the
        # network training pipeline.
        self.observation_space.dtype = np.float32

    def reset(self):
        return super().reset().astype(np.float32)

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        return (observation.astype(np.float32), reward, done, info)

    def clone_state(self):
        return self.clone_full_state()

    def restore_state(self, state):
        self.restore_full_state(state)
        return self.render(mode=self.mode)


class ActionNoiseSokoban(Sokoban):
    """Sokoban with randomized actions."""

    def __init__(self, action_noise, *args, **kwargs):
        """Initializes ActionNoiseSokoban.

        Args:
            action_noise: float, how often action passed to step() should be
                replaced by one sampled uniformly from action space.
            args: passed to Sokoban.__init__()
            kwargs: passed to Sokoban.__init__()
        """
        super().__init__(*args, **kwargs)
        self._action_noise = action_noise

    def step(self, action):
        if random.random() < self._action_noise:
            action = self.action_space.sample()
        return super().step(action)
