"""Environment wrappers."""

import collections

import gym
from gym import wrappers


class ModelWrapper(gym.Wrapper):
    """Base class for wrappers intended for use with model-based environments.

    This class defines an additional interface over gym.Wrapper that is assumed
    by model-based agents. It's just for documentation purposes, doesn't have to
    be subclassed by wrappers used with models (but it can be).
    """

    def clone_state(self):
        """Returns the current environment state."""
        return self.env.clone_state()

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        return self.env.restore_state(state)


_TimeLimitWrapperState = collections.namedtuple(
    'TimeLimitWrapperState',
    ['super_state', 'elapsed_steps']
)


class TimeLimitWrapper(wrappers.TimeLimit, ModelWrapper):
    """Model-based TimeLimit gym.Env wrapper."""

    def clone_state(self):
        """Returns the current environment state."""
        assert self._elapsed_steps is not None, (
            'Environment must be reset before the first clone_state().'
        )

        return _TimeLimitWrapperState(
            super_state=super().clone_state(), elapsed_steps=self._elapsed_steps
        )

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        try:
            self._elapsed_steps = state.elapsed_steps
            state = state.super_state
        except AttributeError:
            self._elapsed_steps = 0

        return super().restore_state(state)
