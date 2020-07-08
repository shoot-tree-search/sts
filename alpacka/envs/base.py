"""Base classes related to environments."""

import gym


class ModelEnv(gym.Env):
    """Environment interface used by model-based agents.

    This class defines an additional interface over gym.Env that is assumed by
    model-based agents. It's just for documentation purposes, doesn't have to be
    subclassed by envs used as models (but it can be).
    """

    def clone_state(self):
        """Returns the current environment state."""
        raise NotImplementedError

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        raise NotImplementedError


class EnvRenderer:
    """Base class for environment renderers."""

    def __init__(self, env):
        """Initializes EnvRenderer."""
        del env

    def render_state(self, state_info):
        """Renders state_info to an image."""
        raise NotImplementedError

    def render_action(self, action):
        """Renders action to a string."""
        raise NotImplementedError
