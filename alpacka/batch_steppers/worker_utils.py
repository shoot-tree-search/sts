"""Utilities for BatchSteppers running in separate workers."""

import gin

from alpacka.batch_steppers import core


init_hooks = []


def register_init_hook(hook):
    """Registers a hook called at the initialization of workers.

    Args:
        hook: callable
    """
    init_hooks.append(hook)


def get_config(env_class, agent_class, network_fn):
    """Returns gin operative config for (at least) env, agent and network.

    It creates env, agent and network to initialize operative gin-config.
    It deletes them afterwords.
    """

    env_class()
    agent_class()
    network_fn()
    return gin.operative_config_str()


class Worker:
    """Class used to step agent-environment-network in a separate worker."""

    def __init__(
        self, env_class, agent_class, network_fn, config, init_hooks,
    ):
        # Limit number of threads used between independent tf.op-s to 1.
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # TODO(xxx): Test that skip_unknown is required!
        gin.parse_config(config, skip_unknown=True)

        for hook in init_hooks:
            hook()

        self.env = env_class()
        self.agent = agent_class()
        self._request_handler = core.RequestHandler(network_fn)

    def run(self, params, solve_kwargs):
        """Runs the episode using the given network parameters."""
        episode_cor = self.agent.solve(self.env, **solve_kwargs)
        return self._request_handler.run_coroutine(episode_cor, params)

    @property
    def network(self):
        return self._request_handler.network
