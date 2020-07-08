"""Ray-distributed environment stepper."""

import typing

import numpy as np
import ray

from alpacka.batch_steppers import core
from alpacka.batch_steppers import worker_utils


class RayObject(typing.NamedTuple):
    """Keeps value and id of an object in the Ray Object Store."""
    id: typing.Any
    value: typing.Any

    @classmethod
    def from_value(cls, value, weakref=False):
        return cls(ray.put(value, weakref=weakref), value)


class RayBatchStepper(core.BatchStepper):
    """Batch stepper running remotely using Ray.

    Runs predictions and steps environments for all Agents separately in their
    own workers.

    It's highly recommended to pass params to run_episode_batch as a numpy array
    or a collection of numpy arrays. Then each worker can retrieve params with
    zero-copy operation on each node.
    """

    def __init__(self, env_class, agent_class, network_fn, n_envs, output_dir):
        super().__init__(env_class, agent_class, network_fn, n_envs, output_dir)

        config = worker_utils.get_config(env_class, agent_class, network_fn)
        ray_worker_cls = ray.remote(worker_utils.Worker)

        if not ray.is_initialized():
            kwargs = {
                # Size of the Plasma object store, hardcoded to 1GB for now.
                # TODO(xxx): Gin-configure if we ever need to change it.
                'object_store_memory': int(1e9),
            }
            ray.init(**kwargs)
        self.workers = [ray_worker_cls.remote(  # pylint: disable=no-member
            env_class, agent_class, network_fn, config, worker_utils.init_hooks)
            for _ in range(n_envs)]

        self._params = RayObject(None, None)
        self._solve_kwargs_per_worker = [
            RayObject(None, None) for _ in range(self.n_envs)
        ]

    def _run_episode_batch(self, params, solve_kwargs_per_agent):
        # Optimization, don't send the same parameters again.
        if self._params.value is None or not all(
            [np.array_equal(p1, p2)
             for p1, p2 in zip(params, self._params.value)]
        ):
            self._params = RayObject.from_value(params)

        # TODO(xxx): Don't send the same solve kwargs again. This is more
        #           problematic than with params, as values may have very
        #           different types e.g. basic data types or np.ndarray or ???.
        self._solve_kwargs_per_worker = [
            RayObject.from_value(solve_kwargs)
            for solve_kwargs in solve_kwargs_per_agent
        ]

        episodes = ray.get([
            w.run.remote(self._params.id, solve_kwargs.id)
            for w, solve_kwargs in
            zip(self.workers, self._solve_kwargs_per_worker)]
        )
        return episodes
