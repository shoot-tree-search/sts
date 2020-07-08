"""Single-threaded environment stepper."""

import numpy as np

from alpacka import data
from alpacka.batch_steppers import core


class _NetworkRequestBatcher:
    """Batches network requests."""

    def __init__(self, requests):
        self._requests = requests
        self._model_request = None

    @property
    def batched_request(self):
        """Determines model request and returns it."""
        if self._model_request is not None:
            return self._model_request
        self._model_request = next(x for x in self._requests if x is not None)
        return self._model_request

    def unbatch_responses(self, x):
        return (x if req is not None else None for req in self._requests)


class _PredictionRequestBatcher:
    """Batches prediction requests."""

    def __init__(self, requests):
        self._requests = requests
        self._n_agents = len(requests)
        self._batched_request = None

    @property
    def batched_request(self):
        """Batches requests and returns batched request."""
        if self._batched_request is not None:
            return self._batched_request

        # Request used as a filler for coroutines that have already
        # finished.
        filler = next(x for x in self._requests if x is not None)
        # Fill with 0s for easier debugging.
        filler = data.nested_map(np.zeros_like, filler)

        # Substitute the filler for Nones.
        self._requests = [x if x is not None else filler
                          for x in self._requests]

        def assert_not_scalar(x):
            assert np.array(x).shape, (
                'All arrays in a PredictRequest must be at least rank 1.'
            )
        data.nested_map(assert_not_scalar, self._requests)

        def flatten_first_2_dims(x):
            return np.reshape(x, (-1,) + x.shape[2:])

        # Stack instead of concatenate to ensure that all requests have
        # the same shape.
        self._batched_request = data.nested_stack(self._requests)
        # (n_agents, n_requests, ...) -> (n_agents * n_requests, ...)
        self._batched_request = data.nested_map(flatten_first_2_dims,
                                                self._batched_request)
        return self._batched_request

    def unbatch_responses(self, x):
        def unflatten_first_2_dims(x):
            return np.reshape(
                x, (self._n_agents, -1) + x.shape[1:]
            )
        # (n_agents * n_requests, ...) -> (n_agents, n_requests, ...)
        return data.nested_unstack(
            data.nested_map(unflatten_first_2_dims, x)
        )


class LocalBatchStepper(core.BatchStepper):
    """Batch stepper running locally.

    Runs batched prediction for all Agents at the same time.
    """

    def __init__(self, env_class, agent_class, network_fn, n_envs, output_dir):
        super().__init__(env_class, agent_class, network_fn, n_envs, output_dir)

        def make_env_and_agent():
            env = env_class()
            agent = agent_class()
            return (env, agent)

        self._envs_and_agents = [make_env_and_agent() for _ in range(n_envs)]
        self._request_handler = core.RequestHandler(network_fn)

    def _get_request_batcher(self, requests):
        """Determines the common type of requests and returns a batcher.

        All requests must of the same type.
        """
        model_request = next(x for x in requests if x is not None)
        if isinstance(model_request, data.NetworkRequest):
            request_batcher = _NetworkRequestBatcher(requests)
        else:
            request_batcher = _PredictionRequestBatcher(requests)
        return request_batcher

    def _batch_coroutines(self, cors):
        """Batches a list of coroutines into one.

        Handles waiting for the slowest coroutine and filling blanks in
        prediction requests.
        """
        # Store the final episodes in a list.
        episodes = [None] * len(cors)

        def store_transitions(i, cor):
            episodes[i] = yield from cor
            # End with an infinite stream of Nones, so we don't have
            # to deal with StopIteration later on.
            while True:
                yield None
        cors = [store_transitions(i, cor) for(i, cor) in enumerate(cors)]

        def all_finished(xs):
            return all(x is None for x in xs)

        requests = [next(cor) for cor in cors]
        while not all_finished(requests):
            batcher = self._get_request_batcher(requests)
            responses = yield batcher.batched_request
            requests = [
                cor.send(inp)
                for cor, inp in zip(cors, batcher.unbatch_responses(responses))
            ]
        return episodes

    def _run_episode_batch(self, params, solve_kwargs_per_agent):
        episode_cor = self._batch_coroutines([
            agent.solve(env, **solve_kwargs)
            for (env, agent), solve_kwargs in
            zip(self._envs_and_agents, solve_kwargs_per_agent)
        ])
        return self._request_handler.run_coroutine(episode_cor, params)
