"""Process-distributed environment stepper."""

import multiprocessing as mp

from alpacka.batch_steppers import core
from alpacka.batch_steppers import worker_utils


class ProcessBatchStepper(core.BatchStepper):
    """BatchStepper running in multiple processes.

    Runs predictions and steps environments for all Agents separately in their
    own workers.
    """

    def __init__(
        self, env_class, agent_class, network_fn, n_envs, output_dir,
        process_class=mp.Process,
    ):
        super().__init__(env_class, agent_class, network_fn, n_envs, output_dir)

        config = worker_utils.get_config(env_class, agent_class, network_fn)

        def target(worker, queue_in, queue_out):
            while True:
                msg = queue_in.get()
                if msg is None:
                    # None means shutdown.
                    break
                (params, solve_kwargs) = msg
                episode = worker.run(params, solve_kwargs)
                queue_out.put(episode)

        def start_worker():
            worker = worker_utils.Worker(
                env_class=env_class,
                agent_class=agent_class,
                network_fn=network_fn,
                config=config,
                init_hooks=worker_utils.init_hooks,
            )
            queue_in = mp.Queue()
            queue_out = mp.Queue()
            process = process_class(
                target=target, args=(worker, queue_in, queue_out)
            )
            process.start()
            return (queue_in, queue_out)

        self._queues = [start_worker() for _ in range(n_envs)]

    def _run_episode_batch(self, params, solve_kwargs_per_agent):
        for ((queue_in, _), solve_kwargs) in zip(
            self._queues, solve_kwargs_per_agent
        ):
            queue_in.put((params, solve_kwargs))
        return [queue_out.get() for (_, queue_out) in self._queues]

    def close(self):
        # Send a shutdown message to all processes.
        for (queue_in, _) in self._queues:
            queue_in.put(None)
