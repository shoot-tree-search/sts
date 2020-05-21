"""Tests for alpacka.batch_steppers.ray."""

import functools
import platform
from unittest import mock

import gin
import pytest
import ray

from alpacka import agents
from alpacka import batch_steppers
from alpacka import envs
from alpacka import networks
from alpacka.batch_steppers import worker_utils


class _TestWorker(worker_utils.Worker):

    def get_state(self):
        return self.env, self.agent, self.network


@pytest.mark.parametrize('compress_episodes', [False, True])
def test_integration_with_cartpole(compress_episodes):
    n_envs = 3

    bs = batch_steppers.RayBatchStepper(
        env_class=envs.CartPole,
        agent_class=agents.RandomAgent,
        network_fn=functools.partial(
            networks.DummyNetwork, network_signature=None
        ),
        n_envs=n_envs,
        output_dir=None,
        compress_episodes=compress_episodes,
    )
    episodes = bs.run_episode_batch(params=None, time_limit=10)

    assert len(episodes) == n_envs
    for episode in episodes:
        assert hasattr(episode, 'transition_batch')


# TODO(xxx): Test ProcessBatchStepper as well (how?).
@mock.patch('alpacka.batch_steppers.worker_utils.Worker', _TestWorker)
@pytest.mark.skipif(platform.system() == 'Darwin',
                    reason='Ray does not work on Mac, see awarelab/alpacka#27')
def test_ray_batch_stepper_worker_members_initialization_with_gin_config():
    # Set up
    solved_at = 7
    env_class = envs.CartPole
    agent_class = agents.RandomAgent
    network_class = networks.DummyNetwork
    n_envs = 3

    gin.bind_parameter('CartPole.solved_at', solved_at)

    env = env_class()
    env.reset()
    root_state = env.clone_state()

    # Run
    bs = batch_steppers.RayBatchStepper(
        env_class=env_class,
        agent_class=agent_class,
        network_fn=functools.partial(network_class, network_signature=None),
        n_envs=n_envs,
        output_dir=None,
    )
    bs.run_episode_batch(params=None, init_state=root_state, time_limit=10)

    # Test
    assert env.solved_at == solved_at
    assert len(bs.workers) == n_envs
    for worker in bs.workers:
        env, agent, network = ray.get(worker.get_state.remote())
        assert isinstance(env, env_class)
        assert isinstance(agent, agent_class)
        assert isinstance(network, network_class)
        assert env.solved_at == solved_at

    # Clean up.
    bs.close()
