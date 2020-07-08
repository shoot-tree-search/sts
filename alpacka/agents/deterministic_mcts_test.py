"""Tests for alpacka.agents.deterministic_mcts."""

from alpacka import agents
from alpacka import envs
from alpacka import testing


def test_integration_with_cartpole():
    env = envs.CartPole()
    agent = agents.DeterministicMCTSAgent(n_passes=2)
    network_sig = agent.network_signature(
        env.observation_space, env.action_space
    )
    episode = testing.run_with_dummy_network_prediction(
        agent.solve(env), network_sig)
    assert episode.transition_batch.observation.shape[0]  # pylint: disable=no-member
